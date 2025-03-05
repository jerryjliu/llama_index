"""Notion reader."""

import os
import time
from typing import Any, Dict, Optional, Callable
import requests  # type: ignore
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document
from typing import NewType, Iterable, Generator
import datetime
from itertools import chain


# -------------------------------------------------

INTEGRATION_TOKEN_NAME = "NOTION_INTEGRATION_TOKEN"
BLOCK_CHILD_URL_TMPL = "https://api.notion.com/v1/blocks/{block_id}/children"
DATABASE_URL_TMPL = "https://api.notion.com/v1/databases/{database_id}/query"
SEARCH_URL = "https://api.notion.com/v1/search"

PRINT_PREFIX = "Notion reader : "

PAGE_SIZE = 100
PAGE_PREVIEW_LENGTH = 150
DATABASE_PREVIEW_LENGTH = 200
PAGE_SIZE_DICTIONARY = {"page_size": PAGE_SIZE}

json_t = dict[str, Any]
format_json_f = Callable[[json_t], str]
page_id_t = NewType("page_id_t", str)
notion_db_id_t = NewType("notion_db_id_t", str)


# TODO may be a good idea to use https://github.com/ramnes/notion-sdk-py

# -------------------------------------------------
# Notes


# This code has two types of databases
# 1. Notion as a Database
# 2. Notion databases https://www.notion.com/help/intro-to-databases
# make sure not to mix them up


# What is the start_cursor?
#
#   A string that can be used to retrieve the next page of results
#   by passing the value as the start_cursor parameter to the same endpoint.
#   Only available when has_more is true.
#   ie if you have read up to page 10, your start cursor will be page 11
#
#   https://developers.notion.com/reference/intro#pagination
#
#   you should try not to have any custom code to handle pagination, just use library code
#   you may get an error related to start_cursor if you have a previous request fail


# -------------------------------------------------


def iterate_paginated_api(
    function: Callable[..., Any], **kwargs: Any
) -> Generator[json_t, None, None]:
    """Return an iterator over the results of any paginated Notion API."""
    # copy function from ...
    # from notion_client.helpers import iterate_paginated_api
    # there was a bug in the original function, and it removes the dependency

    next_cursor = kwargs.pop("start_cursor", None)

    while True:
        response = function(**kwargs, start_cursor=next_cursor)
        yield from response.get("results", [])

        next_cursor = response.get("next_cursor")
        if not response.get("has_more") or not next_cursor:
            return


class NotionPageReader(BasePydanticReader):
    """Notion Page reader.

    Reads a set of Notion pages.

    Args:
        integration_token (str): Notion integration token.
        print_feedback (bool): Whether to print feedback during operations.

    """

    is_remote: bool = True
    token: str
    headers: Dict[str, str]
    print_feedback: bool = False

    def __init__(
        self, integration_token: Optional[str] = None, print_feedback: bool = False
    ) -> None:
        """Initialize with parameters."""
        if integration_token is None:
            integration_token = os.getenv(INTEGRATION_TOKEN_NAME)
            if integration_token is None:
                raise ValueError(
                    "Must specify `integration_token` or set environment "
                    "variable `NOTION_INTEGRATION_TOKEN`."
                )

        token = integration_token
        headers = {
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }

        super().__init__(token=token, headers=headers)
        self.print_feedback = print_feedback

    @classmethod
    def class_name(cls) -> str:
        """Get the name identifier of the class."""
        return "NotionPageReader"

    def load_data(self, include_pages_in_databases: bool = False) -> list[Document]:
        self._print("load_data")
        return list(
            self.lazy_load_data(include_pages_in_databases=include_pages_in_databases)
        )

    def lazy_load_data(
        self, include_pages_in_databases: bool = False
    ) -> Iterable[Document]:
        self._print("lazy_load_data")
        return self.get_all_pages_and_databases(
            include_pages_in_databases=include_pages_in_databases
        )

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # NETWORK API FUNCTIONS

    def _request_block(self, block_id: str, query_dict: json_t = {}) -> json_t:
        # AI: Helper function to get block data
        block_url = BLOCK_CHILD_URL_TMPL.format(block_id=block_id)

        # Extract start_cursor if present and use it as a query parameter
        params: json_t = {}
        if "start_cursor" in query_dict:
            params["start_cursor"] = query_dict.pop("start_cursor")

        res = self._request_with_retry(
            "GET",
            block_url,
            headers=self.headers,
            params=params,
            json=query_dict,
        )
        return res.json()

    def _request_database(
        self,
        database_id: notion_db_id_t,
        query_dict: json_t,
        start_cursor: Optional[str] = None,
    ) -> json_t:
        self._print("_request_database")

        # start_cursor is ok in this function, you need to pass it into your custom request, DO NOT REMOVE
        # for some reason it is not put into the query_dict by iterate_paginated_api but whatever
        if start_cursor is not None:
            query_dict = {**query_dict, "start_cursor": start_cursor}
        # Don't remove this ^

        res = self._request_with_retry(
            "POST",
            DATABASE_URL_TMPL.format(database_id=database_id),
            headers=self.headers,
            json=query_dict,
        )
        return res.json()

    def _request_search(self, query_dict: json_t) -> json_t:
        # AI: Helper function for search endpoint
        res = self._request_with_retry(
            "POST", SEARCH_URL, headers=self.headers, json=query_dict
        )
        return res.json()

    def page_json_list_to_text(
        self, page_json_iter: Iterable[json_t], num_tabs: int = 0
    ) -> str:
        block_text: str = "\n"
        for result in page_json_iter:
            result_type: str = result["type"]
            result_obj: json_t = result[result_type]

            if "rich_text" in result_obj:
                for rich_text in result_obj["rich_text"]:
                    # skip if doesn't have text object
                    if "text" in rich_text:
                        text: str = rich_text["text"]["content"]
                        prefix: str = "\t" * num_tabs
                        block_text += prefix + text + "\n"

            result_block_id: str = result["id"]
            has_children: bool = result["has_children"]
            if has_children:
                children_text: str = self._read_page_block(
                    result_block_id, num_tabs=num_tabs + 1
                )
                block_text += children_text + "\n"
            block_text += "\n"

        return block_text

    def _read_page_block(self, block_id: str, num_tabs: int = 0) -> str:
        """Read a block."""

        def get_block_next_page(start_cursor: Optional[str]) -> json_t:
            self._print("_read_page_block get page")
            query_dict: json_t = {}

            # TODO ideally this logic could be removed
            if start_cursor and start_cursor != block_id:
                # Only add start_cursor to query_dict if it's a valid pagination cursor
                if len(start_cursor) > 0 and start_cursor != block_id:
                    query_dict["start_cursor"] = start_cursor

            return self._request_block(block_id, query_dict)

        # Iterate through all block results using the paginated API helper
        return self.page_json_list_to_text(
            iterate_paginated_api(get_block_next_page), num_tabs=num_tabs
        )

    def _request_with_retry(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> requests.Response:
        """Make a request with retry and rate limit handling."""
        max_retries = 5
        backoff_factor = 1

        RATE_LIMIT_ERROR_CODE: int = 429

        for attempt in range(max_retries):
            try:
                response: requests.Response = requests.request(
                    method, url, headers=headers, params=params, json=json
                )
                response.raise_for_status()
                return response
            except requests.exceptions.HTTPError:
                if response.status_code == RATE_LIMIT_ERROR_CODE:
                    retry_after = int(response.headers.get("Retry-After", 1))
                    time.sleep(backoff_factor * (2**attempt) + retry_after)
                else:
                    raise requests.exceptions.HTTPError(
                        f"Request failed: {response.text}"
                    )
            except requests.exceptions.RequestException as err:
                raise requests.exceptions.RequestException(f"Request failed: {err}")
        raise Exception("Maximum retries exceeded")

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # OTHER FUNCTIONS

    def get_pages_from_database(
        self, database_id: notion_db_id_t, query_dict: Dict[str, Any] = {}
    ) -> Iterable[json_t]:
        self._print("get_pages_from_database")

        """Get all pages from a database using pagination."""

        # AI: Create a closure that captures database_id and query_dict for pagination
        def get_database_page(start_cursor: Optional[str]) -> json_t:
            self._print("get_pages_from_database -- getting new pages")
            return self._request_database(database_id, query_dict, start_cursor)

        # use an iterator to save memory
        # There may be a type error here, but Iterable[json_t] is correct, may be a bug in iterate_paginated_api?
        return iterate_paginated_api(get_database_page)

    def get_page_ids_from_database(
        self,
        database_id: notion_db_id_t,
        query_dict: Dict[str, Any] = PAGE_SIZE_DICTIONARY,
    ) -> Iterable[page_id_t]:
        """Get all page ids from a database using pagination."""
        self._print("get_page_ids_from_database")
        pages = self.get_pages_from_database(database_id, query_dict)
        return (page["id"] for page in pages)

    def _extract_ids_from_results(self, results: Iterable[json_t]) -> Iterable[str]:
        """Helper function to extract IDs from search/query results."""
        return (result["id"] for result in results)

    def search(self, query: str) -> Iterable[str]:
        self._print("search")
        """Search Notion page given a text query. Return ids"""

        def search_pages(start_cursor: Optional[str]) -> json_t:
            self._print("search_pages -- getting new data")

            query_dict = {"query": query}
            if start_cursor is not None:
                query_dict["start_cursor"] = start_cursor
            return self._request_search(query_dict)

        results: Iterable[json_t] = iterate_paginated_api(search_pages)
        return self._extract_ids_from_results(results)

    def get_page_ids_from_databases(
        self, database_id_list: Iterable[notion_db_id_t]
    ) -> set[page_id_t]:
        self._print("get_page_ids_from_databases")

        page_ids: set[page_id_t] = set()
        for database_id in database_id_list:
            page_ids.update(self.get_page_ids_from_database(database_id))

        return page_ids

    def get_pages_and_databases_from_ids(
        self,
        page_ids: Iterable[page_id_t] = [],
        database_ids: Iterable[notion_db_id_t] = [],
    ) -> Iterable[Document]:
        """Load data from the input directory."""
        self._print("get_pages_and_databases_from_ids")
        return chain(
            self.get_notion_databases(databases=database_ids),
            filter(None, self.get_pages(pages=page_ids)),
        )

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # META FUNCTIONS

    def _list_ids(self, function_name: str, value: str) -> Iterable[str]:
        """Get all of one type of id, for example database or page ids."""

        def search_databases(start_cursor: Optional[str]) -> json_t:
            self._print(f"{function_name} -- getting new data")
            query_dict: json_t = {"filter": {"property": "object", "value": value}}
            if start_cursor is not None:
                query_dict["start_cursor"] = start_cursor

            return self._request_search(query_dict)

        results = iterate_paginated_api(search_databases)
        ids: Iterable[str] = list(self._extract_ids_from_results(results))

        self._print(f"_list_ids {function_name} -- found {len(ids)} {value}s")
        ids = list(set(ids))
        self._print(
            f"_list_ids {function_name} -- found {len(ids)} {value}s which are unique"
        )

        return ids

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # NOTION DATABASES INTERNAL

    @staticmethod
    def default_format_db_json(json_database: json_t) -> str:
        # TODO get title of the database

        database_text: str = "\n Notion Database Start  -----------------\n"
        for row in json_database.get("results", []):
            properties: json_t = row.get("properties", {})

            database_text += "\nNew row\n"
            for prop_name, prop_value in properties.items():
                prop_value: json_t = prop_value

                # this logic remove useless metadata and makes the table human readable compared to json
                from_type: Any = prop_value.get(prop_value["type"], [])
                database_text += prop_name + ",type:" + prop_value["type"] + ",data:"

                # allow most customization in future, although error checking can be difficult
                if prop_value["type"] == "relation":
                    database_text += str(from_type)
                elif prop_value["type"] == "multi_select":
                    database_text += str(from_type)
                elif prop_value["type"] == "rich_text":
                    database_text += str(from_type)
                elif prop_value["type"] == "title":
                    database_text += str(from_type)
                elif prop_value["type"] == "checkbox":
                    database_text += str(from_type)
                elif prop_value["type"] == "url":
                    database_text += str(from_type)
                elif prop_value["type"] == "text":
                    database_text += str(from_type)
                elif prop_value["type"] == "email":
                    database_text += str(from_type)
                else:
                    # print("Unknown type: ", prop_value["type"])
                    database_text += str(prop_value)

                # database_text += "  ORIGINAL "+ str(prop_value) # use this line to see what data is being filtered
                database_text += "\n"

        return database_text + "\nNotion Database End  -----------------\n"

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # NOTION DATABASES

    def get_all_notion_database_ids(self) -> Iterable[notion_db_id_t]:
        """List all databases in the Notion workspace."""
        self._print("get_all_notion_database_ids")
        return (
            notion_db_id_t(id)
            for id in self._list_ids("get_all_notion_database_ids", "database")
        )

    def get_notion_database_text(
        self,
        database_id: notion_db_id_t,
        format_db_json: format_json_f = default_format_db_json,
    ) -> str:
        """Read a database."""
        self._print("get_notion_database_text")
        self._print(f"reading database {database_id}")
        database_data = self._request_database(database_id, {})

        return format_db_json(database_data)

    def get_notion_database(
        self,
        database_id: notion_db_id_t,
        format_db_json: format_json_f = default_format_db_json,
    ) -> Document:
        """Get a database in the Notion workspace."""
        self._print(f"get_notion_database {database_id}")

        database_text = self.get_notion_database_text(
            database_id, format_db_json=format_db_json
        )

        self._print(f"get_notion_database {database_id} -- len {len(database_text)}")
        self._print(
            f"get_notion_database {database_id} -- text {database_text[:DATABASE_PREVIEW_LENGTH]}"
        )

        return Document(
            text=database_text, id_=database_id, extra_info={"database_id": database_id}
        )

    def get_notion_databases(
        self,
        databases: Iterable[notion_db_id_t],
        format_db_json: format_json_f = default_format_db_json,
    ) -> Iterable[Document]:
        """Get all databases in the Notion workspace."""
        self._print("get_notion_databases")
        return (
            self.get_notion_database(database_id, format_db_json=format_db_json)
            for database_id in databases
        )

    def get_all_notion_databases(
        self,
        format_db_json: format_json_f = default_format_db_json,
    ) -> Iterable[Document]:
        """Get all databases in the Notion workspace."""
        self._print("get_all_notion_databases")
        databases = self.get_all_notion_database_ids()
        return self.get_notion_databases(
            databases=databases, format_db_json=format_db_json
        )

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # PAGES

    def get_page_ids_not_in_databases(self) -> Iterable[page_id_t]:
        """List all pages in the Notion workspace."""
        return (
            page_id_t(id)
            for id in self._list_ids("get_page_ids_not_in_databases", "page")
        )

    def get_all_page_ids(self) -> Iterable[page_id_t]:
        """List all pages in the Notion workspace."""
        return chain(
            self.get_page_ids_not_in_databases(),
            self.get_page_ids_from_databases(self.get_all_notion_database_ids()),
        )

    def read_page_text(self, page_id: page_id_t) -> Optional[str]:
        """Read a page."""
        page_text = self._read_page_block(page_id)
        self._print(
            f"Reading page {page_id} of len {len(page_text)}: \n {page_text[:PAGE_PREVIEW_LENGTH]}... \n\n\n"
        )
        return page_text

    @staticmethod
    def _is_page_text_empty(page_text: str | None) -> bool:
        # empty pages don't return ""
        return (
            page_text is None
            or page_text == ""
            or page_text == "\n"
            or page_text == " "
        )

    def get_page(self, page_id: page_id_t) -> Optional[Document]:
        """Get a page in the Notion workspace."""
        page_text = self.read_page_text(page_id)

        if NotionPageReader._is_page_text_empty(page_text):
            self._print(f"WARNING : page {page_id} is empty")
            return None

        return Document(text=page_text, id_=page_id, extra_info={"page_id": page_id})

    def get_pages(self, pages: Iterable[page_id_t]) -> Iterable[Document | None]:
        """Get all pages in the Notion workspace."""
        self._print("get_pages")
        return (self.get_page(page_id) for page_id in pages)

    def get_pages_filter_empty(self, pages: Iterable[page_id_t]) -> Iterable[Document]:
        return filter(None, self.get_pages(pages=pages))

    def get_all_pages_ignore_databases(
        self,
        include_pages_in_databases: bool = False,
    ) -> Iterable[Document]:
        """Get all pages in the Notion workspace."""
        pages: set[page_id_t] = set(self.get_page_ids_not_in_databases())
        if include_pages_in_databases:
            pages.update(
                self.get_page_ids_from_databases(self.get_all_notion_database_ids())
            )

        return filter(None, self.get_pages(pages=pages))

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # CONVENIENCE FUNCTIONS

    def get_all_pages_and_databases(
        self,
        format_db_json: format_json_f = default_format_db_json,
        include_pages_in_databases: bool = False,
    ) -> Iterable[Document]:
        # note :
        # this logic is more complicated than it needs to be,
        # so that you can avoid fetching database ids more than once
        # as these are used in more than on function

        db_ids = self.get_all_notion_database_ids()
        page_ids: set[page_id_t] = set()
        page_ids.update(self.get_page_ids_not_in_databases())

        if include_pages_in_databases:  # there can be a lot of pages inside databases
            page_ids.update(self.get_page_ids_from_databases(db_ids))

        return chain(
            self.get_notion_databases(db_ids), self.get_pages_filter_empty(page_ids)
        )

    def get_all(self) -> Iterable[Document]:
        return self.get_all_pages_and_databases(include_pages_in_databases=True)

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # HELPER FUNCTIONS

    def _print(self, message: str) -> None:
        """Helper method to print feedback messages if print_feedback is enabled.

        Args:
            message (str): The message to print
        """
        if self.print_feedback:
            total_message: str = (
                PRINT_PREFIX + message + datetime.datetime.now().strftime(" %H:%M")
            )

            print(total_message)

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # TESTING

    def _INTERNAL_TEST(self) -> None:
        """
        Used to check that running all functions throws no errors.
        This is a good example of how to use the library.
        """
        # Note : in this function list is used to call the generators

        # there can be a lot of pages in databases, so limiting this can save time
        NUMBER_OF_DBS_TO_EXTRACT_PAGES: int = 1

        # isolate pages with errors
        PROBLEMATIC_PAGE_IDS: list[page_id_t] = [
            page_id_t("149c2e93-a412-80eb-af36-f334f97f1b93"),
            page_id_t("1fe2e39c-bd66-457d-aacb-e4aac3b91920"),
        ]

        list(self.get_pages(PROBLEMATIC_PAGE_IDS))  # call these early
        list(self.get_all_pages_ignore_databases())
        list(self.search(query=""))
        list(self.load_data(include_pages_in_databases=True))
        list(self.get_all_notion_databases())

        database_ids: list[notion_db_id_t] = list(self.get_all_notion_database_ids())
        for db_id in database_ids[:NUMBER_OF_DBS_TO_EXTRACT_PAGES]:
            list(self.get_pages_from_database(database_id=db_id, query_dict={}))
