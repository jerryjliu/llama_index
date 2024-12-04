import logging
import tempfile
from typing import List, Optional, Dict, Any
from pathlib import Path
from abc import abstractmethod

from typing_extensions import Unpack

from llama_index.core.readers import (
    SimpleDirectoryReader,
    FileSystemReaderMixin,
    DirectoryReaderArgs,
)
from llama_index.core.readers.base import (
    BaseReader,
    ResourcesReaderMixin,
)
from llama_index.core.schema import Document

from llama_index.readers.box.BoxAPI.box_api import (
    add_extra_header_to_box_client,
    box_check_connection,
    get_box_files_details,
    get_box_folder_files_details,
    download_file_by_id,
    get_file_content_by_id,
    search_files,
    search_files_by_metadata,
)

from box_sdk_gen import (
    BoxClient,
    SearchForContentScope,
    SearchForContentContentTypes,
    File,
)

from llama_index.readers.box.BoxAPI.box_llama_adaptors import (
    box_file_to_llama_document_metadata,
)

logger = logging.getLogger(__name__)


class BoxReaderBase(BaseReader, ResourcesReaderMixin, FileSystemReaderMixin):
    _box_client: BoxClient

    @classmethod
    def class_name(cls) -> str:
        return "BoxReader"

    def __init__(
        self,
        box_client: BoxClient,
        **kwargs: Unpack[DirectoryReaderArgs],
    ):
        super().__init__(**kwargs)
        self._box_client = add_extra_header_to_box_client(box_client)

    @abstractmethod
    def load_data(
        self,
        *args,
        **kwargs,
    ) -> List[Document]:
        pass

    def load_resource(self, box_file_id: str) -> List[Document]:
        """
        Load data from a specific resource.

        Args:
            resource (str): The resource identifier.

        Returns:
            List[Document]: A list of documents loaded from the resource.
        """
        return self.load_data(file_ids=[box_file_id])

    def get_resource_info(self, box_file_id: str) -> Dict:
        """
        Get information about a specific resource.

        Args:
            resource_id (str): The resource identifier.

        Returns:
            Dict: A dictionary of information about the resource.
        """
        # Connect to Box
        box_check_connection(self._box_client)

        resource = get_box_files_details(
            box_client=self._box_client, file_ids=[box_file_id]
        )

        return resource[0].to_dict()

    def list_resources(
        self,
        folder_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
        is_recursive: bool = False,
    ) -> List[str]:
        """
        Lists the IDs of Box files based on the specified folder or file IDs.

        This method retrieves a list of Box file identifiers based on the provided
        parameters. You can either specify a list of file IDs or a folder ID with an
        optional `is_recursive` flag to include files from sub-folders as well.

        Args:
            folder_id (Optional[str], optional): The ID of the Box folder to list files
                from. If provided, along with `is_recursive` set to True, retrieves data
                from sub-folders as well. Defaults to None.
            file_ids (Optional[List[str]], optional): A list of Box file IDs to retrieve.
                If provided, this takes precedence over `folder_id`. Defaults to None.
            is_recursive (bool, optional): If True and `folder_id` is provided, retrieves
                resource IDs from sub-folders within the specified folder. Defaults to False.

        Returns:
            List[str]: A list containing the IDs of the retrieved Box files.
        """
        # Connect to Box
        box_check_connection(self._box_client)

        # Get the file resources
        box_files: List[File] = []
        if file_ids is not None:
            box_files.extend(
                get_box_files_details(box_client=self._box_client, file_ids=file_ids)
            )
        elif folder_id is not None:
            box_files.extend(
                get_box_folder_files_details(
                    box_client=self._box_client,
                    folder_id=folder_id,
                    is_recursive=is_recursive,
                )
            )
        return [file.id for file in box_files]

    def read_file_content(self, input_file: Path, **kwargs) -> bytes:
        file_id = input_file.name
        return get_file_content_by_id(box_client=self._box_client, box_file_id=file_id)

    def search_resources(
        self,
        query: Optional[str] = None,
        scope: Optional[SearchForContentScope] = None,
        file_extensions: Optional[List[str]] = None,
        created_at_range: Optional[List[str]] = None,
        updated_at_range: Optional[List[str]] = None,
        size_range: Optional[List[int]] = None,
        owner_user_ids: Optional[List[str]] = None,
        recent_updater_user_ids: Optional[List[str]] = None,
        ancestor_folder_ids: Optional[List[str]] = None,
        content_types: Optional[List[SearchForContentContentTypes]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[str]:
        """
        Searches for Box resources based on specified criteria and returns a list of their IDs.

        This method utilizes the Box API search functionality to find resources
        matching the provided parameters. It then returns a list containing the IDs
        of the found resources.

        Args:
            query (Optional[str], optional): A search query string. Defaults to None.
            scope (Optional[SearchForContentScope], optional): The scope of the search.
                Defaults to None.
            file_extensions (Optional[List[str]], optional): A list of file extensions
                to filter by. Defaults to None.
            created_at_range (Optional[List[str]], optional): A list representing a date
                range for file creation time. Defaults to None.
            updated_at_range (Optional[List[str]], optional): A list representing a date
                range for file update time. Defaults to None.
            size_range (Optional[List[int]], optional): A list representing a size range
                for files. Defaults to None.
            owner_user_ids (Optional[List[str]], optional): A list of user IDs to filter
                by owner. Defaults to None.
            recent_updater_user_ids (Optional[List[str]], optional): A list of user IDs to
                filter by recent updater. Defaults to None.
            ancestor_folder_ids (Optional[List[str]], optional): A list of folder IDs to
                search within. Defaults to None.
            content_types (Optional[List[SearchForContentContentTypes]], optional): A list
                of content types to filter by. Defaults to None.
            limit (Optional[int], optional): The maximum number of results to return.
                Defaults to None.
            offset (Optional[int], optional): The number of results to skip before
                starting to collect. Defaults to None.

        Returns:
            List[str]: A list of Box resource IDs matching the search criteria.
        """
        # Connect to Box
        box_check_connection(self._box_client)

        box_files = search_files(
            box_client=self._box_client,
            query=query,
            scope=scope,
            file_extensions=file_extensions,
            created_at_range=created_at_range,
            updated_at_range=updated_at_range,
            size_range=size_range,
            owner_user_ids=owner_user_ids,
            recent_updater_user_ids=recent_updater_user_ids,
            ancestor_folder_ids=ancestor_folder_ids,
            content_types=content_types,
            limit=limit,
            offset=offset,
        )
        return [box_file.id for box_file in box_files]

    def search_resources_by_metadata(
        self,
        from_: str,
        ancestor_folder_id: str,
        query: Optional[str] = None,
        query_params: Optional[Dict[str, str]] = None,
        limit: Optional[int] = None,
        marker: Optional[str] = None,
    ) -> List[str]:
        """
        Searches for Box resources based on metadata and returns a list of their IDs.

        This method utilizes the Box API search functionality to find resources
        matching the provided metadata query. It then returns a list containing the IDs
        of the found resources.

        Args:
            box_client (BoxClient): An authenticated Box client object used
                for interacting with the Box API.
            from_ (str): The metadata template key to search from.
            ancestor_folder_id (str): The ID of the Box folder to search within.
            query (Optional[str], optional): A search query string. Defaults to None.
            query_params (Optional[Dict[str, str]], optional): Additional query parameters
                to filter the search results. Defaults to None.
            limit (Optional[int], optional): The maximum number of results to return.
                Defaults to None.
            marker (Optional[str], optional): The marker for the start of the next page of
                results. Defaults to None.

        Returns:
            List[str]: A list of Box resource IDs matching the search criteria.
        """
        # Connect to Box
        box_check_connection(self._box_client)

        box_files = search_files_by_metadata(
            box_client=self._box_client,
            from_=from_,
            ancestor_folder_id=ancestor_folder_id,
            query=query,
            query_params=query_params,
            limit=limit,
            marker=marker,
        )
        return [box_file.id for box_file in box_files]


class BoxReader(BoxReaderBase):
    """
    A reader class for loading data from Box files.

    This class inherits from the BaseReader class and provides functionality
    to retrieve, download, and process data from Box files. It utilizes the
    provided BoxClient object to interact with the Box API and can optionally
    leverage a user-defined file extractor for more complex file formats.

    Attributes:
        _box_client (BoxClient): An authenticated Box client object used
            for interacting with the Box API.
        **kwargs: Additional keyword arguments passed to SimpleDirectoryReader.
    """

    def __init__(
        self,
        box_client: BoxClient,
        **kwargs: Unpack[DirectoryReaderArgs],
    ):
        super().__init__(box_client=box_client, **kwargs)

    def load_data(
        self,
        folder_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
        is_recursive: bool = False,
    ) -> List[Document]:
        """
        Loads data from Box files into a list of Document objects.

        This method retrieves Box files based on the provided parameters and
        processes them into a structured format using a SimpleDirectoryReader.

        Args:
            self (BoxDataHandler): An instance of the BoxDataHandler class.
            folder_id (Optional[str], optional): The ID of the Box folder to load
                data from. If provided, along with is_recursive set to True, retrieves
                data from sub-folders as well. Defaults to None.
            file_ids (Optional[List[str]], optional): A list of Box file IDs to
                load data from. If provided, folder_id is ignored. Defaults to None.
            is_recursive (bool, optional): If True and folder_id is provided, retrieves
                data from sub-folders within the specified folder. Defaults to False.

        Returns:
            List[Document]: A list of Document objects containing the processed data
                extracted from the Box files.

        Raises:
            BoxAPIError: If an error occurs while interacting with the Box API.
        """
        # Connect to Box
        box_check_connection(self._box_client)

        # Get the file resources
        box_files: List[File] = []
        if file_ids is not None:
            box_files.extend(
                get_box_files_details(box_client=self._box_client, file_ids=file_ids)
            )
        elif folder_id is not None:
            box_files.extend(
                get_box_folder_files_details(
                    box_client=self._box_client,
                    folder_id=folder_id,
                    is_recursive=is_recursive,
                )
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            box_files_with_path = self._download_files(box_files, temp_dir)

            file_name_to_metadata = {
                file.downloaded_file_path: box_file_to_llama_document_metadata(file)
                for file in box_files_with_path
            }

            def get_metadata(filename: str) -> Any:
                return file_name_to_metadata[filename]

            simple_loader = SimpleDirectoryReader(
                input_dir=temp_dir,
                file_metadata=get_metadata,
                **self.model_dump(),
            )
            return simple_loader.load_data()

    def _download_files(self, box_files: List[File], temp_dir: str) -> List[File]:
        """
        Downloads Box files and updates the corresponding payloads with local paths.

        This internal helper function iterates through the provided payloads,
        downloads each file referenced by the payload's resource_info attribute
        to the specified temporary directory, and updates the downloaded_file_path
        attribute of the payload with the local file path.

        Args:
            self (BoxReader): An instance of the BoxReader class.
            payloads (List[_BoxResourcePayload]): A list of _BoxResourcePayload objects
                containing information about Box files.
            temp_dir (str): The path to the temporary directory where the files will
                be downloaded.

        Returns:
            List[_BoxResourcePayload]: The updated list of _BoxResourcePayload objects
                with the downloaded_file_path attribute set for each payload.
        """
        box_files_with_path: List[File] = []

        for file in box_files:
            local_path = download_file_by_id(
                box_client=self._box_client, box_file=file, temp_dir=temp_dir
            )
            file.downloaded_file_path = local_path
            box_files_with_path.append(file)
        return box_files_with_path
