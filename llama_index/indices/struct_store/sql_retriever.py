"""SQL Retriever."""

from abc import ABC, abstractmethod
from enum import Enum
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.query.schema import QueryBundle
from llama_index.schema import NodeWithScore, TextNode
from typing import List, Optional, Any, Union, Callable
from llama_index.utilities.sql_wrapper import SQLDatabase
from llama_index.prompts import BasePromptTemplate, PromptTemplate
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.default_prompts import (
    DEFAULT_TEXT_TO_SQL_PGVECTOR_PROMPT,
    DEFAULT_TEXT_TO_SQL_PROMPT,
)
from llama_index.prompts.prompt_type import PromptType
from sqlalchemy import Table
import logging
from llama_index.objects.base import ObjectRetriever
from llama_index.objects.table_node_mapping import SQLTableSchema

logger = logging.getLogger(__name__)

class SQLRetriever(BaseRetriever):
    """SQL Retriever.

    Retrieves via raw SQL statements.

    Args:
        sql_database (SQLDatabase): SQL database.
        return_raw (bool): Whether to return raw results or format results.
            Defaults to True.

    """

    def __init__(
        self,
        sql_database: SQLDatabase,
        return_raw: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._sql_database = sql_database
        self._return_raw = return_raw

    def _format_node_results(self, results: List[List[Any]], col_keys: List[str]) -> List[NodeWithScore]:
        """Format node results."""
        nodes = []
        for result in results:
            # associate column keys with result tuple
            metadata = dict(zip(col_keys, result))
            # NOTE: leave text field blank for now
            text_node = TextNode(
                text="",
                metadata=metadata,
            )
            nodes.append(NodeWithScore(node=text_node))
        return nodes

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        raw_response_str, metadata = self._sql_database.run_sql(query_bundle.query_str)
        if self._return_raw:
            return raw_response_str
        else:
            # return formatted 
            results = metadata["result"]
            col_keys = metadata["col_keys"]
            return self._format_node_results(results, col_keys)


class SQLParserMode(str, Enum):
    """SQL Parser Mode."""
    DEFAULT = "default"
    PGVECTOR = "pgvector"


class BaseSQLParser(ABC):
    """Base SQL Parser."""
    @abstractmethod
    def parse_response_to_sql(self, response: str, query_bundle: QueryBundle) -> str:
        """Parse response to SQL."""


class DefaultSQLParser(BaseSQLParser):
    """Default SQL Parser."""

    def parse_response_to_sql(self, response: str, query_bundle: QueryBundle) -> str:
        """Parse response to SQL."""
        sql_query_start = response.find("SQLQuery:")
        if sql_query_start != -1:
            response = response[sql_query_start:]
            # TODO: move to removeprefix after Python 3.9+
            if response.startswith("SQLQuery:"):
                response = response[len("SQLQuery:") :]
        sql_result_start = response.find("SQLResult:")
        if sql_result_start != -1:
            response = response[:sql_result_start]
        return response.strip().strip("```").strip()


class PGVectorSQLParser(BaseSQLParser):
    """PGVector SQL Parser."""

    def __init__(self, embed_model: str) -> None:
        """Initialize params."""
        self._embed_model = embed_model

    def parse_response_to_sql(self, response: str, query_bundle: QueryBundle) -> str:
        """Parse response to SQL."""
        sql_query_start = response.find("SQLQuery:")
        if sql_query_start != -1:
            response = response[sql_query_start:]
            # TODO: move to removeprefix after Python 3.9+
            if response.startswith("SQLQuery:"):
                response = response[len("SQLQuery:") :]
        sql_result_start = response.find("SQLResult:")
        if sql_result_start != -1:
            response = response[:sql_result_start]

        # this gets you the sql string with [query_vector] placeholders
        raw_sql_str = response.strip().strip("```").strip()
        query_embedding = self._service_context.embed_model.get_query_embedding(
            query_bundle.query_str
        )
        query_embedding_str = str(query_embedding)
        return raw_sql_str.replace("[query_vector]", query_embedding_str)


class NLSQLRetriever(BaseRetriever):
    """Text-to-SQL Retriever.

    Retrieves via text.

    Args:
        sql_database (SQLDatabase): SQL database.
        text_to_sql_prompt (BasePromptTemplate): Prompt template for text-to-sql.
            Defaults to DEFAULT_TEXT_TO_SQL_PROMPT.
        context_query_kwargs (dict): Mapping from table name to context query.
            Defaults to None.
        tables (Union[List[str], List[Table]]): List of table names or Table objects.
        table_retriever (ObjectRetriever[SQLTableSchema]): Object retriever for
            SQLTableSchema objects. Defaults to None.
        context_str_prefix (str): Prefix for context string. Defaults to None.
        service_context (ServiceContext): Service context. Defaults to None.
    
    """
    def __init__(
        self,
        sql_database: SQLDatabase,
        text_to_sql_prompt: Optional[BasePromptTemplate] = None,
        context_query_kwargs: Optional[dict] = None,
        tables: Optional[Union[List[str], List[Table]]] = None,
        table_retriever: Optional[ObjectRetriever[SQLTableSchema]] = None,
        context_str_prefix: Optional[str] = None,
        sql_parser_mode: SQLParserMode = SQLParserMode.DEFAULT,
        service_context: Optional[ServiceContext] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._sql_retriever = SQLRetriever(sql_database)
        self._sql_database = sql_database
        self._get_tables = self._load_get_tables_fn(
            sql_database, tables, context_query_kwargs, table_retriever
        )
        self._context_str_prefix = context_str_prefix
        self._service_context = service_context or ServiceContext.from_defaults()
        self._text_to_sql_prompt = text_to_sql_prompt or DEFAULT_TEXT_TO_SQL_PROMPT
        self._sql_parser_mode = sql_parser_mode
        self._sql_parser = self._load_sql_parser(sql_parser_mode, self._service_context)

    def _load_sql_parser(self, sql_parser_mode: SQLParserMode, service_context: ServiceContext) -> BaseSQLParser:
        """Load SQL parser."""
        if sql_parser_mode == SQLParserMode.DEFAULT:
            return DefaultSQLParser()
        elif sql_parser_mode == SQLParserMode.PGVECTOR:
            return PGVectorSQLParser(embed_model=service_context.embed_model)
        else:
            raise ValueError(f"Unknown SQL parser mode: {sql_parser_mode}")

    def _load_get_tables_fn(
        self,
        sql_database: SQLDatabase,
        tables: Optional[Union[List[str], List[Table]]] = None,
        context_query_kwargs: Optional[dict] = None,
        table_retriever: Optional[ObjectRetriever[SQLTableSchema]] = None,
    ) -> Callable[[str], List[SQLTableSchema]]:
        """Load get_tables function."""
        if table_retriever is not None:
            return lambda query_str: table_retriever.retrieve(query_str)
        else:
            if tables is not None:
                table_names = [t.name if isinstance(t, Table) else t for t in tables]
            else:
                table_names = sql_database.get_usable_table_names()
            context_strs = [context_query_kwargs.get(t, None) for t in table_names]
            table_schemas = [
                SQLTableSchema(table_name=t, context_str=c)
                for t, c in zip(table_names, context_strs)
            ]
            return lambda _: table_schemas

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        table_desc_str = self._get_table_context(query_bundle)
        logger.info(f"> Table desc str: {table_desc_str}")

        response_str = self._service_context.llm_predictor.predict(
            self._text_to_sql_prompt,
            query_str=query_bundle.query_str,
            schema=table_desc_str,
            dialect=self._sql_database.dialect,
        )

        sql_query_str = self._sql_parser.parse_response_to_sql(response_str, query_bundle)
        # assume that it's a valid SQL query
        logger.debug(f"> Predicted SQL query: {sql_query_str}")
        return self._sql_retriever.retrieve(sql_query_str)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Async retrieve nodes given query."""
        table_desc_str = self._get_table_context(query_bundle)
        logger.info(f"> Table desc str: {table_desc_str}")

        response_str = await self._service_context.llm_predictor.apredict(
            self._text_to_sql_prompt,
            query_str=query_bundle.query_str,
            schema=table_desc_str,
            dialect=self._sql_database.dialect,
        )

        sql_query_str = self._sql_parser.parse_response_to_sql(response_str, query_bundle)
        # assume that it's a valid SQL query
        logger.debug(f"> Predicted SQL query: {sql_query_str}")
        return self._sql_retriever.aretrieve(sql_query_str)

    def _get_table_context(self, query_bundle: QueryBundle) -> str:
        """Get table context.

        Get tables schema + optional context as a single string.

        """
        table_schema_objs = self._get_tables(query_bundle.query_str)
        context_strs = []
        if self._context_str_prefix is not None:
            context_strs = [self._context_str_prefix]

        for table_schema_obj in table_schema_objs:
            table_info = self._sql_database.get_single_table_info(
                table_schema_obj.table_name
            )

            if table_schema_obj.context_str:
                table_opt_context = " The table description is: "
                table_opt_context += table_schema_obj.context_str
                table_info += table_opt_context

            context_strs.append(table_info)

        return "\n\n".join(context_strs)

        

    
