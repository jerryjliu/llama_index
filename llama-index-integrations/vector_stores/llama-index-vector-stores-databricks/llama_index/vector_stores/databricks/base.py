"""
Databricks Vector Search index.

Supports Delta Sync indexes and Direct Access indexes in Databricks Vector Search.
"""

import json
import ast
import logging
from typing import Any, List, Dict, Optional, cast
from enum import Enum
import json

from databricks.vector_search.client import VectorSearchIndex

from llama_index.core.bridge.pydantic import BaseModel, Field, PrivateAttr
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    FilterCondition,
    FilterOperator,
    VectorStoreQuery,
    VectorStoreQueryResult,
    VectorStoreQueryMode,
)
from llama_index.core.vector_stores.utils import node_to_metadata_dict
from llama_index.core.schema import TextNode, BaseNode
from llama_index.core.bridge.pydantic import PrivateAttr


class _DatabricksIndexType(str, Enum):
    DIRECT_ACCESS = "DIRECT_ACCESS"
    DELTA_SYNC = "DELTA_SYNC"


class _DatabricksIndexDescription(BaseModel):
    primary_key: str
    index_type: _DatabricksIndexType
    delta_sync_index_spec: Dict = Field(default_factory=dict)
    direct_access_index_spec: Dict = Field(default_factory=dict)


_logger = logging.getLogger(__name__)


_filter_translation = {
    FilterOperator.EQ: "",
    FilterOperator.GT: ">",
    FilterOperator.LT: "<",
    FilterOperator.NE: "NOT",
    FilterOperator.GTE: ">=",
    FilterOperator.LTE: "<=",
    FilterOperator.IN: "",
    FilterOperator.NIN: "NOT",
}


def _transform_databricks_filter_operator(operator: FilterOperator) -> str:
    try:
        return _filter_translation[operator]

    except KeyError as e:
        raise ValueError(f"filter operator {operator} is not supported")


def _to_databricks_filter(standard_filters: MetadataFilters) -> dict:
    """Convert from standard dataclass to databricks filter dict."""
    filters = {}

    condition = standard_filters.condition or FilterOperator.AND

    for filter in standard_filters.filters:
        value = filter.value if isinstance(filter.value, list) else [filter.value]

        transformed_operator = _transform_databricks_filter_operator(filter.operator)

        if transformed_operator == "":
            key = filter.key

        else:
            key = f"{filter.key} {transformed_operator}"

        if key in filters:
            raise ValueError(f"filter condition already exists for {key}")

        filters[key] = value

    if condition == FilterCondition.AND:
        return filters

    elif condition == FilterCondition.OR:
        keys, values = zip(*filters.items())
        return {" OR ".join(keys): values}

    raise ValueError(f"condition {condition} is not supported")


class DatabricksVectorSearch(BasePydanticVectorStore):
    """
    Vector store for Databricks Vector Search.

    Install ``databricks-vectorsearch`` package using the following in a Databricks notebook:
    %pip install databricks-vectorsearch
    dbutils.library.restartPython()

    """

    stores_text: bool = True
    text_column: Optional[str]
    columns: Optional[List[str]]

    _index: VectorSearchIndex = PrivateAttr()
    _primary_key: str = PrivateAttr()
    _index_type: str = PrivateAttr()
    _delta_sync_index_spec: dict = PrivateAttr()
    _direct_access_index_spec: dict = PrivateAttr()
    _doc_id_to_pk: dict = PrivateAttr()

    def __init__(
        self,
        index: VectorSearchIndex,
        text_column: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> None:
        super().__init__(text_column=text_column, columns=columns)

        try:
            from databricks.vector_search.client import VectorSearchIndex
        except ImportError:
            raise ImportError(
                "`databricks-vectorsearch` package not found: "
                "please run `pip install databricks-vectorsearch`"
            )
        if not isinstance(index, VectorSearchIndex):
            raise TypeError(
                f"index must be of type `VectorSearchIndex`, not {type(index)}"
            )

        self._index = index

        # unpack the index spec
        index_description = _DatabricksIndexDescription.parse_obj(
            self._index.describe()
        )

        self._primary_key = index_description.primary_key
        self._index_type = index_description.index_type
        self._delta_sync_index_spec = index_description.delta_sync_index_spec
        self._direct_access_index_spec = index_description.direct_access_index_spec
        self._doc_id_to_pk = {}
        self._node_info_columns = ["relationships", "start_char_idx", "end_char_idx"]

        if columns is None:
            columns = []
        if "doc_id" not in columns:
            columns = columns[:19] + ["doc_id"]

        # initialize the column name for the text column in the delta table
        if self._is_databricks_managed_embeddings():
            index_source_column = self._embedding_source_column_name()

            # check if input text column matches the source column of the index
            if text_column is not None and text_column != index_source_column:
                raise ValueError(
                    f"text_column '{text_column}' does not match with the "
                    f"source column of the index: '{index_source_column}'."
                )

            self.text_column = index_source_column
        else:
            if text_column is None:
                raise ValueError("text_column is required for self-managed embeddings.")
            self.text_column = text_column

        # Fold primary key and text column into columns if they're not empty.
        columns_to_add = set(columns or [])
        columns_to_add.add(self._primary_key)
        columns_to_add.add(self.text_column)
        columns_to_add -= {"", None}

        self.columns = list(columns_to_add)

        # If the index schema is known, all our columns should be in that index.
        # Validate specified columns are in the index
        index_schema = self._index_schema()

        if self._is_direct_access_index() and index_schema:
            missing_columns = columns_to_add - set(index_schema.keys())

            if missing_columns:
                raise ValueError(
                    f"columns missing from schema: {', '.join(missing_columns)}"
                )

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        if self._is_databricks_managed_embeddings():
            raise ValueError(
                "Adding nodes is not supported for Databricks-managed embeddings."
            )

        # construct the entries to upsert
        entries = []
        ids = []
        for node in nodes:
            node_id = node.node_id

            # unroll metadata in node_content
            node_metadata = node_to_metadata_dict(
                node, remove_text=True, flat_metadata=True
            )
            node_content = json.loads(node_metadata.get("_node_content", {}))
            metadata = node_content.get("metadata", {})

            # add node_info (star_char, end_char, relationships)
            node_info = {key: node_content[key] for key in self._node_info_columns}
            metadata.update(
                {"doc_id": node_metadata["doc_id"], "node_info": str(node_info)}
            )
            metadata_columns = self.columns or []

            # explicitly record doc_id as metadata (for delete)
            if "doc_id" not in metadata_columns:
                metadata_columns.append("doc_id")

            entry = {
                self._primary_key: node_id,
                self.text_column: node.get_content(),
                self._embedding_vector_column_name(): node.get_embedding(),
                **{
                    col: metadata.get(col)
                    for col in filter(
                        lambda column: column
                        not in (self._primary_key, self.text_column),
                        metadata_columns,
                    )
                },
            }
            doc_id = metadata.get("doc_id")
            self._doc_id_to_pk[doc_id] = list(
                set(self._doc_id_to_pk.get(doc_id, []) + [node_id])  # noqa: RUF005
            )  # associate this node_id with this doc_id

            entries.append(entry)
            ids.append(node_id)

        # attempt the upsert
        upsert_resp = self._index.upsert(
            entries,
        )

        # return the successful IDs
        response_status = upsert_resp.get("status")

        failed_ids = (
            set(upsert_resp["result"]["failed_primary_keys"] or [])
            if "result" in upsert_resp
            and "failed_primary_keys" in upsert_resp["result"]
            else set()
        )

        if response_status not in ("PARTIAL_SUCCESS", "FAILURE") or not failed_ids:
            return ids

        elif response_status == "PARTIAL_SUCCESS":
            _logger.warning(
                "failed to add %d out of %d texts to the index",
                len(failed_ids),
                len(ids),
            )

        elif response_status == "FAILURE":
            _logger.error("failed to add all %d texts to the index", len(ids))

        return list(filter(lambda id_: id_ not in failed_ids, ids))

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        primary_keys = self._doc_id_to_pk.get(
            ref_doc_id, None
        )  # get the node_ids associated with the doc_id
        if primary_keys is not None:
            self._index.delete(
                primary_keys=primary_keys,
            )
            self._doc_id_to_pk.pop(
                ref_doc_id
            )  # remove this doc_id from the doc_id-to-node_id map

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes."""
        if self._is_databricks_managed_embeddings():
            query_text = query.query_str
            query_vector = None
        else:
            query_text = None
            query_vector = cast(List[float], query.query_embedding)

        if query.mode not in (
            VectorStoreQueryMode.DEFAULT,
            VectorStoreQueryMode.HYBRID,
        ):
            raise ValueError(
                "Only DEFAULT and HYBRID modes are supported for Databricks Vector Search."
            )

        if query.filters is not None:
            filters = _to_databricks_filter(query.filters)
        else:
            filters = None

        search_resp = self._index.similarity_search(
            columns=self.columns,
            query_text=query_text,
            query_vector=query_vector,
            filters=filters,
            num_results=query.similarity_top_k,
        )

        columns = [
            col["name"] for col in search_resp.get("manifest", {}).get("columns", [])
        ]
        top_k_nodes = []
        top_k_ids = []
        top_k_scores = []
        for result in search_resp.get("result", {}).get("data_array", []):
            doc_id = result[columns.index(self._primary_key)]
            text_content = result[columns.index(self.text_column)]

            # extract metadata and node_info respectively from metadata_dict
            metadata_dict = {
                col: value
                for col, value in zip(columns[:-1], result[:-1])
                if col not in [self._primary_key, self.text_column, "doc_id"]
            }
            metadata = {
                col: value for col, value in metadata_dict.items() if col != "node_info"
            }
            node_info = ast.literal_eval(metadata_dict.get("node_info", "{}"))
            score = result[-1]
            node = TextNode(
                text=text_content,
                id_=doc_id,
                metadata=metadata,
                start_char_idx=node_info.get("start_char_idx", None),
                end_char_idx=node_info.get("end_char_idx", None),
                relationships=node_info.get("relationships", {}),
            )
            top_k_ids.append(doc_id)
            top_k_nodes.append(node)
            top_k_scores.append(score)

        return VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )

    @property
    def client(self) -> Any:
        """Return VectorStoreIndex."""
        return self._index

    # The remaining utilities (and snippets of the above) are taken from
    # https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/databricks_vector_search.py
    def _index_schema(self) -> Optional[dict]:
        """Return the index schema as a dictionary.
        Return None if no schema found.
        """
        if self._is_direct_access_index():
            schema_json = self._direct_access_index_spec.get("schema_json")
            if schema_json is not None:
                return json.loads(schema_json)
        return None

    def _embedding_vector_column_name(self) -> Optional[str]:
        """Return the name of the embedding vector column.
        None if the index is not a self-managed embedding index.
        """
        return self._embedding_vector_column().get("name")

    def _embedding_vector_column(self) -> dict:
        """Return the embedding vector column configs as a dictionary.
        Empty if the index is not a self-managed embedding index.
        """
        index_spec = (
            self._delta_sync_index_spec
            if self._is_delta_sync_index()
            else self._direct_access_index_spec
        )
        return next(iter(index_spec.get("embedding_vector_columns") or []), {})

    def _embedding_source_column_name(self) -> Optional[str]:
        """Return the name of the embedding source column.
        None if the index is not a Databricks-managed embedding index.
        """
        return self._embedding_source_column().get("name")

    def _embedding_source_column(self) -> dict:
        """Return the embedding source column configs as a dictionary.
        Empty if the index is not a Databricks-managed embedding index.
        """
        return next(
            iter(self._delta_sync_index_spec.get("embedding_source_columns") or []),
            {},
        )

    def _is_delta_sync_index(self) -> bool:
        """Return True if the index is a delta-sync index."""
        return self._index_type == _DatabricksIndexType.DELTA_SYNC

    def _is_direct_access_index(self) -> bool:
        """Return True if the index is a direct-access index."""
        return self._index_type == _DatabricksIndexType.DIRECT_ACCESS

    def _is_databricks_managed_embeddings(self) -> bool:
        """Return True if the embeddings are managed by Databricks Vector Search."""
        return (
            self._is_delta_sync_index()
            and self._embedding_source_column_name() is not None
        )
