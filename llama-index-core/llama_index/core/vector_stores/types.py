"""Vector store index types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

import fsspec
from deprecated import deprecated
from llama_index.core.bridge.pydantic import (
    BaseModel,
    ConfigDict,
    StrictFloat,
    StrictInt,
    StrictStr,
)
from llama_index.core.schema import BaseComponent, BaseNode, TextNode

DEFAULT_PERSIST_DIR = "./storage"
DEFAULT_PERSIST_FNAME = "vector_store.json"


# legacy: kept for backward compatibility
NodeWithEmbedding = TextNode


@dataclass
class VectorStoreQueryResult:
    """Vector store query result."""

    nodes: Optional[Sequence[BaseNode]] = None
    similarities: Optional[List[float]] = None
    ids: Optional[List[str]] = None


class VectorStoreQueryMode(str, Enum):
    """Vector store query mode."""

    DEFAULT = "default"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    TEXT_SEARCH = "text_search"
    SEMANTIC_HYBRID = "semantic_hybrid"

    # fit learners
    SVM = "svm"
    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_REGRESSION = "linear_regression"

    # maximum marginal relevance
    MMR = "mmr"


class FilterOperator(str, Enum):
    """Vector store filter operator."""

    # TODO add more operators
    EQ = "=="  # default operator (string, int, float)
    GT = ">"  # greater than (int, float)
    LT = "<"  # less than (int, float)
    NE = "!="  # not equal to (string, int, float)
    GTE = ">="  # greater than or equal to (int, float)
    LTE = "<="  # less than or equal to (int, float)
    IN = "in"  # In array (string or number)
    NIN = "nin"  # Not in array (string or number)
    ANY = "any"  # Contains any (array of strings)
    ALL = "all"  # Contains all (array of strings)
    TEXT_MATCH = "text_match"  # full text match (allows you to search for a specific substring, token or phrase within the text field)
    TEXT_MATCH_INSENSITIVE = "text_match_insensitive"  # full text match (case insensitive)
    CONTAINS = "contains"  # metadata array contains value (string or number)
    IS_EMPTY = "is_empty"  # the field is not exist or empty (null or empty array)


class FilterCondition(str, Enum):
    """Vector store filter conditions to combine different filters."""

    # TODO add more conditions
    AND = "and"
    OR = "or"
    NOT = "not"  # negates the filter condition


class MetadataFilter(BaseModel):
    r"""Comprehensive metadata filter for vector stores to support more operators.

    Value uses Strict* types, as int, float and str are compatible types and were all
    converted to string before.

    See: https://docs.pydantic.dev/latest/usage/types/#strict-types
    """

    key: str
    value: Optional[
        Union[
            StrictInt,
            StrictFloat,
            StrictStr,
            List[StrictStr],
            List[StrictFloat],
            List[StrictInt],
        ]
    ]
    operator: FilterOperator = FilterOperator.EQ

    @classmethod
    def from_dict(
        cls,
        filter_dict: Dict,
    ) -> "MetadataFilter":
        """Create MetadataFilter from dictionary.

        Args:
            filter_dict: Dict with key, value and operator.

        """
        return MetadataFilter.model_validate(filter_dict)

    def matches(self, node: BaseNode) -> bool:
        if self.operator == FilterOperator.EQ:
            return node.metadata.get(self.key) == self.value
        elif self.operator == FilterOperator.GT:
            return node.metadata.get(self.key) > self.value
        elif self.operator == FilterOperator.LT:
            return node.metadata.get(self.key) < self.value
        elif self.operator == FilterOperator.NE:
            return node.metadata.get(self.key) != self.value
        elif self.operator == FilterOperator.GTE:
            return node.metadata.get(self.key) >= self.value
        elif self.operator == FilterOperator.LTE:
            return node.metadata.get(self.key) <= self.value
        elif self.operator == FilterOperator.IN:
            return node.metadata.get(self.key) in self.value
        elif self.operator == FilterOperator.NIN:
            return node.metadata.get(self.key) not in self.value
        elif self.operator == FilterOperator.ANY:
            raise NotImplementedError("ANY operator not implemented yet")
        elif self.operator == FilterOperator.ALL:
            raise NotImplementedError("ALL operator not implemented yet")
        elif self.operator == FilterOperator.TEXT_MATCH:
            raise NotImplementedError("TEXT_MATCH operator not implemented yet")
        elif self.operator == FilterOperator.TEXT_MATCH_INSENSITIVE:
            raise NotImplementedError("TEXT_MATCH_INSENSITIVE operator not implemented yet")
        elif self.operator == FilterOperator.CONTAINS:
            return node.metadata.get(self.key) in self.value
        elif self.operator == FilterOperator.IS_EMPTY:
            if self.key not in node.metadata:
                return True
            elif node.metadata.get(self.key) is None:
                return True
            elif hasattr(node.metadata.get(self.key), "__len__"):
                return len(node.metadata.get(self.key)) == 0
            else:
                return False
        else:
            raise ValueError(f"Unknown filter operator: {self.operator}")


# # TODO: Deprecate ExactMatchFilter and use MetadataFilter instead
# # Keep class for now so that AutoRetriever can still work with old vector stores
# class ExactMatchFilter(BaseModel):
#     key: str
#     value: Union[StrictInt, StrictFloat, StrictStr]

# set ExactMatchFilter to MetadataFilter
ExactMatchFilter = MetadataFilter


class MetadataFilters(BaseModel):
    """Metadata filters for vector stores."""

    # Exact match filters and Advanced filters with operators like >, <, >=, <=, !=, etc.
    filters: List[Union[MetadataFilter, ExactMatchFilter, "MetadataFilters"]]
    # and/or such conditions for combining different filters
    condition: Optional[FilterCondition] = FilterCondition.AND

    @classmethod
    @deprecated(
        "`from_dict()` is deprecated. "
        "Please use `MetadataFilters(filters=.., condition='and')` directly instead."
    )
    def from_dict(cls, filter_dict: Dict) -> "MetadataFilters":
        """Create MetadataFilters from json."""
        filters = []
        for k, v in filter_dict.items():
            filter = MetadataFilter(key=k, value=v, operator=FilterOperator.EQ)
            filters.append(filter)
        return cls(filters=filters)

    @classmethod
    def from_dicts(
        cls,
        filter_dicts: List[Dict],
        condition: Optional[FilterCondition] = FilterCondition.AND,
    ) -> "MetadataFilters":
        """Create MetadataFilters from dicts.

        This takes in a list of individual MetadataFilter objects, along
        with the condition.

        Args:
            filter_dicts: List of dicts, each dict is a MetadataFilter.
            condition: FilterCondition to combine different filters.

        """
        return cls(
            filters=[MetadataFilter.from_dict(filter_dict) for filter_dict in filter_dicts],
            condition=condition,
        )

    def legacy_filters(self) -> List[ExactMatchFilter]:
        """Convert MetadataFilters to legacy ExactMatchFilters."""
        filters = []
        for filter in self.filters:
            if isinstance(filter, MetadataFilters) or filter.operator != FilterOperator.EQ:
                raise ValueError(
                    "Vector Store only supports exact match filters. "
                    "Please use ExactMatchFilter or FilterOperator.EQ instead."
                )
            filters.append(ExactMatchFilter(key=filter.key, value=filter.value))
        return filters

    def matches(self, node: BaseNode) -> bool:
        sub_conditions = [sub_filter.matches(node) for sub_filter in self.filters]

        if self.condition == FilterCondition.AND:
            return all(sub_conditions)
        elif self.condition == FilterCondition.OR:
            return any(sub_conditions)
        elif self.condition == FilterCondition.NOT:
            assert len(sub_conditions) == 1, "NOT condition must have exactly one sub-filter"
            return not sub_conditions[0]
        else:
            raise ValueError(f"Unknown filter condition: {self.condition}")


class VectorStoreQuerySpec(BaseModel):
    """Schema for a structured request for vector store
    (i.e. to be converted to a VectorStoreQuery).

    Currently only used by VectorIndexAutoRetriever.
    """

    query: str
    filters: List[MetadataFilter]
    top_k: Optional[int] = None


class MetadataInfo(BaseModel):
    """Information about a metadata filter supported by a vector store.

    Currently only used by VectorIndexAutoRetriever.
    """

    name: str
    type: str
    description: str


class VectorStoreInfo(BaseModel):
    """Information about a vector store (content and supported metadata filters).

    Currently only used by VectorIndexAutoRetriever.
    """

    metadata_info: List[MetadataInfo]
    content_info: str


@dataclass
class VectorStoreQuery:
    """Vector store query."""

    query_embedding: Optional[List[float]] = None
    similarity_top_k: int = 1
    doc_ids: Optional[List[str]] = None
    node_ids: Optional[List[str]] = None
    query_str: Optional[str] = None
    output_fields: Optional[List[str]] = None
    embedding_field: Optional[str] = None

    mode: VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT

    # NOTE: only for hybrid search (0 for bm25, 1 for vector search)
    alpha: Optional[float] = None

    # metadata filters
    filters: Optional[MetadataFilters] = None

    # only for mmr
    mmr_threshold: Optional[float] = None

    # NOTE: currently only used by postgres hybrid search
    sparse_top_k: Optional[int] = None
    # NOTE: return top k results from hybrid search. similarity_top_k is used for dense search top k
    hybrid_top_k: Optional[int] = None


@runtime_checkable
class VectorStore(Protocol):
    """Abstract vector store protocol."""

    stores_text: bool
    is_embedding_query: bool = True

    @property
    def client(self) -> Any:
        """Get client."""
        ...

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes with embedding to vector store."""
        ...

    async def async_add(
        self,
        nodes: List[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        """
        Asynchronously add nodes with embedding to vector store.
        NOTE: this is not implemented for all vector stores. If not implemented,
        it will just call add synchronously.
        """
        return self.add(nodes)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id."""
        ...

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.
        NOTE: this is not implemented for all vector stores. If not implemented,
        it will just call delete synchronously.
        """
        self.delete(ref_doc_id, **delete_kwargs)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query vector store."""
        ...

    async def aquery(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Asynchronously query vector store.
        NOTE: this is not implemented for all vector stores. If not implemented,
        it will just call query synchronously.
        """
        return self.query(query, **kwargs)

    def persist(self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None) -> None:
        return None


# TODO: Temp copy of VectorStore for pydantic, can't mix with runtime_checkable
class BasePydanticVectorStore(BaseComponent, ABC):
    """Abstract vector store protocol."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    stores_text: bool
    is_embedding_query: bool = True

    @property
    @abstractmethod
    def client(self) -> Any:
        """Get client."""

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Get nodes from vector store."""
        raise NotImplementedError("get_nodes not implemented")

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Asynchronously get nodes from vector store."""
        return self.get_nodes(node_ids, filters)

    @abstractmethod
    def add(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        """Add nodes to vector store."""

    async def async_add(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        """
        Asynchronously add nodes to vector store.
        NOTE: this is not implemented for all vector stores. If not implemented,
        it will just call add synchronously.
        """
        return self.add(nodes, **kwargs)

    @abstractmethod
    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id."""

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.
        NOTE: this is not implemented for all vector stores. If not implemented,
        it will just call delete synchronously.
        """
        self.delete(ref_doc_id, **delete_kwargs)

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Delete nodes from vector store."""
        raise NotImplementedError("delete_nodes not implemented")

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Asynchronously delete nodes from vector store."""
        self.delete_nodes(node_ids, filters)

    def clear(self) -> None:
        """Clear all nodes from configured vector store."""
        raise NotImplementedError("clear not implemented")

    async def aclear(self) -> None:
        """Asynchronously clear all nodes from configured vector store."""
        self.clear()

    @abstractmethod
    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query vector store."""

    async def aquery(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Asynchronously query vector store.
        NOTE: this is not implemented for all vector stores. If not implemented,
        it will just call query synchronously.
        """
        return self.query(query, **kwargs)

    def persist(self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None) -> None:
        return None
