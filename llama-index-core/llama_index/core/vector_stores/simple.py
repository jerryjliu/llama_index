"""Simple vector store index."""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, cast

import fsspec
from dataclasses_json import DataClassJsonMixin
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.indices.query.embedding_utils import (
    get_top_k_embeddings,
    get_top_k_embeddings_learner,
    get_top_k_mmr_embeddings,
)
from llama_index.core.schema import BaseNode
from llama_index.core.utils import concat_dirs
from llama_index.core.vector_stores.types import (
    DEFAULT_PERSIST_DIR,
    DEFAULT_PERSIST_FNAME,
    BasePydanticVectorStore,
    MetadataFilters,
    FilterCondition,
    FilterOperator,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import node_to_metadata_dict

logger = logging.getLogger(__name__)

LEARNER_MODES = {
    VectorStoreQueryMode.SVM,
    VectorStoreQueryMode.LINEAR_REGRESSION,
    VectorStoreQueryMode.LOGISTIC_REGRESSION,
}

MMR_MODE = VectorStoreQueryMode.MMR

NAMESPACE_SEP = "__"
DEFAULT_VECTOR_STORE = "default"


def _build_metadata_filter_fn(
    metadata_lookup_fn: Callable[[str], Mapping[str, Any]],
    metadata_filters: Optional[MetadataFilters] = None,
) -> Callable[[str], bool]:
    """Build metadata filter function."""
    filter_list = metadata_filters.filters if metadata_filters else []
    if not filter_list or not metadata_filters:
        return lambda _: True

    filter_condition = cast(MetadataFilters, metadata_filters.condition)

    def filter_fn(node_id: str) -> bool:
        def _process_filter_match(
            operator: FilterOperator, value: Any, metadata_value: Any
        ) -> bool:
            if metadata_value is None:
                return False
            if operator == FilterOperator.EQ:
                return metadata_value == value
            if operator == FilterOperator.NE:
                return metadata_value != value
            if operator == FilterOperator.GT:
                return metadata_value > value
            if operator == FilterOperator.GTE:
                return metadata_value >= value
            if operator == FilterOperator.LT:
                return metadata_value < value
            if operator == FilterOperator.LTE:
                return metadata_value <= value
            if operator == FilterOperator.IN:
                return metadata_value in value
            if operator == FilterOperator.NIN:
                return metadata_value not in value
            if operator == FilterOperator.CONTAINS:
                return value in metadata_value
            if operator == FilterOperator.TEXT_MATCH:
                return value.lower() in metadata_value.lower()
            if operator == FilterOperator.ALL:
                return all(val in metadata_value for val in value)
            if operator == FilterOperator.ANY:
                return any(val in metadata_value for val in value)
            raise ValueError(f"Invalid operator: {operator}")

        metadata = metadata_lookup_fn(node_id)

        filter_matches_list = []
        for filter_ in filter_list:
            if isinstance(filter_, MetadataFilters):
                raise ValueError("Nested MetadataFilters are not supported.")

            filter_matches = True
            metadata_value = metadata.get(filter_.key, None)
            if filter_.operator == FilterOperator.IS_EMPTY:
                filter_matches = (
                    metadata_value is None
                    or metadata_value == ""
                    or metadata_value == []
                )
            else:
                filter_matches = _process_filter_match(
                    operator=filter_.operator,
                    value=filter_.value,
                    metadata_value=metadata_value,
                )

            filter_matches_list.append(filter_matches)

        if filter_condition == FilterCondition.AND:
            return all(filter_matches_list)
        elif filter_condition == FilterCondition.OR:
            return any(filter_matches_list)
        else:
            raise ValueError(f"Invalid filter condition: {filter_condition}")

    return filter_fn


@dataclass
class SimpleVectorStoreData(DataClassJsonMixin):
    """Simple Vector Store Data container.

    Args:
        embedding_dict (Optional[dict]): dict mapping node_ids to embeddings.
        text_id_to_ref_doc_id (Optional[dict]):
            dict mapping text_ids/node_ids to ref_doc_ids.

    """

    embedding_dict: Dict[str, List[float]] = field(default_factory=dict)
    text_id_to_ref_doc_id: Dict[str, str] = field(default_factory=dict)
    metadata_dict: Dict[str, Any] = field(default_factory=dict)


class SimpleVectorStore(BasePydanticVectorStore):
    """Simple Vector Store.

    In this vector store, embeddings are stored within a simple, in-memory dictionary.

    Args:
        simple_vector_store_data_dict (Optional[dict]): data dict
            containing the embeddings and doc_ids. See SimpleVectorStoreData
            for more details.
    """

    stores_text: bool = False

    data: SimpleVectorStoreData = Field(default_factory=SimpleVectorStoreData)
    _fs: fsspec.AbstractFileSystem = PrivateAttr()

    def __init__(
        self,
        data: Optional[SimpleVectorStoreData] = None,
        fs: Optional[fsspec.AbstractFileSystem] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(data=data or SimpleVectorStoreData())  # type: ignore[call-arg]
        self._fs = fs or fsspec.filesystem("file")

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        namespace: str = DEFAULT_VECTOR_STORE,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "SimpleVectorStore":
        """Load from persist dir."""
        persist_fname = f"{namespace}{NAMESPACE_SEP}{DEFAULT_PERSIST_FNAME}"

        if fs is not None:
            persist_path = concat_dirs(persist_dir, persist_fname)
        else:
            persist_path = os.path.join(persist_dir, persist_fname)
        return cls.from_persist_path(persist_path, fs=fs)

    @classmethod
    def from_namespaced_persist_dir(
        cls,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> Dict[str, BasePydanticVectorStore]:
        """Load from namespaced persist dir."""
        listing_fn = os.listdir if fs is None else fs.listdir

        vector_stores: Dict[str, BasePydanticVectorStore] = {}

        try:
            for fname in listing_fn(persist_dir):
                if fname.endswith(DEFAULT_PERSIST_FNAME):
                    namespace = fname.split(NAMESPACE_SEP)[0]

                    # handle backwards compatibility with stores that were persisted
                    if namespace == DEFAULT_PERSIST_FNAME:
                        vector_stores[DEFAULT_VECTOR_STORE] = cls.from_persist_dir(
                            persist_dir=persist_dir, fs=fs
                        )
                    else:
                        vector_stores[namespace] = cls.from_persist_dir(
                            persist_dir=persist_dir, namespace=namespace, fs=fs
                        )
        except Exception:
            # failed to listdir, so assume there is only one store
            try:
                vector_stores[DEFAULT_VECTOR_STORE] = cls.from_persist_dir(
                    persist_dir=persist_dir, fs=fs, namespace=DEFAULT_VECTOR_STORE
                )
            except Exception:
                # no namespace backwards compat
                vector_stores[DEFAULT_VECTOR_STORE] = cls.from_persist_dir(
                    persist_dir=persist_dir, fs=fs
                )

        return vector_stores

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "SimpleVectorStore"

    @property
    def client(self) -> None:
        """Get client."""
        return

    @property
    def _data(self) -> SimpleVectorStoreData:
        """Backwards compatibility."""
        return self.data

    def get(self, text_id: str) -> List[float]:
        """Get embedding."""
        return self.data.embedding_dict[text_id]

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        index_id: Optional[str] = None,
    ) -> List[BaseNode]:
        """Get nodes."""
        raise NotImplementedError("SimpleVectorStore does not store nodes directly.")

    def add(
        self,
        nodes: Sequence[BaseNode],
        index_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add nodes to index."""
        for node in nodes:
            self.data.embedding_dict[node.node_id] = node.get_embedding()
            self.data.text_id_to_ref_doc_id[node.node_id] = node.ref_doc_id or "None"

            metadata = node_to_metadata_dict(
                node, remove_text=True, flat_metadata=False
            )
            metadata.pop("_node_content", None)
            self.data.metadata_dict[node.node_id] = metadata
        return [node.node_id for node in nodes]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        text_ids_to_delete = set()
        for text_id, ref_doc_id_ in self.data.text_id_to_ref_doc_id.items():
            if ref_doc_id == ref_doc_id_:
                text_ids_to_delete.add(text_id)

        for text_id in text_ids_to_delete:
            del self.data.embedding_dict[text_id]
            del self.data.text_id_to_ref_doc_id[text_id]
            # Handle metadata_dict not being present in stores that were persisted
            # without metadata, or, not being present for nodes stored
            # prior to metadata functionality.
            if self.data.metadata_dict is not None:
                self.data.metadata_dict.pop(text_id, None)

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        index_id: Optional[str] = None,
        **delete_kwargs: Any,
    ) -> None:
        filter_fn = _build_metadata_filter_fn(
            lambda node_id: self.data.metadata_dict[node_id], filters
        )

        if node_ids is not None:
            node_id_set = set(node_ids)

            def node_filter_fn(node_id: str) -> bool:
                return node_id in node_id_set and filter_fn(node_id)

        else:

            def node_filter_fn(node_id: str) -> bool:
                return filter_fn(node_id)

        for node_id in list(self.data.embedding_dict.keys()):
            if node_filter_fn(node_id):
                del self.data.embedding_dict[node_id]
                del self.data.text_id_to_ref_doc_id[node_id]
                self.data.metadata_dict.pop(node_id, None)

    def clear(self, index_id: Optional[str] = None) -> None:
        """Clear the store."""
        self.data = SimpleVectorStoreData()

    def query(
        self,
        query: VectorStoreQuery,
        index_id: Optional[str] = None,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Get nodes for response."""
        # Prevent metadata filtering on stores that were persisted without metadata.
        if (
            query.filters is not None
            and self.data.embedding_dict
            and not self.data.metadata_dict
        ):
            raise ValueError(
                "Cannot filter stores that were persisted without metadata. "
                "Please rebuild the store with metadata to enable filtering."
            )
        # Prefilter nodes based on the query filter and node ID restrictions.
        query_filter_fn = _build_metadata_filter_fn(
            lambda node_id: self.data.metadata_dict[node_id], query.filters
        )

        if query.node_ids is not None:
            available_ids = set(query.node_ids)

            def node_filter_fn(node_id: str) -> bool:
                return node_id in available_ids

        else:

            def node_filter_fn(node_id: str) -> bool:
                return True

        node_ids = []
        embeddings = []
        # TODO: consolidate with get_query_text_embedding_similarities
        for node_id, embedding in self.data.embedding_dict.items():
            if node_filter_fn(node_id) and query_filter_fn(node_id):
                node_ids.append(node_id)
                embeddings.append(embedding)

        query_embedding = cast(List[float], query.query_embedding)

        if query.mode in LEARNER_MODES:
            top_similarities, top_ids = get_top_k_embeddings_learner(
                query_embedding,
                embeddings,
                similarity_top_k=query.similarity_top_k,
                embedding_ids=node_ids,
            )
        elif query.mode == MMR_MODE:
            mmr_threshold = kwargs.get("mmr_threshold", None)
            top_similarities, top_ids = get_top_k_mmr_embeddings(
                query_embedding,
                embeddings,
                similarity_top_k=query.similarity_top_k,
                embedding_ids=node_ids,
                mmr_threshold=mmr_threshold,
            )
        elif query.mode == VectorStoreQueryMode.DEFAULT:
            top_similarities, top_ids = get_top_k_embeddings(
                query_embedding,
                embeddings,
                similarity_top_k=query.similarity_top_k,
                embedding_ids=node_ids,
            )
        else:
            raise ValueError(f"Invalid query mode: {query.mode}")

        return VectorStoreQueryResult(similarities=top_similarities, ids=top_ids)

    def persist(
        self,
        persist_path: str = os.path.join(DEFAULT_PERSIST_DIR, DEFAULT_PERSIST_FNAME),
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> None:
        """Persist the SimpleVectorStore to a directory."""
        fs = fs or self._fs
        dirpath = os.path.dirname(persist_path)
        if not fs.exists(dirpath):
            fs.makedirs(dirpath)

        with fs.open(persist_path, "w") as f:
            json.dump(self.data.to_dict(), f)

    @classmethod
    def from_persist_path(
        cls, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> "SimpleVectorStore":
        """Create a SimpleKVStore from a persist directory."""
        fs = fs or fsspec.filesystem("file")
        if not fs.exists(persist_path):
            raise ValueError(
                f"No existing {__name__} found at {persist_path}, skipping load."
            )

        logger.debug(f"Loading {__name__} from {persist_path}.")
        with fs.open(persist_path, "rb") as f:
            data_dict = json.load(f)
            data = SimpleVectorStoreData.from_dict(data_dict)
        return cls(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs: Any) -> "SimpleVectorStore":
        save_data = SimpleVectorStoreData.from_dict(data)
        return cls(save_data)

    def to_dict(self, **kwargs: Any) -> Dict[str, Any]:
        return self.data.to_dict()
