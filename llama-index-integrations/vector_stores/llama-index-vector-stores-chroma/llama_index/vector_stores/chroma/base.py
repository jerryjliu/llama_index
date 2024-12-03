"""Chroma vector store."""

import logging
import math
from typing import Any, Dict, Generator, List, Optional, cast, Sequence

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import (
    IncludeEnum,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.utils import truncate_text
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)
logger = logging.getLogger(__name__)


def _transform_chroma_filter_condition(condition: str) -> str:
    """Translate standard metadata filter op to Chroma specific spec."""
    if condition == "and":
        return "$and"
    elif condition == "or":
        return "$or"
    else:
        raise ValueError(f"Filter condition {condition} not supported")


def _transform_chroma_filter_operator(operator: str) -> str:
    """Translate standard metadata filter operator to Chroma specific spec."""
    if operator == "!=":
        return "$ne"
    elif operator == "==":
        return "$eq"
    elif operator == ">":
        return "$gt"
    elif operator == "<":
        return "$lt"
    elif operator == ">=":
        return "$gte"
    elif operator == "<=":
        return "$lte"
    elif operator == "in":
        return "$in"
    elif operator == "nin":
        return "$nin"
    else:
        raise ValueError(f"Filter operator {operator} not supported")


def _to_chroma_filter(
    standard_filters: MetadataFilters,
) -> dict:
    """Translate standard metadata filters to Chroma specific spec."""
    filters = {}
    filters_list = []
    condition = standard_filters.condition or "and"
    condition = _transform_chroma_filter_condition(condition)
    if standard_filters.filters:
        for filter in standard_filters.filters:
            if filter.operator:
                filters_list.append(
                    {
                        filter.key: {
                            _transform_chroma_filter_operator(
                                filter.operator
                            ): filter.value
                        }
                    }
                )
            else:
                filters_list.append({filter.key: filter.value})

    if len(filters_list) == 1:
        # If there is only one filter, return it directly
        return filters_list[0]
    elif len(filters_list) > 1:
        filters[condition] = filters_list
    return filters


import_err_msg = "`chromadb` package not found, please run `pip install chromadb`"

MAX_CHUNK_SIZE = 41665  # One less than the max chunk size for ChromaDB


def chunk_list(
    lst: List[BaseNode], max_chunk_size: int
) -> Generator[List[BaseNode], None, None]:
    """Yield successive max_chunk_size-sized chunks from lst.

    Args:
        lst (List[BaseNode]): list of nodes with embeddings
        max_chunk_size (int): max chunk size

    Yields:
        Generator[List[BaseNode], None, None]: list of nodes with embeddings
    """
    for i in range(0, len(lst), max_chunk_size):
        yield lst[i : i + max_chunk_size]


class ChromaVectorStore(BasePydanticVectorStore):
    """Chroma vector store.

    In this vector store, embeddings are stored within a ChromaDB collection.

    During query time, the index uses ChromaDB to query for the top
    k most similar nodes.

    Args:
        chroma_collection (chromadb.api.models.Collection.Collection):
            ChromaDB collection instance

    Examples:
        `pip install llama-index-vector-stores-chroma`

        ```python
        import chromadb
        from llama_index.vector_stores.chroma import ChromaVectorStore

        # Create a Chroma client and collection
        chroma_client = chromadb.EphemeralClient()
        chroma_collection = chroma_client.create_collection("example_collection")

        # Set up the ChromaVectorStore and StorageContext
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        ```

    """

    stores_text: bool = True
    flat_metadata: bool = True

    collection_name: Optional[str]
    # _client: ClientAPI = PrivateAttr()
    host: Optional[str]
    port: Optional[str]
    ssl: bool
    headers: Optional[Dict[str, str]]
    persist_dir: Optional[str]
    collection_kwargs: Dict[str, Any] = Field(default_factory=dict)

    _collection: Collection = PrivateAttr()

    def __init__(
        self,
        chroma_collection: Optional[Any] = None,
        collection_name: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None,
        ssl: bool = False,
        headers: Optional[Dict[str, str]] = None,
        persist_dir: Optional[str] = None,
        collection_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        collection_kwargs = collection_kwargs or {}

        super().__init__(
            host=host,
            port=port,
            ssl=ssl,
            headers=headers,
            collection_name=collection_name,
            persist_dir=persist_dir,
            collection_kwargs=collection_kwargs or {},
        )
        if chroma_collection is None:
            client = chromadb.HttpClient(host=host, port=port, ssl=ssl, headers=headers)
            self._collection = client.get_or_create_collection(
                name=collection_name, **collection_kwargs
            )
        else:
            self._collection = cast(Collection, chroma_collection)


    @classmethod
    def from_collection(cls, collection: Any) -> "ChromaVectorStore":
        try:
            from chromadb import Collection
        except ImportError:
            raise ImportError(import_err_msg)

        if not isinstance(collection, Collection):
            raise Exception("argument is not chromadb collection instance")

        return cls(chroma_collection=collection)

    @classmethod
    def from_params(
        cls,
        collection_name: str,
        host: Optional[str] = None,
        port: Optional[str] = None,
        ssl: bool = False,
        headers: Optional[Dict[str, str]] = None,
        persist_dir: Optional[str] = None,
        collection_kwargs: dict = {},
        **kwargs: Any,
    ) -> "ChromaVectorStore":
        if persist_dir:
            client = chromadb.PersistentClient(path=persist_dir)
            collection = client.get_or_create_collection(
                name=collection_name, **collection_kwargs
            )
        elif host and port:
            client = chromadb.HttpClient(host=host, port=port, ssl=ssl, headers=headers)
            collection = client.get_or_create_collection(
                name=collection_name, **collection_kwargs
            )
        else:
            raise ValueError(
                "Either `persist_dir` or (`host`,`port`) must be specified"
            )
        return cls(
            chroma_collection=collection,
            host=host,
            port=port,
            ssl=ssl,
            headers=headers,
            persist_dir=persist_dir,
            collection_kwargs=collection_kwargs,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "ChromaVectorStore"

    def move_nodes(self, from_index_id: str, to_index_id: str):
        all_nodes = self._collection.get(where={"__index_id__": from_index_id})
        ids = all_nodes["ids"]
        metadatas = [{**m, "__index_id__": to_index_id} for m in all_nodes["metadatas"]]
        self._collection.update(ids, metadatas=metadatas)
        print("")

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        index_id: Optional[str] = None,
    ) -> List[BaseNode]:

        """Get nodes from index.

        Args:
            node_ids (List[str]): list of node ids
            filters (List[MetadataFilters]): list of metadata filters

        """
        # if not self._collection:
        #     raise ValueError("Collection not initialized")
        node_ids = node_ids or []

        if filters:
            where = _to_chroma_filter(filters)
        else:
            where = None

        result = self._get(None, where=where, ids=node_ids, index_id=index_id)

        return result.nodes

    def add(self, nodes: List[BaseNode], index_id: Optional[str] = None, **add_kwargs: Any) -> List[str]:

        """Add nodes to index.

        Args:
            index_id: Id of the index the nodes will be added too
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        if not self._collection:
            raise ValueError("Collection not initialized")

        max_chunk_size = MAX_CHUNK_SIZE
        node_chunks = chunk_list(nodes, max_chunk_size)

        all_ids = []
        for node_chunk in node_chunks:
            embeddings = []
            metadatas = []
            ids = []
            documents = []
            for node in node_chunk:
                embeddings.append(node.get_embedding())
                metadata_dict = node_to_metadata_dict(
                    node, remove_text=True, flat_metadata=self.flat_metadata
                )
                metadata_dict["__index_id__"] = index_id
                for key in metadata_dict:
                    if metadata_dict[key] is None:
                        metadata_dict[key] = ""
                metadatas.append(metadata_dict)
                ids.append(node.node_id)
                documents.append(node.get_content(metadata_mode=MetadataMode.NONE))

            self._collection.add(
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
                documents=documents,
            )
            all_ids.extend(ids)

        return all_ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        self._collection.delete(where={"document_id": ref_doc_id})

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        index_id: Optional[str] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Delete nodes from index.

        Args:
            node_ids (List[str]): list of node ids
            filters (List[MetadataFilters]): list of metadata filters

        """
        if not self._collection:
            raise ValueError("Collection not initialized")

        node_ids = node_ids or []

        if filters:
            where = _to_chroma_filter(filters)
        else:
            where = None

        where = {
            "$and": [
                {
                    {"__index_id__": index_id},
                    where
                }
            ]
        } if where is not None else {"__index_id__": index_id}

        self._collection.delete(ids=node_ids, where=where)

    def clear(self, index_id: Optional[str] = None) -> None:
        """Clear the collection."""

        ids = self._collection.get(where={"__index_id__": index_id})["ids"]
        self._collection.delete(ids=ids)

    @property
    def client(self) -> Any:
        """Return client."""
        return self._client

    def query(self, query: VectorStoreQuery, index_id: Optional[str] = None, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes

        """
        if query.filters is not None:
            if "where" in kwargs:
                raise ValueError(
                    "Cannot specify metadata filters via both query and kwargs. "
                    "Use kwargs only for chroma specific items that are "
                    "not supported via the generic query interface."
                )
            where = _to_chroma_filter(query.filters)
        else:
            where = kwargs.pop("where", None)

        if not query.query_embedding:
            return self._get(limit=query.similarity_top_k, where=where, index_id=index_id, **kwargs)

        return self._query(
            query_embeddings=query.query_embedding,
            n_results=query.similarity_top_k,
            where=where,
            index_id=index_id,
            **kwargs,
        )

    def _query(
        self, query_embeddings: List["float"], n_results: int, where: dict, index_id: str, **kwargs
    ) -> VectorStoreQueryResult:

        where = {
            "$and": [
                {"__index_id__": index_id},
                where
            ]
        } if where else {"__index_id__": index_id}

        results = self._collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            **kwargs,
        )

        logger.debug(f"> Top {len(results['documents'][0])} nodes:")
        nodes = []
        similarities = []
        ids = []
        for node_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            try:
                node = metadata_dict_to_node(metadata)
                node.set_content(text)
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                metadata, node_info, relationships = legacy_metadata_dict_to_node(
                    metadata
                )

                node = TextNode(
                    text=text,
                    id_=node_id,
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships=relationships,
                )

            nodes.append(node)

            similarity_score = math.exp(-distance)
            similarities.append(similarity_score)

            logger.debug(
                f"> [Node {node_id}] [Similarity score: {similarity_score}] "
                f"{truncate_text(str(text), 100)}"
            )
            ids.append(node_id)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def _get(
        self, limit: Optional[int], where: dict, index_id: str, **kwargs
    ) -> VectorStoreQueryResult:

        where = {
            "$and": [
                {
                    "__index_id__": index_id,
                },
                where
            ]
        } if where else {"__index_id__": index_id}

        results = self._collection.get(
            limit=limit,
            where=where,
            **kwargs,
        )

        logger.debug(f"> Top {len(results['documents'])} nodes:")
        nodes = []
        ids = []

        if not results["ids"]:
            results["ids"] = [[]]

        for node_id, text, metadata in zip(results["ids"], results["documents"], results["metadatas"]):
            try:
                metadata.pop("__index_id__")
                node = metadata_dict_to_node(metadata)
                node.set_content(text)
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                metadata, node_info, relationships = legacy_metadata_dict_to_node(
                    metadata
                )

                node = TextNode(
                    text=text,
                    id_=node_id,
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships=relationships,
                )

            nodes.append(node)

            logger.debug(
                f"> [Node {node_id}] [Similarity score: N/A - using get()] "
                f"{truncate_text(str(text), 100)}"
            )
            ids.append(node_id)

        return VectorStoreQueryResult(nodes=nodes, ids=ids)
