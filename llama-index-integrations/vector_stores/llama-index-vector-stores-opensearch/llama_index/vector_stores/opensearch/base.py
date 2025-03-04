"""Elasticsearch/Opensearch vector store."""

import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Union, cast

from llama_index.core.async_utils import asyncio_run
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from opensearchpy.client import Client as OSClient

IMPORT_OPENSEARCH_PY_ERROR = (
    "Could not import OpenSearch. Please install it with `pip install opensearch-py`."
)
IMPORT_ASYNC_OPENSEARCH_PY_ERROR = "Could not import AsyncOpenSearch. Please install it with `pip install opensearch-py`."
INVALID_HYBRID_QUERY_ERROR = (
    "Please specify the lexical_query and search_pipeline for hybrid search."
)
MATCH_ALL_QUERY = {"match_all": {}}  # type: Dict

VALID_DATA_TYPES = ["float", "byte", "binary"]
BYTE_VECTOR_ENGINES = ["lucene", "faiss"]
BINARY_VECTOR_ENGINE = "faiss"
INVALID_BYTE_VECTOR_ENGINE = (
    "Byte vectors only support 'lucene' or 'faiss' as the engine type."
)
INVALID_DATA_TYPE = f"Data type must be one of {VALID_DATA_TYPES}"
INVALID_BINARY_ENGINE = "Binary vectors must use 'faiss' as the engine type"
INVALID_BINARY_SPACE_TYPE = "Binary vectors must use 'hamming' as the space type"


class OpensearchVectorClient:
    """
    Object encapsulating an Opensearch index that has vector search enabled.

    If the index does not yet exist, it is created during init.
    Therefore, the underlying index is assumed to either:
    1) not exist yet or 2) be created due to previous usage of this class.

    Args:
        endpoint (str): URL (http/https) of elasticsearch endpoint
        index (str): Name of the elasticsearch index
        dim (int): Dimension of the vector
        embedding_field (str): Name of the field in the index to store
            embedding array in.
        text_field (str): Name of the field to grab text from
        data_type (str): Type of vector data. One of ["float", "byte", "binary"]
        method (Optional[dict]): Opensearch "method" JSON obj for configuring
            the KNN index.
            This includes engine, metric, and other config params.
        space_type (Optional[str]): space type for distance metric calculation.
        **kwargs: Optional arguments passed to the OpenSearch client.
    """

    def __init__(
        self,
        endpoint: str,
        index: str,
        dim: int,
        embedding_field: str = "embedding",
        text_field: str = "content",
        data_type: str = "float",
        method: Optional[dict] = None,
        settings: Optional[dict] = None,
        engine: Optional[str] = "nmslib",
        space_type: Optional[str] = "l2",
        max_chunk_bytes: int = 1 * 1024 * 1024,
        search_pipeline: Optional[str] = None,
        os_client: Optional[OSClient] = None,
        os_async_client: Optional[OSClient] = None,
        **kwargs: Any,
    ):
        """Init params."""
        if method is not None:
            engine = method.get("engine", engine)
            space_type = method.get("space_type", space_type)

        if data_type not in VALID_DATA_TYPES:
            raise ValueError(INVALID_DATA_TYPE)

        if data_type == "byte" and engine not in BYTE_VECTOR_ENGINES:
            raise ValueError(INVALID_BYTE_VECTOR_ENGINE)

        if data_type == "binary":
            if engine != BINARY_VECTOR_ENGINE:
                raise ValueError(INVALID_BINARY_ENGINE)
            if space_type != "hamming":
                raise ValueError(INVALID_BINARY_SPACE_TYPE)
        # Default method configuration
        if method is None:
            method = {
                "name": "hnsw",
                "space_type": space_type,
                "engine": engine,
                "parameters": {"ef_construction": 256, "m": 48},
            }
        if settings is None:
            settings = {"index": {"knn": True, "knn.algo_param.ef_search": 100}}
        if embedding_field is None:
            embedding_field = "embedding"

        self._method = method
        self._embedding_field = embedding_field
        self._endpoint = endpoint
        self._dim = dim
        self._index = index
        self._text_field = text_field
        self._max_chunk_bytes = max_chunk_bytes
        self._data_type = data_type

        self._search_pipeline = search_pipeline
        http_auth = kwargs.get("http_auth")
        self.space_type = space_type
        self.is_aoss = self._is_aoss_enabled(http_auth=http_auth)
        # initialize mapping
        idx_conf = {
            "settings": settings,
            "mappings": {
                "properties": {
                    embedding_field: {
                        "type": "knn_vector",
                        "dimension": dim,
                        "data_type": data_type,
                        "method": method,
                    },
                }
            },
        }
        self._os_client = os_client or self._get_opensearch_client(
            self._endpoint, **kwargs
        )
        self._os_async_client = os_async_client or self._get_async_opensearch_client(
            self._endpoint, **kwargs
        )
        self._efficient_filtering_enabled = self._is_efficient_filtering_enabled()
        not_found_error = self._import_not_found_error()

        try:
            self._os_client.indices.get(index=self._index)
        except TypeError:
            # Probably using async so switch to async client
            try:
                asyncio_run(self._os_async_client.indices.get(index=self._index))
            except not_found_error:
                asyncio_run(
                    self._os_async_client.indices.create(
                        index=self._index, body=idx_conf
                    )
                )
                asyncio_run(self._os_async_client.indices.refresh(index=self._index))
        except not_found_error:
            self._os_client.indices.create(index=self._index, body=idx_conf)
            self._os_client.indices.refresh(index=self._index)

    def _import_opensearch(self) -> Any:
        """Import OpenSearch if available, otherwise raise error."""
        try:
            from opensearchpy import OpenSearch
        except ImportError:
            raise ImportError(IMPORT_OPENSEARCH_PY_ERROR)
        return OpenSearch

    def _import_async_opensearch(self) -> Any:
        """Import AsyncOpenSearch if available, otherwise raise error."""
        try:
            from opensearchpy import AsyncOpenSearch
        except ImportError:
            raise ImportError(IMPORT_ASYNC_OPENSEARCH_PY_ERROR)
        return AsyncOpenSearch

    def _import_bulk(self) -> Any:
        """Import bulk if available, otherwise raise error."""
        try:
            from opensearchpy.helpers import bulk
        except ImportError:
            raise ImportError(IMPORT_OPENSEARCH_PY_ERROR)
        return bulk

    def _import_async_bulk(self) -> Any:
        """Import async_bulk if available, otherwise raise error."""
        try:
            from opensearchpy.helpers import async_bulk
        except ImportError:
            raise ImportError(IMPORT_ASYNC_OPENSEARCH_PY_ERROR)
        return async_bulk

    def _import_not_found_error(self) -> Any:
        """Import not found error if available, otherwise raise error."""
        try:
            from opensearchpy.exceptions import NotFoundError
        except ImportError:
            raise ImportError(IMPORT_OPENSEARCH_PY_ERROR)
        return NotFoundError

    def _get_opensearch_client(self, opensearch_url: str, **kwargs: Any) -> Any:
        """Get OpenSearch client from the opensearch_url, otherwise raise error."""
        try:
            opensearch = self._import_opensearch()
            client = opensearch(opensearch_url, **kwargs)
        except ValueError as e:
            raise ImportError(
                f"OpenSearch client string provided is not in proper format. "
                f"Got error: {e} "
            )
        return client

    def _get_async_opensearch_client(self, opensearch_url: str, **kwargs: Any) -> Any:
        """Get AsyncOpenSearch client from the opensearch_url, otherwise raise error."""
        try:
            opensearch = self._import_async_opensearch()
            client = opensearch(opensearch_url, **kwargs)

        except ValueError as e:
            raise ValueError(
                f"AsyncOpenSearch client string provided is not in proper format. "
                f"Got error: {e} "
            )
        return client

    def _get_opensearch_version(self) -> str:
        info = self._os_client.info()
        return info["version"]["number"]

    def _bulk_ingest_embeddings(
        self,
        client: Any,
        index_name: str,
        embeddings: List[List[float]],
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        vector_field: str = "embedding",
        text_field: str = "content",
        mapping: Optional[Dict] = None,
        max_chunk_bytes: Optional[int] = 1 * 1024 * 1024,
        is_aoss: bool = False,
    ) -> List[str]:
        """Bulk Ingest Embeddings into given index."""
        if not mapping:
            mapping = {}

        bulk = self._import_bulk()
        not_found_error = self._import_not_found_error()
        requests = []
        return_ids = []

        try:
            client.indices.get(index=index_name)
        except not_found_error:
            client.indices.create(index=index_name, body=mapping)

        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            _id = ids[i] if ids else str(uuid.uuid4())
            request = {
                "_op_type": "index",
                "_index": index_name,
                vector_field: embeddings[i],
                text_field: text,
                "metadata": metadata,
            }
            if is_aoss:
                request["id"] = _id
            else:
                request["_id"] = _id
            requests.append(request)
            return_ids.append(_id)

        bulk(client, requests, max_chunk_bytes=max_chunk_bytes)
        if not is_aoss:
            client.indices.refresh(index=index_name)

        return return_ids

    async def _abulk_ingest_embeddings(
        self,
        client: Any,
        index_name: str,
        embeddings: List[List[float]],
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        vector_field: str = "embedding",
        text_field: str = "content",
        mapping: Optional[Dict] = None,
        max_chunk_bytes: Optional[int] = 1 * 1024 * 1024,
        is_aoss: bool = False,
    ) -> List[str]:
        """Async Bulk Ingest Embeddings into given index."""
        if not mapping:
            mapping = {}

        async_bulk = self._import_async_bulk()
        not_found_error = self._import_not_found_error()
        requests = []
        return_ids = []

        try:
            await client.indices.get(index=index_name)
        except not_found_error:
            await client.indices.create(index=index_name, body=mapping)

        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            _id = ids[i] if ids else str(uuid.uuid4())
            request = {
                "_op_type": "index",
                "_index": index_name,
                vector_field: embeddings[i],
                text_field: text,
                "metadata": metadata,
            }
            if is_aoss:
                request["id"] = _id
            else:
                request["_id"] = _id
            requests.append(request)
            return_ids.append(_id)

        await async_bulk(client, requests, max_chunk_bytes=max_chunk_bytes)
        if not is_aoss:
            await client.indices.refresh(index=index_name)

        return return_ids

    def _default_approximate_search_query(
        self,
        query_vector: List[float],
        k: int = 4,
        filters: Optional[Union[Dict, List]] = None,
        vector_field: str = "embedding",
    ) -> Dict:
        """For Approximate k-NN Search, this is the default query."""
        query = {
            "size": k,
            "query": {
                "knn": {
                    vector_field: {
                        "vector": query_vector,
                        "k": k,
                    }
                }
            },
        }

        if filters:
            # filter key must be added only when filtering to avoid "filter doesn't support values of type: START_ARRAY" exception
            query["query"]["knn"][vector_field]["filter"] = filters
        return query

    def _is_text_field(self, value: Any) -> bool:
        """Check if value is a string and keyword filtering needs to be performed.

        Not applied to datetime strings.
        """
        if isinstance(value, str):
            try:
                datetime.fromisoformat(value)
                return False
            except ValueError as e:
                return True
        else:
            return False

    def _parse_filter(self, filter: MetadataFilter) -> dict:
        """Parse a single MetadataFilter to equivalent OpenSearch expression.

        As Opensearch does not differentiate between scalar/array keyword fields, IN and ANY are equivalent.
        """
        key = f"metadata.{filter.key}"
        op = filter.operator

        equality_postfix = ".keyword" if self._is_text_field(value=filter.value) else ""

        if op == FilterOperator.EQ:
            return {"term": {f"{key}{equality_postfix}": filter.value}}
        elif op in [
            FilterOperator.GT,
            FilterOperator.GTE,
            FilterOperator.LT,
            FilterOperator.LTE,
        ]:
            return {"range": {key: {filter.operator.name.lower(): filter.value}}}
        elif op == FilterOperator.NE:
            return {
                "bool": {
                    "must_not": {"term": {f"{key}{equality_postfix}": filter.value}}
                }
            }
        elif op in [FilterOperator.IN, FilterOperator.ANY]:
            if isinstance(filter.value, list) and all(
                self._is_text_field(val) for val in filter.value
            ):
                return {"terms": {f"{key}.keyword": filter.value}}
            else:
                return {"terms": {key: filter.value}}
        elif op == FilterOperator.NIN:
            return {"bool": {"must_not": {"terms": {key: filter.value}}}}
        elif op == FilterOperator.ALL:
            return {
                "terms_set": {
                    key: {
                        "terms": filter.value,
                        "minimum_should_match_script": {"source": "params.num_terms"},
                    }
                }
            }
        elif op == FilterOperator.TEXT_MATCH:
            return {"match": {key: {"query": filter.value, "fuzziness": "AUTO"}}}
        elif op == FilterOperator.CONTAINS:
            return {"wildcard": {key: f"*{filter.value}*"}}
        else:
            raise ValueError(f"Unsupported filter operator: {filter.operator}")

    def _parse_filters_recursively(self, filters: MetadataFilters) -> dict:
        """Parse (possibly nested) MetadataFilters to equivalent OpenSearch expression."""
        condition_map = {FilterCondition.AND: "must", FilterCondition.OR: "should"}

        bool_clause = condition_map[filters.condition]
        bool_query: dict[str, dict[str, list[dict]]] = {"bool": {bool_clause: []}}

        for filter_item in filters.filters:
            if isinstance(filter_item, MetadataFilter):
                bool_query["bool"][bool_clause].append(self._parse_filter(filter_item))
            elif isinstance(filter_item, MetadataFilters):
                bool_query["bool"][bool_clause].append(
                    self._parse_filters_recursively(filter_item)
                )
            else:
                raise ValueError(f"Unsupported filter type: {type(filter_item)}")

        return bool_query

    def _parse_filters(self, filters: Optional[MetadataFilters]) -> List[dict]:
        """Parse MetadataFilters to equivalent OpenSearch expression."""
        if filters is None:
            return []
        return [self._parse_filters_recursively(filters=filters)]

    def _knn_search_query(
        self,
        embedding_field: str,
        query_embedding: List[float],
        k: int,
        filters: Optional[MetadataFilters] = None,
        search_method="approximate",
    ) -> Dict:
        """
        Perform a k-Nearest Neighbors (kNN) search.

        If the search method is "approximate" and the engine is "lucene" or "faiss", use efficient kNN filtering.
        Otherwise, perform an exhaustive exact kNN search using "painless scripting" if the version of
        OpenSearch supports it. If the OpenSearch version does not support it, use scoring script search.

        Note:
            - AWS OpenSearch Serverless does not support the painless scripting functionality at this time according to AWS.
            - Approximate kNN search does not support pre-filtering.

        Args:
            query_embedding (List[float]): Vector embedding to query.
            k (int): Maximum number of results.
            filters (Optional[MetadataFilters]): Optional filters to apply for the search.
                Supports filter-context queries documented at
                https://opensearch.org/docs/latest/query-dsl/query-filter-context/

        Returns:
            Dict: Up to k documents closest to query_embedding.
        """
        filters = self._parse_filters(filters)

        if not filters:
            search_query = self._default_approximate_search_query(
                query_embedding,
                k,
                vector_field=embedding_field,
            )
        elif (
            search_method == "approximate"
            and self._method["engine"]
            in [
                "lucene",
                "faiss",
            ]
            and self._efficient_filtering_enabled
        ):
            # if engine is lucene or faiss, opensearch recommends efficient-kNN filtering.
            search_query = self._default_approximate_search_query(
                query_embedding,
                k,
                filters={"bool": {"filter": filters}},
                vector_field=embedding_field,
            )
        else:
            if self.is_aoss:
                # if is_aoss is set we are using Opensearch Serverless AWS offering which cannot use
                # painless scripting so default scoring script returned will be just normal knn_score script
                search_query = self._default_scoring_script_query(
                    query_embedding,
                    k,
                    space_type=self.space_type,
                    pre_filter={"bool": {"filter": filters}},
                    vector_field=embedding_field,
                )
            else:
                # https://opensearch.org/docs/latest/search-plugins/knn/painless-functions/
                search_query = self._default_scoring_script_query(
                    query_embedding,
                    k,
                    space_type="l2Squared",
                    pre_filter={"bool": {"filter": filters}},
                    vector_field=embedding_field,
                )
        return search_query

    def _hybrid_search_query(
        self,
        text_field: str,
        query_str: str,
        embedding_field: str,
        query_embedding: List[float],
        k: int,
        filters: Optional[MetadataFilters] = None,
    ) -> Dict:
        knn_query = self._knn_search_query(embedding_field, query_embedding, k, filters)
        lexical_query = self._lexical_search_query(text_field, query_str, k, filters)

        return {
            "size": k,
            "query": {
                "hybrid": {"queries": [lexical_query["query"], knn_query["query"]]}
            },
        }

    def _lexical_search_query(
        self,
        text_field: str,
        query_str: str,
        k: int,
        filters: Optional[MetadataFilters] = None,
    ) -> Dict:
        lexical_query = {
            "bool": {"must": {"match": {text_field: {"query": query_str}}}}
        }

        parsed_filters = self._parse_filters(filters)
        if len(parsed_filters) > 0:
            lexical_query["bool"]["filter"] = parsed_filters

        return {
            "size": k,
            "query": lexical_query,
        }

    def __get_painless_scripting_source(
        self, space_type: str, vector_field: str = "embedding"
    ) -> str:
        """For Painless Scripting, it returns the script source based on space type.
        This does not work with Opensearch Serverless currently.
        """
        source_value = (
            f"(1.0 + {space_type}(params.query_value, doc['{vector_field}']))"
        )
        if space_type == "cosineSimilarity":
            return source_value
        else:
            return f"1/{source_value}"

    def _get_knn_scoring_script(self, space_type, vector_field, query_vector):
        """Default scoring script that will work with AWS Opensearch Serverless."""
        return {
            "source": "knn_score",
            "lang": "knn",
            "params": {
                "field": vector_field,
                "query_value": query_vector,
                "space_type": space_type,
            },
        }

    def _get_painless_scoring_script(self, space_type, vector_field, query_vector):
        source = self.__get_painless_scripting_source(space_type, vector_field)
        return {
            "source": source,
            "params": {
                "field": vector_field,
                "query_value": query_vector,
            },
        }

    def _default_scoring_script_query(
        self,
        query_vector: List[float],
        k: int = 4,
        space_type: str = "l2Squared",
        pre_filter: Optional[Union[Dict, List]] = None,
        vector_field: str = "embedding",
    ) -> Dict:
        """For Scoring Script Search, this is the default query. Has to account for Opensearch Service
        Serverless which does not support painless scripting functions so defaults to knn_score.
        """
        if not pre_filter:
            pre_filter = MATCH_ALL_QUERY

        # check if we can use painless scripting or have to use default knn_score script
        if self.is_aoss:
            if space_type == "l2Squared":
                raise ValueError(
                    "Unsupported space type for aoss. Can only use l1, l2, cosinesimil."
                )
            script = self._get_knn_scoring_script(
                space_type, vector_field, query_vector
            )
        else:
            script = self._get_painless_scoring_script(
                space_type, vector_field, query_vector
            )
        return {
            "size": k,
            "query": {
                "script_score": {
                    "query": pre_filter,
                    "script": script,
                }
            },
        }

    def _is_aoss_enabled(self, http_auth: Any) -> bool:
        """Check if the service is http_auth is set as `aoss`."""
        if (
            http_auth is not None
            and hasattr(http_auth, "service")
            and http_auth.service == "aoss"
        ):
            return True
        return False

    def _is_efficient_filtering_enabled(self) -> bool:
        """Check if kNN with efficient filtering is enabled."""
        # Technically, AOSS supports efficient filtering,
        # but we can't check the version number using .info(); AOSS doesn't support 'GET /'
        #  so we must skip and disable by default.
        if self.is_aoss:
            ef_enabled = False
        else:
            self._os_version = self._get_opensearch_version()
            major, minor, patch = self._os_version.split(".")
            ef_enabled = int(major) >= 2 and int(minor) >= 9
        return ef_enabled

    def index_results(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        """Store results in the index."""
        embeddings: List[List[float]] = []
        texts: List[str] = []
        metadatas: List[dict] = []
        ids: List[str] = []
        for node in nodes:
            ids.append(node.node_id)
            embeddings.append(node.get_embedding())
            texts.append(node.get_content(metadata_mode=MetadataMode.NONE))
            metadatas.append(node_to_metadata_dict(node, remove_text=True))

        return self._bulk_ingest_embeddings(
            self._os_client,
            self._index,
            embeddings,
            texts,
            metadatas=metadatas,
            ids=ids,
            vector_field=self._embedding_field,
            text_field=self._text_field,
            mapping=None,
            max_chunk_bytes=self._max_chunk_bytes,
            is_aoss=self.is_aoss,
        )

    async def aindex_results(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        """Store results in the index."""
        embeddings: List[List[float]] = []
        texts: List[str] = []
        metadatas: List[dict] = []
        ids: List[str] = []
        for node in nodes:
            ids.append(node.node_id)
            embeddings.append(node.get_embedding())
            texts.append(node.get_content(metadata_mode=MetadataMode.NONE))
            metadatas.append(node_to_metadata_dict(node, remove_text=True))

        return await self._abulk_ingest_embeddings(
            self._os_async_client,
            self._index,
            embeddings,
            texts,
            metadatas=metadatas,
            ids=ids,
            vector_field=self._embedding_field,
            text_field=self._text_field,
            mapping=None,
            max_chunk_bytes=self._max_chunk_bytes,
            is_aoss=self.is_aoss,
        )

    def delete_by_doc_id(self, doc_id: str) -> None:
        """
        Deletes all OpenSearch documents corresponding to the given LlamaIndex `Document` ID.

        Args:
            doc_id (str): a LlamaIndex `Document` id
        """
        search_query = {
            "query": {"term": {"metadata.doc_id.keyword": {"value": doc_id}}}
        }
        self._os_client.delete_by_query(
            index=self._index, body=search_query, refresh=True
        )

    async def adelete_by_doc_id(self, doc_id: str) -> None:
        """
        Deletes all OpenSearch documents corresponding to the given LlamaIndex `Document` ID.

        Args:
            doc_id (str): a LlamaIndex `Document` id
        """
        search_query = {
            "query": {"term": {"metadata.doc_id.keyword": {"value": doc_id}}}
        }
        await self._os_async_client.delete_by_query(
            index=self._index, body=search_query, refresh=True
        )

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Deletes nodes.

        Args:
            node_ids (Optional[List[str]], optional): IDs of nodes to delete. Defaults to None.
            filters (Optional[MetadataFilters], optional): Metadata filters. Defaults to None.
        """
        if not node_ids and not filters:
            return

        query = {"query": {"bool": {"filter": []}}}
        if node_ids:
            query["query"]["bool"]["filter"].append({"terms": {"_id": node_ids or []}})

        if filters:
            query["query"]["bool"]["filter"].extend(self._parse_filters(filters))

        self._os_client.delete_by_query(index=self._index, body=query, refresh=True)

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Deletes nodes.

        Args:
            node_ids (Optional[List[str]], optional): IDs of nodes to delete. Defaults to None.
            filters (Optional[MetadataFilters], optional): Metadata filters. Defaults to None.
        """
        if not node_ids and not filters:
            return

        query = {"query": {"bool": {"filter": []}}}
        if node_ids:
            query["query"]["bool"]["filter"].append({"terms": {"_id": node_ids or []}})

        if filters:
            query["query"]["bool"]["filter"].extend(self._parse_filters(filters))

        await self._os_async_client.delete_by_query(
            index=self._index, body=query, refresh=True
        )

    def clear(self) -> None:
        """Clears index."""
        query = {"query": {"bool": {"filter": []}}}
        self._os_client.delete_by_query(index=self._index, body=query, refresh=True)

    async def aclear(self) -> None:
        """Clears index."""
        query = {"query": {"bool": {"filter": []}}}
        await self._os_async_client.delete_by_query(
            index=self._index, body=query, refresh=True
        )

    def query(
        self,
        query_mode: VectorStoreQueryMode,
        query_str: Optional[str],
        query_embedding: List[float],
        k: int,
        filters: Optional[MetadataFilters] = None,
    ) -> VectorStoreQueryResult:
        if query_mode == VectorStoreQueryMode.HYBRID:
            if query_str is None or self._search_pipeline is None:
                raise ValueError(INVALID_HYBRID_QUERY_ERROR)
            search_query = self._hybrid_search_query(
                self._text_field,
                query_str,
                self._embedding_field,
                query_embedding,
                k,
                filters=filters,
            )
            params = {
                "search_pipeline": self._search_pipeline,
            }
        elif query_mode == VectorStoreQueryMode.TEXT_SEARCH:
            search_query = self._lexical_search_query(
                self._text_field, query_str, k, filters=filters
            )
            params = None
        else:
            search_query = self._knn_search_query(
                self._embedding_field, query_embedding, k, filters=filters
            )
            params = None

        res = self._os_client.search(
            index=self._index, body=search_query, params=params
        )

        return self._to_query_result(res)

    async def aquery(
        self,
        query_mode: VectorStoreQueryMode,
        query_str: Optional[str],
        query_embedding: List[float],
        k: int,
        filters: Optional[MetadataFilters] = None,
    ) -> VectorStoreQueryResult:
        if query_mode == VectorStoreQueryMode.HYBRID:
            if query_str is None or self._search_pipeline is None:
                raise ValueError(INVALID_HYBRID_QUERY_ERROR)
            search_query = self._hybrid_search_query(
                self._text_field,
                query_str,
                self._embedding_field,
                query_embedding,
                k,
                filters=filters,
            )
            params = {
                "search_pipeline": self._search_pipeline,
            }
        elif query_mode == VectorStoreQueryMode.TEXT_SEARCH:
            search_query = self._lexical_search_query(
                self._text_field, query_str, k, filters=filters
            )
            params = None
        else:
            search_query = self._knn_search_query(
                self._embedding_field, query_embedding, k, filters=filters
            )
            params = None

        res = await self._os_async_client.search(
            index=self._index, body=search_query, params=params
        )

        return self._to_query_result(res)

    def _to_query_result(self, res) -> VectorStoreQueryResult:
        nodes = []
        ids = []
        scores = []
        for hit in res["hits"]["hits"]:
            source = hit["_source"]
            node_id = hit["_id"]
            text = source[self._text_field]
            metadata = source.get("metadata", None)

            try:
                node = metadata_dict_to_node(metadata)
                node.text = text
            except Exception:
                # TODO: Legacy support for old nodes
                node_info = source.get("node_info")
                relationships = source.get("relationships") or {}
                start_char_idx = None
                end_char_idx = None
                if isinstance(node_info, dict):
                    start_char_idx = node_info.get("start", None)
                    end_char_idx = node_info.get("end", None)

                node = TextNode(
                    text=text,
                    metadata=metadata,
                    id_=node_id,
                    start_char_idx=start_char_idx,
                    end_char_idx=end_char_idx,
                    relationships=relationships,
                    extra_info=source,
                )
            ids.append(node_id)
            nodes.append(node)
            scores.append(hit["_score"])

        return VectorStoreQueryResult(nodes=nodes, ids=ids, similarities=scores)


class OpensearchVectorStore(BasePydanticVectorStore):
    """
    Elasticsearch/Opensearch vector store.

    Args:
        client (OpensearchVectorClient): Vector index client to use
            for data insertion/querying.

    Examples:
        `pip install llama-index-vector-stores-opensearch`

        ```python
        from llama_index.vector_stores.opensearch import (
            OpensearchVectorStore,
            OpensearchVectorClient,
        )

        # http endpoint for your cluster (opensearch required for vector index usage)
        endpoint = "http://localhost:9200"
        # index to demonstrate the VectorStore impl
        idx = "gpt-index-demo"

        # OpensearchVectorClient stores text in this field by default
        text_field = "content"
        # OpensearchVectorClient stores embeddings in this field by default
        embedding_field = "embedding"

        # OpensearchVectorClient encapsulates logic for a
        # single opensearch index with vector search enabled
        client = OpensearchVectorClient(
            endpoint, idx, 1536, embedding_field=embedding_field, text_field=text_field
        )

        # initialize vector store
        vector_store = OpensearchVectorStore(client)
        ```
    """

    stores_text: bool = True
    _client: OpensearchVectorClient = PrivateAttr(default=None)

    def __init__(
        self,
        client: OpensearchVectorClient,
    ) -> None:
        """Initialize params."""
        super().__init__()
        self._client = client

    @property
    def client(self) -> Any:
        """Get client."""
        return self._client

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings.

        """
        self._client.index_results(nodes)
        return [result.node_id for result in nodes]

    async def async_add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Async add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings.

        """
        await self._client.aindex_results(nodes)
        return [result.node_id for result in nodes]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        self._client.delete_by_doc_id(ref_doc_id)

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Async delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        await self._client.adelete_by_doc_id(ref_doc_id)

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Deletes nodes async.

        Args:
            node_ids (Optional[List[str]], optional): IDs of nodes to delete. Defaults to None.
            filters (Optional[MetadataFilters], optional): Metadata filters. Defaults to None.
        """
        self._client.delete_nodes(node_ids, filters, **delete_kwargs)

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Async deletes nodes async.

        Args:
            node_ids (Optional[List[str]], optional): IDs of nodes to delete. Defaults to None.
            filters (Optional[MetadataFilters], optional): Metadata filters. Defaults to None.
        """
        await self._client.adelete_nodes(node_ids, filters, **delete_kwargs)

    def clear(self) -> None:
        """Clears index."""
        self._client.clear()

    async def aclear(self) -> None:
        """Async clears index."""
        await self._client.aclear()

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query index for top k most similar nodes.

        Args:
            query (VectorStoreQuery): Store query object.

        """
        query_embedding = cast(List[float], query.query_embedding)

        return self._client.query(
            query.mode,
            query.query_str,
            query_embedding,
            query.similarity_top_k,
            filters=query.filters,
        )

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """
        Async query index for top k most similar nodes.

        Args:
            query (VectorStoreQuery): Store query object.

        """
        query_embedding = cast(List[float], query.query_embedding)

        return await self._client.aquery(
            query.mode,
            query.query_str,
            query_embedding,
            query.similarity_top_k,
            filters=query.filters,
        )
