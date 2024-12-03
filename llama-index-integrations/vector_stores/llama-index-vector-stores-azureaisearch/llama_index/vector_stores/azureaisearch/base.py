"""Azure AI Search vector store."""

import enum
import json
import logging
from enum import auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

from azure.search.documents import SearchClient
from azure.search.documents.aio import SearchClient as AsyncSearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.aio import (
    SearchIndexClient as AsyncSearchIndexClient,
)
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from llama_index.vector_stores.azureaisearch.azureaisearch_utils import (
    create_node_from_result,
    process_batch_results,
    create_search_request,
    handle_search_error,
)

logger = logging.getLogger(__name__)


class MetadataIndexFieldType(int, enum.Enum):
    """
    Enumeration representing the supported types for metadata fields in an
    Azure AI Search Index, corresponds with types supported in a flat
    metadata dictionary.
    """

    STRING = auto()
    BOOLEAN = auto()
    INT32 = auto()
    INT64 = auto()
    DOUBLE = auto()
    COLLECTION = auto()


class IndexManagement(int, enum.Enum):
    """Enumeration representing the supported index management operations."""

    NO_VALIDATION = auto()
    VALIDATE_INDEX = auto()
    CREATE_IF_NOT_EXISTS = auto()


DEFAULT_MAX_BATCH_SIZE = 700
DEFAULT_MAX_MB_SIZE = 14 * 1024 * 1024  # 14MB in bytes


class AzureAISearchVectorStore(BasePydanticVectorStore):
    """
    Azure AI Search vector store.

    Examples:
        `pip install llama-index-vector-stores-azureaisearch`

        ```python
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient
        from azure.search.documents.indexes import SearchIndexClient
        from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore
        from llama_index.vector_stores.azureaisearch import IndexManagement, MetadataIndexFieldType

        # Azure AI Search setup
        search_service_api_key = "YOUR-AZURE-SEARCH-SERVICE-ADMIN-KEY"
        search_service_endpoint = "YOUR-AZURE-SEARCH-SERVICE-ENDPOINT"
        search_service_api_version = "2024-07-01"
        credential = AzureKeyCredential(search_service_api_key)

        # Index name to use
        index_name = "llamaindex-vector-demo"

        # Use index client to demonstrate creating an index
        index_client = SearchIndexClient(
            endpoint=search_service_endpoint,
            credential=credential,
        )

        metadata_fields = {
            "author": "author",
            "theme": ("topic", MetadataIndexFieldType.STRING),
            "director": "director",
        }

        # Creating an Azure AI Search Vector Store
        vector_store = AzureAISearchVectorStore(
            search_or_index_client=index_client,
            filterable_metadata_field_keys=metadata_fields,
            hidden_field_keys=["embedding"],
            index_name=index_name,
            index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
            id_field_key="id",
            chunk_field_key="chunk",
            embedding_field_key="embedding",
            embedding_dimensionality=1536,
            metadata_string_field_key="metadata",
            doc_id_field_key="doc_id",
            language_analyzer="en.lucene",
            vector_algorithm_type="exhaustiveKnn",
        )
        ```
    """

    stores_text: bool = True
    flat_metadata: bool = False

    _index_client: SearchIndexClient = PrivateAttr()
    _index_name: Optional[str] = PrivateAttr()
    _async_index_client: AsyncSearchIndexClient = PrivateAttr()
    _search_client: SearchClient = PrivateAttr()
    _async_search_client: AsyncSearchClient = PrivateAttr()
    _embedding_dimensionality: int = PrivateAttr()
    _language_analyzer: str = PrivateAttr()
    _hidden_field_keys: List[str] = PrivateAttr()
    _field_mapping: Dict[str, str] = PrivateAttr()
    _index_management: IndexManagement = PrivateAttr()
    _index_mapping: Callable[
        [Dict[str, str], Dict[str, Any]], Dict[str, str]
    ] = PrivateAttr()
    _metadata_to_index_field_map: Dict[
        str, Tuple[str, MetadataIndexFieldType]
    ] = PrivateAttr()
    _vector_profile_name: str = PrivateAttr()
    _compression_type: str = PrivateAttr()
    _user_agent: str = PrivateAttr()

    def _normalise_metadata_to_index_fields(
        self,
        filterable_metadata_field_keys: Union[
            List[str],
            Dict[str, str],
            Dict[str, Tuple[str, MetadataIndexFieldType]],
            None,
        ] = [],
    ) -> Dict[str, Tuple[str, MetadataIndexFieldType]]:
        index_field_spec: Dict[str, Tuple[str, MetadataIndexFieldType]] = {}

        if isinstance(filterable_metadata_field_keys, List):
            for field in filterable_metadata_field_keys:
                # Index field name and the metadata field name are the same
                # Use String as the default index field type
                index_field_spec[field] = (field, MetadataIndexFieldType.STRING)

        elif isinstance(filterable_metadata_field_keys, dict):
            for k, v in filterable_metadata_field_keys.items():
                if isinstance(v, tuple):
                    # Index field name and metadata field name may differ
                    # The index field type used is as supplied
                    index_field_spec[k] = v
                elif isinstance(v, list):
                    # Handle list types as COLLECTION
                    index_field_spec[k] = (k, MetadataIndexFieldType.COLLECTION)
                elif isinstance(v, bool):
                    index_field_spec[k] = (k, MetadataIndexFieldType.BOOLEAN)
                elif isinstance(v, int):
                    index_field_spec[k] = (k, MetadataIndexFieldType.INT32)
                elif isinstance(v, float):
                    index_field_spec[k] = (k, MetadataIndexFieldType.DOUBLE)
                elif isinstance(v, str):
                    index_field_spec[k] = (k, MetadataIndexFieldType.STRING)
                else:
                    # Index field name and metadata field name may differ
                    # Use String as the default index field type
                    index_field_spec[k] = (v, MetadataIndexFieldType.STRING)

        return index_field_spec

    def _index_exists(self, index_name: str) -> bool:
        return index_name in self._index_client.list_index_names()

    async def _aindex_exists(self, index_name: str) -> bool:
        return index_name in [
            name async for name in self._async_index_client.list_index_names()
        ]

    def _create_index_if_not_exists(self, index_name: str) -> None:
        if not self._index_exists(index_name):
            logger.info(
                f"Index {index_name} does not exist in Azure AI Search, creating index"
            )
            self._create_index(index_name)

    async def _acreate_index_if_not_exists(self, index_name: str) -> None:
        if not await self._aindex_exists(index_name):
            logger.info(
                f"Index {index_name} does not exist in Azure AI Search, creating index"
            )
            await self._acreate_index(index_name)

    def _create_metadata_index_fields(self) -> List[Any]:
        """Create a list of index fields for storing metadata values."""
        from azure.search.documents.indexes.models import SimpleField

        index_fields = []

        # create search fields
        for v in self._metadata_to_index_field_map.values():
            field_name, field_type = v

            # Skip if the field is already mapped
            if field_name in self._field_mapping.values():
                continue

            if field_type == MetadataIndexFieldType.STRING:
                index_field_type = "Edm.String"
            elif field_type == MetadataIndexFieldType.INT32:
                index_field_type = "Edm.Int32"
            elif field_type == MetadataIndexFieldType.INT64:
                index_field_type = "Edm.Int64"
            elif field_type == MetadataIndexFieldType.DOUBLE:
                index_field_type = "Edm.Double"
            elif field_type == MetadataIndexFieldType.BOOLEAN:
                index_field_type = "Edm.Boolean"
            elif field_type == MetadataIndexFieldType.COLLECTION:
                index_field_type = "Collection(Edm.String)"

            field = SimpleField(
                name=field_name,
                type=index_field_type,
                filterable=True,
                hidden=field_name in self._hidden_field_keys,
            )
            index_fields.append(field)

        return index_fields

    def _get_compressions(self) -> List[Any]:
        """Get the compressions for the vector search."""
        from azure.search.documents.indexes.models import (
            BinaryQuantizationCompression,
            ScalarQuantizationCompression,
        )

        compressions = []
        if self._compression_type == "binary":
            compressions.append(
                BinaryQuantizationCompression(compression_name="myBinaryCompression")
            )
        elif self._compression_type == "scalar":
            compressions.append(
                ScalarQuantizationCompression(compression_name="myScalarCompression")
            )
        return compressions

    def _create_index(self, index_name: Optional[str]) -> None:
        """
        Creates a default index based on the supplied index name, key field names and
        metadata filtering keys.
        """
        from azure.search.documents.indexes.models import (
            ExhaustiveKnnAlgorithmConfiguration,
            ExhaustiveKnnParameters,
            HnswAlgorithmConfiguration,
            HnswParameters,
            SearchableField,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SemanticConfiguration,
            SemanticField,
            SemanticPrioritizedFields,
            SemanticSearch,
            SimpleField,
            VectorSearch,
            VectorSearchAlgorithmKind,
            VectorSearchAlgorithmMetric,
            VectorSearchProfile,
        )

        logger.info(f"Configuring {index_name} fields for Azure AI Search")
        fields = [
            SimpleField(
                name=self._field_mapping["id"],
                type="Edm.String",
                key=True,
                filterable=True,
                hidden=self._field_mapping["id"] in self._hidden_field_keys,
            ),
            SimpleField(
                name=self._field_mapping["index_id"],
                type="Edm.String",
                filterable=True,
                hidden=self._field_mapping["index_id"] in self._hidden_field_keys,
            ),
            SearchableField(
                name=self._field_mapping["chunk"],
                type="Edm.String",
                analyzer_name=self._language_analyzer,
                hidden=self._field_mapping["chunk"] in self._hidden_field_keys,
            ),
            SearchField(
                name=self._field_mapping["embedding"],
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self._embedding_dimensionality,
                vector_search_profile_name=self._vector_profile_name,
                hidden=self._field_mapping["embedding"] in self._hidden_field_keys,
            ),
            SimpleField(
                name=self._field_mapping["metadata"],
                type="Edm.String",
                hidden=self._field_mapping["metadata"] in self._hidden_field_keys,
            ),
            SimpleField(
                name=self._field_mapping["doc_id"],
                type="Edm.String",
                filterable=True,
                hidden=self._field_mapping["doc_id"] in self._hidden_field_keys,
            ),
        ]
        logger.info(f"Configuring {index_name} metadata fields")
        metadata_index_fields = self._create_metadata_index_fields()
        fields.extend(metadata_index_fields)
        logger.info(f"Configuring {index_name} vector search")
        # Determine the compression type
        compressions = self._get_compressions()

        logger.info(
            f"Configuring {index_name} vector search with {self._compression_type} compression"
        )
        # Configure the vector search algorithms and profiles
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="myHnsw",
                    kind=VectorSearchAlgorithmKind.HNSW,
                    parameters=HnswParameters(
                        m=4,
                        ef_construction=400,
                        ef_search=500,
                        metric=VectorSearchAlgorithmMetric.COSINE,
                    ),
                ),
                ExhaustiveKnnAlgorithmConfiguration(
                    name="myExhaustiveKnn",
                    kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
                    parameters=ExhaustiveKnnParameters(
                        metric=VectorSearchAlgorithmMetric.COSINE,
                    ),
                ),
            ],
            compressions=compressions,
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfile",
                    algorithm_configuration_name="myHnsw",
                    compression_name=(
                        compressions[0].compression_name if compressions else None
                    ),
                ),
                VectorSearchProfile(
                    name="myExhaustiveKnnProfile",
                    algorithm_configuration_name="myExhaustiveKnn",
                    compression_name=None,  # Exhaustive KNN doesn't support compression at the moment
                ),
            ],
        )
        logger.info(f"Configuring {index_name} semantic search")
        semantic_config = SemanticConfiguration(
            name="mySemanticConfig",
            prioritized_fields=SemanticPrioritizedFields(
                content_fields=[SemanticField(field_name=self._field_mapping["chunk"])],
            ),
        )

        semantic_search = SemanticSearch(configurations=[semantic_config])

        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
        )

        logger.debug(f"Creating {index_name} search index")
        self._index_client.create_index(index)

    async def _acreate_index(self, index_name: Optional[str]) -> None:
        """
        Asynchronous version of index creation with optional compression.

            Creates a default index based on the supplied index name, key field names, and metadata filtering keys.
        """
        from azure.search.documents.indexes.models import (
            ExhaustiveKnnAlgorithmConfiguration,
            ExhaustiveKnnParameters,
            HnswAlgorithmConfiguration,
            HnswParameters,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SearchableField,
            SemanticConfiguration,
            SemanticField,
            SemanticPrioritizedFields,
            SemanticSearch,
            SimpleField,
            VectorSearch,
            VectorSearchAlgorithmKind,
            VectorSearchAlgorithmMetric,
            VectorSearchProfile,
        )

        logger.info(f"Configuring {index_name} fields for Azure AI Search")
        fields = [
            SimpleField(name=self._field_mapping["id"], type="Edm.String", key=True),
            SearchableField(
                name=self._field_mapping["chunk"],
                type="Edm.String",
                analyzer_name=self._language_analyzer,
                hidden=self._field_mapping["chunk"] in self._hidden_field_keys,
            ),
            SearchField(
                name=self._field_mapping["embedding"],
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self._embedding_dimensionality,
                vector_search_profile_name=self._vector_profile_name,
                hidden=self._field_mapping["embedding"] in self._hidden_field_keys,
            ),
            SimpleField(
                name=self._field_mapping["metadata"],
                type="Edm.String",
                hidden=self._field_mapping["metadata"] in self._hidden_field_keys,
            ),
            SimpleField(
                name=self._field_mapping["doc_id"],
                type="Edm.String",
                filterable=True,
                hidden=self._field_mapping["doc_id"] in self._hidden_field_keys,
            ),
        ]
        logger.info(f"Configuring {index_name} metadata fields")
        metadata_index_fields = self._create_metadata_index_fields()
        fields.extend(metadata_index_fields)
        # Determine the compression type
        compressions = self._get_compressions()

        logger.info(
            f"Configuring {index_name} vector search with {self._compression_type} compression"
        )
        # Configure the vector search algorithms and profiles
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="myHnsw",
                    kind=VectorSearchAlgorithmKind.HNSW,
                    # For more information on HNSw parameters, visit https://learn.microsoft.com//azure/search/vector-search-ranking#creating-the-hnsw-graph
                    parameters=HnswParameters(
                        m=4,
                        ef_construction=400,
                        ef_search=500,
                        metric=VectorSearchAlgorithmMetric.COSINE,
                    ),
                ),
                ExhaustiveKnnAlgorithmConfiguration(
                    name="myExhaustiveKnn",
                    kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
                    parameters=ExhaustiveKnnParameters(
                        metric=VectorSearchAlgorithmMetric.COSINE,
                    ),
                ),
            ],
            compressions=compressions,
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfile",
                    algorithm_configuration_name="myHnsw",
                    compression_name=(
                        compressions[0].compression_name if compressions else None
                    ),
                ),
                VectorSearchProfile(
                    name="myExhaustiveKnnProfile",
                    algorithm_configuration_name="myExhaustiveKnn",
                    compression_name=None,  # Exhaustive KNN doesn't support compression at the moment
                ),
            ],
        )
        logger.info(f"Configuring {index_name} semantic search")
        semantic_config = SemanticConfiguration(
            name="mySemanticConfig",
            prioritized_fields=SemanticPrioritizedFields(
                content_fields=[SemanticField(field_name=self._field_mapping["chunk"])],
            ),
        )

        semantic_search = SemanticSearch(configurations=[semantic_config])

        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
        )
        logger.debug(f"Creating {index_name} search index")

        await self._async_index_client.create_index(index)

    def _validate_index(self, index_name: Optional[str]) -> None:
        if self._index_client and index_name and not self._index_exists(index_name):
            raise ValueError(f"Validation failed, index {index_name} does not exist.")

    async def _avalidate_index(self, index_name: Optional[str]) -> None:
        if (
            self._async_index_client
            and index_name
            and not await self._aindex_exists(index_name)
        ):
            raise ValueError(f"Validation failed, index {index_name} does not exist.")

    def __init__(
        self,
        search_or_index_client: Any,
        id_field_key: str,
        chunk_field_key: str,
        embedding_field_key: str,
        metadata_string_field_key: str,
        doc_id_field_key: str,
        filterable_metadata_field_keys: Optional[
            Union[
                List[str],
                Dict[str, str],
                Dict[str, Tuple[str, MetadataIndexFieldType]],
            ]
        ] = None,
        hidden_field_keys: Optional[List[str]] = None,
        index_name: Optional[str] = None,
        index_mapping: Optional[
            Callable[[Dict[str, str], Dict[str, Any]], Dict[str, str]]
        ] = None,
        index_management: IndexManagement = IndexManagement.NO_VALIDATION,
        embedding_dimensionality: int = 1536,
        vector_algorithm_type: str = "exhaustiveKnn",
        # If we have content in other languages, it is better to enable the language analyzer to be adjusted in searchable fields.
        # https://learn.microsoft.com/en-us/azure/search/index-add-language-analyzers
        language_analyzer: str = "en.lucene",
        compression_type: str = "none",
        user_agent: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # ruff: noqa: E501
        """
        Embeddings and documents are stored in an Azure AI Search index,
        a merge or upload approach is used when adding embeddings.
        When adding multiple embeddings the index is updated by this vector store
        in batches of 10 documents, very large nodes may result in failure due to
        the batch byte size being exceeded.

        Args:
            search_client (azure.search.documents.SearchClient):
                Client for index to populated / queried.
            id_field_key (str): Index field storing the id
            chunk_field_key (str): Index field storing the node text
            embedding_field_key (str): Index field storing the embedding vector
            metadata_string_field_key (str):
                Index field storing node metadata as a json string.
                Schema is arbitrary, to filter on metadata values they must be stored
                as separate fields in the index, use filterable_metadata_field_keys
                to specify the metadata values that should be stored in these filterable fields
            doc_id_field_key (str): Index field storing doc_id
            hidden_field_keys (List[str]):
                List of index fields that should be hidden from the client.
                This is useful for fields that are not needed for retrieving,
                but are used for similarity search, like the embedding field.
            index_mapping:
                Optional function with definition
                (enriched_doc: Dict[str, str], metadata: Dict[str, Any]): Dict[str,str]
                used to map document fields to the AI search index fields
                (return value of function).
                If none is specified a default mapping is provided which uses
                the field keys. The keys in the enriched_doc are
                ["id", "chunk", "embedding", "metadata"]
                The default mapping is:
                    - "id" to id_field_key
                    - "chunk" to chunk_field_key
                    - "embedding" to embedding_field_key
                    - "metadata" to metadata_field_key
            *kwargs (Any): Additional keyword arguments.

        Raises:
            ImportError: Unable to import `azure-search-documents`
            ValueError: If `search_or_index_client` is not provided
            ValueError: If `index_name` is not provided and `search_or_index_client`
                is of type azure.search.documents.SearchIndexClient
            ValueError: If `index_name` is provided and `search_or_index_client`
                is of type azure.search.documents.SearchClient
            ValueError: If `create_index_if_not_exists` is true and
                `search_or_index_client` is of type azure.search.documents.SearchClient
        """
        import_err_msg = (
            "`azure-search-documents` package not found, please run "
            "`pip install azure-search-documents==11.4.0`"
        )

        try:
            import azure.search.documents  # noqa
            from azure.search.documents import SearchClient
            from azure.search.documents.indexes import SearchIndexClient
        except ImportError:
            raise ImportError(import_err_msg)

        super().__init__()
        base_user_agent = "llamaindex-python"
        self._user_agent = (
            f"{base_user_agent} {user_agent}" if user_agent else base_user_agent
        )

        self._index_client: SearchIndexClient = cast(SearchIndexClient, None)
        self._async_index_client: AsyncSearchIndexClient = cast(
            AsyncSearchIndexClient, None
        )
        self._search_client: SearchClient = cast(SearchClient, None)
        self._async_search_client: AsyncSearchClient = cast(AsyncSearchClient, None)
        self._embedding_dimensionality = embedding_dimensionality
        self._index_name = index_name

        if vector_algorithm_type == "exhaustiveKnn":
            self._vector_profile_name = "myExhaustiveKnnProfile"
        elif vector_algorithm_type == "hnsw":
            self._vector_profile_name = "myHnswProfile"
        else:
            raise ValueError(
                "Only 'exhaustiveKnn' and 'hnsw' are supported for vector_algorithm_type"
            )

        self._language_analyzer = language_analyzer
        self._compression_type = compression_type.lower()

        # Validate search_or_index_client
        if search_or_index_client is not None:
            if isinstance(search_or_index_client, SearchIndexClient):
                # If SearchIndexClient is supplied so must index_name
                self._index_client = cast(SearchIndexClient, search_or_index_client)
                self._index_client._client._config.user_agent_policy.add_user_agent(
                    self._user_agent
                )
                if not index_name:
                    raise ValueError(
                        "index_name must be supplied if search_or_index_client is of "
                        "type azure.search.documents.SearchIndexClient"
                    )

                self._search_client = self._index_client.get_search_client(
                    index_name=index_name
                )
                self._search_client._client._config.user_agent_policy.add_user_agent(
                    self._user_agent
                )

            elif isinstance(search_or_index_client, AsyncSearchIndexClient):
                # If SearchIndexClient is supplied so must index_name
                self._async_index_client = cast(
                    AsyncSearchIndexClient, search_or_index_client
                )
                self._async_index_client._client._config.user_agent_policy.add_user_agent(
                    self._user_agent
                )

                if not index_name:
                    raise ValueError(
                        "index_name must be supplied if search_or_index_client is of "
                        "type azure.search.documents.aio.SearchIndexClient"
                    )

                self._async_search_client = self._async_index_client.get_search_client(
                    index_name=index_name
                )
                self._async_search_client._client._config.user_agent_policy.add_user_agent(
                    self._user_agent
                )

            elif isinstance(search_or_index_client, SearchClient):
                self._search_client = cast(SearchClient, search_or_index_client)
                self._search_client._client._config.user_agent_policy.add_user_agent(
                    self._user_agent
                )
                # Validate index_name
                if index_name:
                    raise ValueError(
                        "index_name cannot be supplied if search_or_index_client "
                        "is of type azure.search.documents.SearchClient"
                    )

            elif isinstance(search_or_index_client, AsyncSearchClient):
                self._async_search_client = cast(
                    AsyncSearchClient, search_or_index_client
                )
                self._async_search_client._client._config.user_agent_policy.add_user_agent(
                    self._user_agent
                )

                # Validate index_name
                if index_name:
                    raise ValueError(
                        "index_name cannot be supplied if search_or_index_client "
                        "is of type azure.search.documents.SearchClient"
                    )

            if isinstance(search_or_index_client, AsyncSearchIndexClient):
                if not self._async_index_client and not self._async_search_client:
                    raise ValueError(
                        "search_or_index_client must be of type "
                        "azure.search.documents.SearchIndexClient or "
                        "azure.search.documents.SearchClient"
                    )

            if isinstance(search_or_index_client, SearchIndexClient):
                if not self._index_client and not self._search_client:
                    raise ValueError(
                        "search_or_index_client must be of type "
                        "azure.search.documents.SearchIndexClient or "
                        "azure.search.documents.SearchClient"
                    )
        else:
            raise ValueError("search_or_index_client not specified")

        if index_management == IndexManagement.CREATE_IF_NOT_EXISTS and not (
            self._index_client or self._async_index_client
        ):
            raise ValueError(
                "index_management has value of IndexManagement.CREATE_IF_NOT_EXISTS "
                "but search_or_index_client is not of type "
                "azure.search.documents.SearchIndexClient or azure.search.documents.aio.SearchIndexClient "
            )

        self._index_management = index_management

        # Default field mapping
        field_mapping = {
            "id": id_field_key,
            "index_id": "index_id",
            "chunk": chunk_field_key,
            "embedding": embedding_field_key,
            "metadata": metadata_string_field_key,
            "doc_id": doc_id_field_key,
        }

        self._field_mapping = field_mapping
        self._hidden_field_keys = hidden_field_keys or []

        self._index_mapping = (
            self._default_index_mapping if index_mapping is None else index_mapping
        )

        # self._filterable_metadata_field_keys = filterable_metadata_field_keys
        self._metadata_to_index_field_map = self._normalise_metadata_to_index_fields(
            filterable_metadata_field_keys
        )

        # need to do lazy init for async client
        if not isinstance(search_or_index_client, AsyncSearchIndexClient):
            if self._index_management == IndexManagement.CREATE_IF_NOT_EXISTS:
                if index_name:
                    self._create_index_if_not_exists(index_name)

            if self._index_management == IndexManagement.VALIDATE_INDEX:
                self._validate_index(index_name)

    @property
    def client(self) -> Any:
        """Get client."""
        return self._search_client

    @property
    def aclient(self) -> Any:
        """Get async client."""
        return self._async_search_client

    def _default_index_mapping(
        self, enriched_doc: Dict[str, str], metadata: Dict[str, Any]
    ) -> Dict[str, str]:
        index_doc: Dict[str, str] = {}

        for field in self._field_mapping:
            index_doc[self._field_mapping[field]] = enriched_doc[field]

        for metadata_field_name, (
            index_field_name,
            _,
        ) in self._metadata_to_index_field_map.items():
            metadata_value = metadata.get(metadata_field_name)
            if metadata_value:
                index_doc[index_field_name] = metadata_value

        return index_doc

    def move_nodes(self, from_index_id: str, to_index_id: str):
        nodes = self.get_nodes(index_id=from_index_id)
        updates = [
            {"id": n.id_, "index_id": to_index_id} for n in nodes
        ]
        self._search_client.merge_documents(updates)

    def add(
        self,
        nodes: List[BaseNode],
        index_id: Optional[str] = None,
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Add nodes to index associated with the configured search client.

        Args:
            nodes: List[BaseNode]: nodes with embeddings

        """
        from azure.search.documents import IndexDocumentsBatch

        if not self._search_client:
            raise ValueError("Search client not initialized")

        accumulator = IndexDocumentsBatch()
        documents = []

        ids = []
        accumulated_size = 0
        max_size = DEFAULT_MAX_MB_SIZE  # 16MB in bytes
        max_docs = DEFAULT_MAX_BATCH_SIZE

        for node in nodes:
            logger.debug(f"Processing embedding: {node.node_id}")
            ids.append(node.node_id)

            index_document = self._create_index_document(node, index_id)
            document_size = len(json.dumps(index_document).encode("utf-8"))
            documents.append(index_document)
            accumulated_size += document_size

            accumulator.add_upload_actions(index_document)

            if len(documents) >= max_docs or accumulated_size >= max_size:
                logger.info(
                    f"Uploading batch of size {len(documents)}, "
                    f"current progress {len(ids)} of {len(nodes)}, "
                    f"accumulated size {accumulated_size / (1024 * 1024):.2f} MB"
                )
                self._search_client.index_documents(accumulator)
                accumulator.dequeue_actions()
                documents = []
                accumulated_size = 0

        # Upload remaining batch
        if documents:
            logger.info(
                f"Uploading remaining batch of size {len(documents)}, "
                f"current progress {len(ids)} of {len(nodes)}, "
                f"accumulated size {accumulated_size / (1024 * 1024):.2f} MB"
            )
            self._search_client.index_documents(accumulator)

        return ids

    async def async_add(
        self,
        nodes: List[BaseNode],
        index_id: Optional[str] = None,
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Add nodes to index associated with the configured search client.

        Args:
            nodes: List[BaseNode]: nodes with embeddings

        """
        from azure.search.documents import IndexDocumentsBatch

        if not self._async_search_client:
            raise ValueError("Async Search client not initialized")

        if len(nodes) > 0:
            if self._index_management == IndexManagement.CREATE_IF_NOT_EXISTS:
                if self._index_name:
                    await self._acreate_index_if_not_exists(self._index_name)

            if self._index_management == IndexManagement.VALIDATE_INDEX:
                await self._avalidate_index(self._index_name)

        accumulator = IndexDocumentsBatch()
        documents = []

        ids = []
        accumulated_size = 0
        max_size = DEFAULT_MAX_MB_SIZE  # 16MB in bytes
        max_docs = DEFAULT_MAX_BATCH_SIZE

        for node in nodes:
            logger.debug(f"Processing embedding: {node.node_id}")
            ids.append(node.node_id)

            index_document = self._create_index_document(node, index_id)
            document_size = len(json.dumps(index_document).encode("utf-8"))
            documents.append(index_document)
            accumulated_size += document_size

            accumulator.add_upload_actions(index_document)

            if len(documents) >= max_docs or accumulated_size >= max_size:
                logger.info(
                    f"Uploading batch of size {len(documents)}, "
                    f"current progress {len(ids)} of {len(nodes)}, "
                    f"accumulated size {accumulated_size / (1024 * 1024):.2f} MB"
                )
                await self._async_search_client.index_documents(accumulator)
                accumulator.dequeue_actions()
                documents = []
                accumulated_size = 0

        # Upload remaining batch
        if documents:
            logger.info(
                f"Uploading remaining batch of size {len(documents)}, "
                f"current progress {len(ids)} of {len(nodes)}, "
                f"accumulated size {accumulated_size / (1024 * 1024):.2f} MB"
            )
            await self._async_search_client.index_documents(accumulator)

        return ids

    def _create_index_document(self, node: BaseNode, index_id: str) -> Dict[str, Any]:
        """Create AI Search index document from embedding result."""
        doc: Dict[str, Any] = {}
        doc["id"] = node.node_id
        doc["index_id"] = index_id
        doc["chunk"] = node.get_content(metadata_mode=MetadataMode.NONE) or ""
        doc["embedding"] = node.get_embedding()
        doc["doc_id"] = node.ref_doc_id

        node_metadata = node_to_metadata_dict(
            node,
            remove_text=True,
            flat_metadata=self.flat_metadata,
        )

        doc["metadata"] = json.dumps(node_metadata)

        return self._index_mapping(doc, node_metadata)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete documents from the AI Search Index
        with doc_id_field_key field equal to ref_doc_id.
        """
        if not self._index_exists(self._index_name):
            return

        # Locate documents to delete
        filter = f'{self._field_mapping["doc_id"]} eq \'{ref_doc_id}\''
        batch_size = 1000

        while True:
            results = self._search_client.search(
                search_text="*",
                filter=filter,
                top=batch_size,
            )

            logger.debug(f"Searching with filter {filter}")

            docs_to_delete = [
                {"id": result[self._field_mapping["id"]]} for result in results
            ]

            if docs_to_delete:
                logger.debug(f"Deleting {len(docs_to_delete)} documents")
                self._search_client.delete_documents(docs_to_delete)
            else:
                break

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete documents from the AI Search Index
        with doc_id_field_key field equal to ref_doc_id.
        """
        if not await self._aindex_exists(self._index_name):
            return

        # Locate documents to delete
        filter = f'{self._field_mapping["doc_id"]} eq \'{ref_doc_id}\''
        batch_size = 1000

        while True:
            results = await self._async_search_client.search(
                search_text="*",
                filter=filter,
                top=batch_size,
            )

            logger.debug(f"Searching with filter {filter}")

            docs_to_delete = [
                {"id": result[self._field_mapping["id"]]} async for result in results
            ]

            if docs_to_delete:
                logger.debug(f"Deleting {len(docs_to_delete)} documents")
                await self._async_search_client.delete_documents(docs_to_delete)
            else:
                break

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        index_id: Optional[str] = None,
        **delete_kwargs: Any,
    ) -> None:
        """
        Delete documents from the AI Search Index.
        """
        if node_ids is None and filters is None:
            raise ValueError("Either node_ids or filters must be provided")

        user_filter = self._build_filter_delete_query(node_ids, filters)
        filter = f'({self._field_mapping["index_id"]} eq \'{index_id}\')'
        if user_filter:
            filter += f' and ({user_filter})'

        batch_size = 1000

        while True:
            results = self._search_client.search(
                search_text="*",
                filter=filter,
                top=batch_size,
            )

            logger.debug(f"Searching with filter {filter}")

            docs_to_delete = [
                {"id": result[self._field_mapping["id"]]} for result in results
            ]

            if docs_to_delete:
                logger.debug(f"Deleting {len(docs_to_delete)} documents")
                self._search_client.delete_documents(docs_to_delete)
            else:
                break

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        index_id: Optional[str] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """
        Delete documents from the AI Search Index.
        """
        if node_ids is None and filters is None:
            raise ValueError("Either node_ids or filters must be provided")

        user_filter = self._build_filter_delete_query(node_ids, filters)
        filter = f'({self._field_mapping["index_id"]} eq \'{index_id}\')'
        if user_filter:
            filter += f' and ({user_filter})'

        batch_size = 1000

        while True:
            results = await self._async_search_client.search(
                search_text="*",
                filter=filter,
                top=batch_size,
            )

            logger.debug(f"Searching with filter {filter}")

            docs_to_delete = [
                {"id": result[self._field_mapping["id"]]} async for result in results
            ]

            if docs_to_delete:
                logger.debug(f"Deleting {len(docs_to_delete)} documents")
                await self._async_search_client.delete_documents(docs_to_delete)
            else:
                break

    def _build_filter_delete_query(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> str:
        """Build the OData filter query for the deletion process."""
        if node_ids:
            return " or ".join(
                [
                    f'{self._field_mapping["id"]} eq \'{node_id}\''
                    for node_id in node_ids
                ]
            )

        if filters and filters.filters:
            # Find the filter with key doc_ids
            doc_ids_filter = next(
                (f for f in filters.filters if f.key == "doc_id"), None
            )
            if doc_ids_filter and doc_ids_filter.operator == FilterOperator.IN:
                # use search.in to filter on multiple values
                doc_ids_str = ",".join(doc_ids_filter.value)
                return (
                    f"search.in({self._field_mapping['doc_id']}, '{doc_ids_str}', ',')"
                )

            return self._create_odata_filter(filters)

        raise ValueError("Invalid filter configuration")

    def _create_odata_filter(self, metadata_filters: MetadataFilters) -> str:
        """Generate an OData filter string using supplied metadata filters."""
        odata_filter: List[str] = []

        for subfilter in metadata_filters.filters:
            if isinstance(subfilter, MetadataFilters):
                nested_filter = self._create_odata_filter(subfilter)
                odata_filter.append(f"({nested_filter})")
                continue

            # Join values with ' or ' to create an OR condition inside the any function
            metadata_mapping = self._metadata_to_index_field_map.get(subfilter.key)

            if not metadata_mapping:
                raise ValueError(
                    f"Metadata field '{subfilter.key}' is missing a mapping to an index field, "
                    "provide entry in 'filterable_metadata_field_keys' for this "
                    "vector store"
                )
            index_field = metadata_mapping[0]

            if subfilter.operator == FilterOperator.IN:
                value_str = " or ".join(
                    [
                        f"t eq '{value}'" if isinstance(value, str) else f"t eq {value}"
                        for value in subfilter.value
                    ]
                )
                odata_filter.append(f"{index_field}/any(t: {value_str})")

            elif subfilter.operator == FilterOperator.EQ:
                if isinstance(subfilter.value, str):
                    escaped_value = "".join(
                        [("''" if s == "'" else s) for s in subfilter.value]
                    )
                    odata_filter.append(f"{index_field} eq '{escaped_value}'")
                else:
                    odata_filter.append(f"{index_field} eq {subfilter.value}")

            else:
                raise ValueError(f"Unsupported filter operator {subfilter.operator}")

        if metadata_filters.condition == FilterCondition.AND:
            odata_expr = " and ".join(odata_filter)
        elif metadata_filters.condition == FilterCondition.OR:
            odata_expr = " or ".join(odata_filter)
        else:
            raise ValueError(
                f"Unsupported filter condition {metadata_filters.condition}"
            )

        logger.info(f"Odata filter: {odata_expr}")

        return odata_expr

    def query(self, query: VectorStoreQuery, index_id: Optional[str] = None, **kwargs: Any) -> VectorStoreQueryResult:
        odata_filter = f'{self._field_mapping["index_id"]} eq \'{index_id}\''
        if query.filters is not None:
            odata_filter = f'({odata_filter}) and ({self._create_odata_filter(query.filters)})'

        azure_query_result_search: AzureQueryResultSearchBase = (
            AzureQueryResultSearchDefault(
                query, self._field_mapping, odata_filter, self._search_client
            )
        )
        if query.mode == VectorStoreQueryMode.SPARSE:
            azure_query_result_search = AzureQueryResultSearchSparse(
                query, self._field_mapping, odata_filter, self._search_client
            )
        elif query.mode == VectorStoreQueryMode.HYBRID:
            azure_query_result_search = AzureQueryResultSearchHybrid(
                query, self._field_mapping, odata_filter, self._search_client
            )
        elif query.mode == VectorStoreQueryMode.SEMANTIC_HYBRID:
            azure_query_result_search = AzureQueryResultSearchSemanticHybrid(
                query, self._field_mapping, odata_filter, self._search_client
            )
        return azure_query_result_search.search()

    async def aquery(
        self, query: VectorStoreQuery, index_id: Optional[str] = None, **kwargs: Any
    ) -> VectorStoreQueryResult:
        odata_filter = f'{self._field_mapping["index_id"]} eq {index_id}'

        # NOTE: users can provide odata_filters directly to the query
        odata_filters = kwargs.get("odata_filters")
        if odata_filters is not None:
            odata_filter = f'({odata_filter}) and ({odata_filters})'
        else:
            if query.filters is not None:
                odata_filter = f'({odata_filter}) and ({self._create_odata_filter(query.filters)})'

        azure_query_result_search: AzureQueryResultSearchBase = (
            AzureQueryResultSearchDefault(
                query, self._field_mapping, odata_filter, self._async_search_client
            )
        )
        if query.mode == VectorStoreQueryMode.SPARSE:
            azure_query_result_search = AzureQueryResultSearchSparse(
                query, self._field_mapping, odata_filter, self._async_search_client
            )
        elif query.mode == VectorStoreQueryMode.HYBRID:
            azure_query_result_search = AzureQueryResultSearchHybrid(
                query, self._field_mapping, odata_filter, self._async_search_client
            )
        elif query.mode == VectorStoreQueryMode.SEMANTIC_HYBRID:
            azure_query_result_search = AzureQueryResultSearchSemanticHybrid(
                query, self._field_mapping, odata_filter, self._async_search_client
            )
        return await azure_query_result_search.asearch()

    def _build_filter_str(
        self,
        field_mapping: Dict[str, str],
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> Optional[str]:
        """Build OData filter string from node IDs and metadata filters.

        Args:
            field_mapping (Dict[str, str]): Field mapping dictionary
            node_ids (Optional[List[str]]): List of node IDs to filter by
            filters (Optional[MetadataFilters]): Metadata filters to apply

        Returns:
            Optional[str]: OData filter string or None if no filters
        """
        filter_str = None
        if node_ids is not None:
            filter_str = " or ".join(
                [f"{field_mapping['id']} eq '{node_id}'" for node_id in node_ids]
            )

        if filters is not None:
            metadata_filter = self._create_odata_filter(filters)
            if filter_str is not None:
                filter_str = f"({filter_str}) or ({metadata_filter})"
            else:
                filter_str = metadata_filter

        return filter_str

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        index_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[BaseNode]:
        """Get nodes from the Azure AI Search index.

        Args:
            node_ids (Optional[List[str]]): List of node IDs to retrieve.
            filters (Optional[MetadataFilters]): Metadata filters to apply.
            limit (Optional[int]): Maximum number of nodes to retrieve.

        Returns:
            List[BaseNode]: List of nodes retrieved from the index.
        """
        if not self._search_client:
            raise ValueError("Search client not initialized")

        user_filter_str = self._build_filter_str(self._field_mapping, node_ids, filters)
        if user_filter_str:
            filter_str = f'({self._field_mapping["index_id"]} eq \'{index_id}\') and ({user_filter_str})'
        else:
            filter_str = f'{self._field_mapping["index_id"]} eq \'{index_id}\''
        nodes = []
        batch_size = 1000  # Azure Search batch size limit

        while True:
            try:
                search_request = create_search_request(
                    self._field_mapping, filter_str, batch_size, len(nodes)
                )
                results = self._search_client.search(**search_request)
            except Exception as e:
                handle_search_error(e)
                break

            batch_nodes = [
                create_node_from_result(result, self._field_mapping)
                for result in results
            ]

            nodes, continue_fetching = process_batch_results(
                batch_nodes, nodes, batch_size, limit
            )
            if not continue_fetching:
                break

        return nodes

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        index_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[BaseNode]:
        """Get nodes asynchronously from the Azure AI Search index.

        Args:
            node_ids (Optional[List[str]]): List of node IDs to retrieve.
            filters (Optional[MetadataFilters]): Metadata filters to apply.
            limit (Optional[int]): Maximum number of nodes to retrieve.

        Returns:
            List[BaseNode]: List of nodes retrieved from the index.
        """
        if not self._async_search_client:
            raise ValueError("Async Search client not initialized")

        user_filter_str = self._build_filter_str(self._field_mapping, node_ids, filters)
        filter_str = f'({self._field_mapping["index_id"]} eq \'{index_id}\') and ({user_filter_str})'
        nodes = []
        batch_size = 1000  # Azure Search batch size limit

        while True:
            try:
                search_request = create_search_request(
                    self._field_mapping, filter_str, batch_size, len(nodes)
                )
                results = await self._async_search_client.search(**search_request)
            except Exception as e:
                handle_search_error(e)
                break

            batch_nodes = []
            async for result in results:
                batch_nodes.append(create_node_from_result(result, self._field_mapping))

            nodes, continue_fetching = process_batch_results(
                batch_nodes, nodes, batch_size, limit
            )
            if not continue_fetching:
                break

        return nodes


class AzureQueryResultSearchBase:
    def __init__(
        self,
        query: VectorStoreQuery,
        field_mapping: Dict[str, str],
        odata_filter: Optional[str],
        search_client: Any,
    ) -> None:
        self._query = query
        self._field_mapping = field_mapping
        self._odata_filter = odata_filter
        self._search_client = search_client

    @property
    def _select_fields(self) -> List[str]:
        return [
            self._field_mapping["id"],
            self._field_mapping["chunk"],
            self._field_mapping["metadata"],
            self._field_mapping["doc_id"],
        ]

    def _create_search_query(self) -> str:
        return "*"

    def _create_query_vector(self) -> Optional[List[Any]]:
        return None

    def _create_query_result(
        self, search_query: str, vectors: Optional[List[Any]]
    ) -> VectorStoreQueryResult:
        results = self._search_client.search(
            search_text=search_query,
            vector_queries=vectors,
            top=self._query.similarity_top_k,
            select=self._select_fields,
            filter=self._odata_filter,
        )

        id_result = []
        node_result = []
        score_result = []
        for result in results:
            node_id = result[self._field_mapping["id"]]
            metadata_str = result[self._field_mapping["metadata"]]
            metadata = json.loads(metadata_str) if metadata_str else {}
            score = result["@search.score"]
            chunk = result[self._field_mapping["chunk"]]

            try:
                node = metadata_dict_to_node(metadata)
                node.set_content(chunk)
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                metadata, node_info, relationships = legacy_metadata_dict_to_node(
                    metadata
                )

                node = TextNode(
                    text=chunk,
                    id_=node_id,
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships=relationships,
                )

            logger.debug(f"Retrieved node id {node_id} with node data of {node}")

            id_result.append(node_id)
            node_result.append(node)
            score_result.append(score)

        logger.debug(
            f"Search query '{search_query}' returned {len(id_result)} results."
        )

        return VectorStoreQueryResult(
            nodes=node_result, similarities=score_result, ids=id_result
        )

    async def _acreate_query_result(
        self, search_query: str, vectors: Optional[List[Any]]
    ) -> VectorStoreQueryResult:
        results = await self._search_client.search(
            search_text=search_query,
            vector_queries=vectors,
            top=self._query.similarity_top_k,
            select=self._select_fields,
            filter=self._odata_filter,
        )

        id_result = []
        node_result = []
        score_result = []

        async for result in results:
            node_id = result[self._field_mapping["id"]]
            metadata_str = result[self._field_mapping["metadata"]]
            metadata = json.loads(metadata_str) if metadata_str else {}
            score = result["@search.score"]
            chunk = result[self._field_mapping["chunk"]]

            try:
                node = metadata_dict_to_node(metadata)
                node.set_content(chunk)
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                metadata, node_info, relationships = legacy_metadata_dict_to_node(
                    metadata
                )

                node = TextNode(
                    text=chunk,
                    id_=node_id,
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships=relationships,
                )

            logger.debug(f"Retrieved node id {node_id} with node data of {node}")

            id_result.append(node_id)
            node_result.append(node)
            score_result.append(score)

        logger.debug(
            f"Search query '{search_query}' returned {len(id_result)} results."
        )

        return VectorStoreQueryResult(
            nodes=node_result, similarities=score_result, ids=id_result
        )

    def search(self) -> VectorStoreQueryResult:
        search_query = self._create_search_query()
        vectors = self._create_query_vector()
        return self._create_query_result(search_query, vectors)

    async def asearch(self) -> VectorStoreQueryResult:
        search_query = self._create_search_query()
        vectors = self._create_query_vector()
        return await self._acreate_query_result(search_query, vectors)


class AzureQueryResultSearchDefault(AzureQueryResultSearchBase):
    def _create_query_vector(self) -> Optional[List[Any]]:
        """Query vector store."""
        from azure.search.documents.models import VectorizedQuery

        if not self._query.query_embedding:
            raise ValueError("Query missing embedding")

        vectorized_query = VectorizedQuery(
            vector=self._query.query_embedding,
            k_nearest_neighbors=self._query.similarity_top_k,
            fields=self._field_mapping["embedding"],
        )
        vector_queries = [vectorized_query]
        logger.info("Vector search with supplied embedding")
        return vector_queries


class AzureQueryResultSearchSparse(AzureQueryResultSearchBase):
    def _create_search_query(self) -> str:
        if self._query.query_str is None:
            raise ValueError("Query missing query string")

        search_query = self._query.query_str

        logger.info(f"Hybrid search with search text: {search_query}")
        return search_query


class AzureQueryResultSearchHybrid(
    AzureQueryResultSearchDefault, AzureQueryResultSearchSparse
):
    def _create_query_vector(self) -> Optional[List[Any]]:
        return AzureQueryResultSearchDefault._create_query_vector(self)

    def _create_search_query(self) -> str:
        return AzureQueryResultSearchSparse._create_search_query(self)


class AzureQueryResultSearchSemanticHybrid(AzureQueryResultSearchHybrid):
    def _create_query_vector(self) -> Optional[List[Any]]:
        """Query vector store."""
        from azure.search.documents.models import VectorizedQuery

        if not self._query.query_embedding:
            raise ValueError("Query missing embedding")
        # k is set to 50 to align with the number of accept document in azure semantic reranking model.
        # https://learn.microsoft.com/azure/search/semantic-search-overview
        vectorized_query = VectorizedQuery(
            vector=self._query.query_embedding,
            k_nearest_neighbors=50,
            fields=self._field_mapping["embedding"],
        )
        vector_queries = [vectorized_query]
        logger.info("Vector search with supplied embedding")
        return vector_queries

    def _create_query_result(
        self, search_query: str, vector_queries: Optional[List[Any]]
    ) -> VectorStoreQueryResult:
        results = self._search_client.search(
            search_text=search_query,
            vector_queries=vector_queries,
            top=self._query.similarity_top_k,
            select=self._select_fields,
            filter=self._odata_filter,
            query_type="semantic",
            semantic_configuration_name="mySemanticConfig",
        )

        id_result = []
        node_result = []
        score_result = []
        for result in results:
            node_id = result[self._field_mapping["id"]]
            metadata_str = result[self._field_mapping["metadata"]]
            metadata = json.loads(metadata_str) if metadata_str else {}
            # use reranker_score instead of score
            score = result["@search.reranker_score"]
            chunk = result[self._field_mapping["chunk"]]

            try:
                node = metadata_dict_to_node(metadata)
                node.set_content(chunk)
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                metadata, node_info, relationships = legacy_metadata_dict_to_node(
                    metadata
                )

                node = TextNode(
                    text=chunk,
                    id_=node_id,
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships=relationships,
                )

            logger.debug(f"Retrieved node id {node_id} with node data of {node}")

            id_result.append(node_id)
            node_result.append(node)
            score_result.append(score)

        logger.debug(
            f"Search query '{search_query}' returned {len(id_result)} results."
        )

        return VectorStoreQueryResult(
            nodes=node_result, similarities=score_result, ids=id_result
        )


CognitiveSearchVectorStore = AzureAISearchVectorStore
