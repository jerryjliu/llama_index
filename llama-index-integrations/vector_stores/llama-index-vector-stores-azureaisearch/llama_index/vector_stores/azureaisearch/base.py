"""Azure AI Search vector store."""

import enum
import json
import logging
from enum import auto
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from azure.search.documents import SearchClient
from azure.search.documents.aio import SearchClient as AsyncSearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.aio import (
    SearchIndexClient as AsyncSearchIndexClient,
)
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchField,
    SemanticSearch,
    SemanticConfiguration,
    VectorSearch,
    VectorSearchProfile,
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
    create_search_request,
    handle_search_error,
    process_batch_results,
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
    SEARCHABLE = auto()
    FILTERABLE = auto()
    SORTABLE = auto()
    KEYWORD = auto()


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
        from azure.search.documents.indexes.models import SearchField, SearchFieldDataType
        # SearchableField returns SearchField, as does SimpleField


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

        metadata_string_fields = [
            SimpleField(name="author", type="Edm.String", filterable=True, sortable=False, hidden=False),
            SimpleField(name="theme", type="Edm.String", filterable=True, sortable=False, hidden=False),
            SimpleField(name="director", type="Edm.String", filterable=True, sortable=False, hidden=False),
        ]

        # Creating an Azure AI Search Vector Store
        vector_store = AzureAISearchVectorStore(
            search_or_index_client=index_client,
            filterable_metadata_field_keys=metadata_string_fields,
            hidden_field_keys=["embedding"],
            index_name=index_name,
            index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
            id_field="id" | SearchableField(name="id", type="Edm.String", key=True, filterable=True, sortable=True, analyzer_name="keyword", hidden=False),
            chunk_field_key="chunk" | SearchableField(name="chunk", type="Edm.String", analyzer_name="en.lucene", hidden=False),
            embedding_field_key="embedding" | SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="mySearchProfile", hidden=False),
            embedding_dimensionality=1536,
            metadata_string_field="metadata" | SimpleField(name="metadata", type="Edm.String", hidden=False),
            doc_id_field="doc_id" | SimpleField(name="doc_id", type="Edm.String", filterable=True, hidden=False),
            semantic_search_config="mySemanticConfig" | SemanticSearch(configurations=[SemanticConfiguration(name="mySemanticConfig")]),
            vector_search_config="mySearchProfile" | VectorSearch(profiles=[VectorSearchProfile(name="mySearchProfile")]),
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
    _field_names: Dict[str, str] = PrivateAttr()
    _field_mapping: Dict[str, SearchField] = PrivateAttr()
    _index_management: IndexManagement = PrivateAttr()
    _index_mapping: Callable[[Dict[str, str], Dict[str, Any]], Dict[str, str]] = (
        PrivateAttr()
    )
    _metadata_to_index_field_map: Dict[str, Tuple[str, MetadataIndexFieldType]] = (
        PrivateAttr()
    )
    _vector_profile_name: str = PrivateAttr()
    _semantic_config_name: str = PrivateAttr()
    _compression_type: str = PrivateAttr()
    _semantic_search_config: Optional[SemanticSearch | str | None] = PrivateAttr()
    _vector_search_config: Optional[VectorSearch | str | None] = PrivateAttr()
    _vector_search_algorithm: Optional[Literal["hnsw", "exhaustiveKnn"] | None] = (
        PrivateAttr()
    )
    _user_agent: str = PrivateAttr()

    def _normalise_metadata_to_index_fields(
        self,
        filterable_metadata_field_keys: Optional[
            Union[
                List[str],
                Dict[str, str],
                Dict[str, Tuple[str, MetadataIndexFieldType]],
                Dict[str, SearchField],
                List[SearchField],
            ]
        ] = None,
    ) -> Dict[str, Union[Tuple[str, MetadataIndexFieldType], SearchField]]:
        """Normalize metadata field configuration.

        Args:
            filterable_metadata_field_keys: Configuration for metadata fields.
                Can be:
                - List[str]: Field names to use as both metadata key and index field name
                - Dict[str, str]: Mapping of metadata keys to index field names
                - Dict[str, Tuple[str, MetadataIndexFieldType]]: Mapping of metadata keys
                to (field name, field type) pairs
                - Dict[str, SearchField]: Mapping of metadata keys to field configurations
                - List[SearchField]: List of field configurations

        Returns:
            Dict mapping metadata keys to either:
            - Tuple[str, MetadataIndexFieldType] for basic field configurations
            - SearchField for direct field configurations
        """
        index_field_spec: Dict[
            str, Union[Tuple[str, MetadataIndexFieldType], SearchField]
        ] = {}

        if filterable_metadata_field_keys is None:
            return index_field_spec

        if isinstance(filterable_metadata_field_keys, List):
            for field in filterable_metadata_field_keys:
                if isinstance(field, str):
                    index_field_spec[field] = (field, MetadataIndexFieldType.STRING)
                elif hasattr(field, "name"):  # SearchField-like object
                    index_field_spec[field.name] = (
                        field  # Store the actual field object
                    )

        elif isinstance(filterable_metadata_field_keys, dict):
            for k, v in filterable_metadata_field_keys.items():
                if isinstance(v, tuple):
                    index_field_spec[k] = v
                elif hasattr(v, "name"):  # SearchField-like object
                    index_field_spec[k] = v  # Store the actual field object
                elif isinstance(v, list):
                    index_field_spec[k] = (k, MetadataIndexFieldType.COLLECTION)
                elif isinstance(v, bool):
                    index_field_spec[k] = (k, MetadataIndexFieldType.BOOLEAN)
                elif isinstance(v, int):
                    index_field_spec[k] = (k, MetadataIndexFieldType.INT32)
                elif isinstance(v, float):
                    index_field_spec[k] = (k, MetadataIndexFieldType.DOUBLE)
                elif isinstance(v, str):
                    index_field_spec[k] = (v, MetadataIndexFieldType.STRING)
                else:
                    logger.warning(
                        f"Unexpected type {type(v)} for metadata field {k}, "
                        "defaulting to STRING type"
                    )
                    index_field_spec[k] = (str(v), MetadataIndexFieldType.STRING)

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

    def _create_metadata_index_fields(self) -> List[SearchField]:
        """Create a list of index fields for storing metadata values."""
        from azure.search.documents.indexes.models import SimpleField, SearchableField

        index_fields = []

        for k, v in self._metadata_to_index_field_map.items():
            # Skip if the field is already mapped
            if k in self._field_mapping:
                continue

            if isinstance(v, tuple):
                field_name, field_type = v
            elif isinstance(v, SearchField):
                index_fields.append(v)
                continue
            else:
                raise ValueError(f"Invalid field configuration: {v}")

            # Skip if the field is already mapped
            if field_name in self._field_mapping.values():
                continue

            if field_type == MetadataIndexFieldType.SEARCHABLE:
                field = SearchableField(
                    name=field_name,
                    type="Edm.String",
                    analyzer_name=self._language_analyzer,
                    searchable=True,
                    filterable=True,
                    hidden=field_name in self._hidden_field_keys,
                )
            elif field_type == MetadataIndexFieldType.KEYWORD:
                field = SearchableField(
                    name=field_name,
                    type="Edm.String",
                    analyzer_name="keyword",
                    filterable=True,
                    hidden=field_name in self._hidden_field_keys,
                )
            elif field_type == MetadataIndexFieldType.SORTABLE:
                field = SimpleField(
                    name=field_name,
                    type="Edm.String",
                    filterable=True,
                    sortable=True,
                    hidden=field_name in self._hidden_field_keys,
                )
            elif field_type == MetadataIndexFieldType.FILTERABLE:
                field = SimpleField(
                    name=field_name,
                    type="Edm.String",
                    filterable=True,
                    hidden=field_name in self._hidden_field_keys,
                )
            else:
                # Handle existing basic types
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

    def _get_default_field(self, field_key: str, field_name: str) -> SearchField:
        """
        Create default field configuration based on field key and name.

        Args:
            field_key (str): Key identifying the field type (id, chunk, embedding, etc.)
            field_name (str): Name to use for the field

        Returns:
            SearchField: Configured field object
        """
        from azure.search.documents.indexes.models import (
            SearchFieldDataType,
            SearchableField,
        )

        if field_key == "id":
            return SearchableField(
                name=field_name,
                type="Edm.String",
                key=True,
                filterable=True,
                sortable=True,
                analyzer_name="keyword",
                hidden=field_name in self._hidden_field_keys,
            )
        elif field_key == "chunk":
            return SearchableField(
                name=field_name,
                type="Edm.String",
                analyzer_name=self._language_analyzer,
                hidden=field_name in self._hidden_field_keys,
            )
        elif field_key == "embedding":
            return SearchField(
                name=field_name,
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self._embedding_dimensionality,
                vector_search_profile_name=self._vector_profile_name,
                hidden=field_name in self._hidden_field_keys,
            )
        elif field_key == "metadata":
            return SimpleField(
                name=field_name,
                type="Edm.String",
                hidden=field_name in self._hidden_field_keys,
            )
        elif field_key == "doc_id":
            return SimpleField(
                name=field_name,
                type="Edm.String",
                filterable=True,
                hidden=field_name in self._hidden_field_keys,
            )
        else:
            # Default to a simple filterable field
            return SimpleField(
                name=field_name,
                type="Edm.String",
                filterable=True,
                hidden=field_name in self._hidden_field_keys,
            )

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

    def _validate_index_configuration(self, index: Any) -> None:
        """Validate the index configuration.

        Args:
            index (SearchIndex): The index to validate

        Raises:
            ValueError: If the index configuration is invalid
        """
        required_fields = {
            "id": ("Edm.String", self._field_mapping["id"]),
            "chunk": ("Edm.String", self._field_mapping["chunk"]),
            "embedding": ("Collection(Edm.Single)", self._field_mapping["embedding"]),
            "doc_id": ("Edm.String", self._field_mapping["doc_id"]),
            "metadata": ("Edm.String", self._field_mapping["metadata"]),
        }

        # Check each required field
        for expected_type, field_config in required_fields.values():
            if isinstance(field_config, str):
                # If field_config is a string, just check field existence and type
                field = next((f for f in index.fields if f.name == field_config), None)
                if not field:
                    raise ValueError(
                        f"Required field '{field_config}' is missing from index configuration"
                    )
                if field.type != expected_type:
                    raise ValueError(
                        f"Field '{field_config}' has incorrect type. Expected {expected_type}, got {field.type}"
                    )
            else:
                # If field_config is a SearchField, compare the entire field configuration
                field = next(
                    (f for f in index.fields if f.name == field_config.name), None
                )
                if not field:
                    raise ValueError(
                        f"Required field '{field_config.name}' is missing from index configuration"
                    )
                # Compare relevant field attributes
                if field.type != field_config.type:
                    raise ValueError(
                        f"Field '{field_config.name}' has incorrect type. "
                        f"Expected {field_config.type}, got {field.type}"
                    )
                if getattr(field, "searchable", None) != getattr(
                    field_config, "searchable", None
                ):
                    raise ValueError(
                        f"Field '{field_config.name}' has incorrect searchable configuration"
                    )
                if getattr(field, "filterable", None) != getattr(
                    field_config, "filterable", None
                ):
                    raise ValueError(
                        f"Field '{field_config.name}' has incorrect filterable configuration"
                    )
                if getattr(field, "sortable", None) != getattr(
                    field_config, "sortable", None
                ):
                    raise ValueError(
                        f"Field '{field_config.name}' has incorrect sortable configuration"
                    )
                if getattr(field, "hidden", None) != getattr(
                    field_config, "hidden", None
                ):
                    raise ValueError(
                        f"Field '{field_config.name}' has incorrect hidden configuration"
                    )

        # Validate vector search configuration
        if not index.vector_search:
            raise ValueError("Vector search configuration is missing from index")

        vector_config: list[VectorSearchProfile] = index.vector_search.profiles
        if not any(
            profile.name == self._vector_profile_name for profile in vector_config
        ):
            raise ValueError(
                f"Missing vector search profile: {self._vector_profile_name}"
            )

        if not any(
            profile.algorithm_configuration_name == algo.name
            for profile in vector_config
            for algo in index.vector_search.algorithms
        ):
            raise ValueError(
                f"Vector search profile references non-existent algorithm configuration"
            )

        # Validate semantic configuration if specified
        if self._semantic_config_name and not index.semantic_search:
            raise ValueError(
                f"Semantic configuration '{self._semantic_config_name}' specified but no semantic settings found in index"
            )

    def _create_index(self, index_name: Optional[str]) -> None:
        """Creates an index based on configured fields and search profiles."""
        from azure.search.documents.indexes.models import (
            ExhaustiveKnnAlgorithmConfiguration,
            ExhaustiveKnnParameters,
            HnswAlgorithmConfiguration,
            HnswParameters,
            SearchField,
            SearchIndex,
            SemanticConfiguration,
            SemanticField,
            SemanticPrioritizedFields,
            SemanticSearch,
            VectorSearch,
            VectorSearchAlgorithmKind,
            VectorSearchAlgorithmMetric,
            VectorSearchProfile,
        )

        logger.info(f"Configuring {index_name} fields for Azure AI Search")

        # Use provided field configurations or create defaults
        fields = []
        # Map the internal field keys to their configured names
        field_keys = {
            "id": self._field_names["id"],
            "chunk": self._field_names["chunk"],
            "embedding": self._field_names["embedding"],
            "metadata": self._field_names["metadata"],
            "doc_id": self._field_names["doc_id"],
        }

        for internal_key, field_name in field_keys.items():
            field_config = self._field_mapping.get(internal_key)
            if isinstance(field_config, SearchField):
                fields.append(field_config)
            else:
                fields.append(self._get_default_field(internal_key, field_name))

        logger.info(f"Configuring {index_name} metadata fields")
        metadata_index_fields = self._create_metadata_index_fields()
        fields.extend(metadata_index_fields)

        logger.info(f"Configuring {index_name} vector search")

        # Configure vector search
        if self._vector_search_config is not None:
            if isinstance(self._vector_search_config, str):
                # Create default vector search with specified profile name
                compressions = self._get_compressions()
                algorithms = [
                    HnswAlgorithmConfiguration(
                        name="myHnsw",
                        kind=VectorSearchAlgorithmKind.HNSW,
                        parameters=HnswParameters(
                            m=4,
                            ef_construction=400,
                            ef_search=500,
                            metric=VectorSearchAlgorithmMetric.COSINE,
                        ),
                    )
                ]
                profiles = [
                    VectorSearchProfile(
                        name=self._vector_search_config,
                        algorithm_configuration_name=algorithms[0].name,
                        compression_name=(
                            compressions[0].compression_name if compressions else None
                        ),
                    )
                ]
                vector_search = VectorSearch(
                    algorithms=algorithms,
                    compressions=compressions,
                    profiles=profiles,
                )
            else:
                # Use provided VectorSearch configuration directly
                vector_search = self._vector_search_config
        else:
            # Create default vector search configuration
            compressions = self._get_compressions()
            algorithms = [
                HnswAlgorithmConfiguration(
                    name="myHnsw",
                    kind=VectorSearchAlgorithmKind.HNSW,
                    parameters=HnswParameters(
                        m=4,
                        ef_construction=400,
                        ef_search=500,
                        metric=VectorSearchAlgorithmMetric.COSINE,
                    ),
                )
            ]
            profiles = [
                VectorSearchProfile(
                    name=self._vector_profile_name,
                    algorithm_configuration_name=algorithms[0].name,
                    compression_name=(
                        compressions[0].compression_name if compressions else None
                    ),
                )
            ]
            vector_search = VectorSearch(
                algorithms=algorithms,
                compressions=compressions,
                profiles=profiles,
            )

        logger.info(f"Configuring {index_name} semantic search")
        # Configure semantic search
        if self._semantic_search_config is not None:
            if isinstance(self._semantic_search_config, str):
                # Create default semantic search with specified configuration name
                semantic_config = SemanticConfiguration(
                    name=self._semantic_search_config,
                    prioritized_fields=SemanticPrioritizedFields(
                        content_fields=[
                            SemanticField(field_name=self._field_names["chunk"])
                        ]
                    ),
                )
                semantic_search = SemanticSearch(configurations=[semantic_config])
            else:
                # Use provided SemanticSearch configuration directly
                semantic_search = self._semantic_search_config
        else:
            # Create default semantic search configuration
            semantic_config = SemanticConfiguration(
                name=self._semantic_config_name,
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[
                        SemanticField(field_name=self._field_names["chunk"])
                    ],
                ),
            )
            semantic_search = SemanticSearch(configurations=[semantic_config])

        # Create and validate the index
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
        )

        # Validate required fields and configurations
        self._validate_index_configuration(index)

        logger.debug(f"Creating {index_name} search index")
        self._index_client.create_index(index)

    async def _acreate_index(self, index_name: Optional[str]) -> None:
        """Creates an index based on configured fields and search profiles."""
        from azure.search.documents.indexes.models import (
            ExhaustiveKnnAlgorithmConfiguration,
            ExhaustiveKnnParameters,
            HnswAlgorithmConfiguration,
            HnswParameters,
            SearchField,
            SearchIndex,
            SemanticConfiguration,
            SemanticField,
            SemanticPrioritizedFields,
            SemanticSearch,
            VectorSearch,
            VectorSearchAlgorithmKind,
            VectorSearchAlgorithmMetric,
            VectorSearchProfile,
        )

        logger.info(f"Configuring {index_name} fields for Azure AI Search")

        # Use provided field configurations or create defaults
        fields = []
        # Map the internal field keys to their configured names
        field_keys = {
            "id": self._field_names["id"],
            "chunk": self._field_names["chunk"],
            "embedding": self._field_names["embedding"],
            "metadata": self._field_names["metadata"],
            "doc_id": self._field_names["doc_id"],
        }

        for internal_key, field_name in field_keys.items():
            field_config = self._field_mapping.get(internal_key)
            if isinstance(field_config, SearchField):
                fields.append(field_config)
            else:
                fields.append(self._get_default_field(internal_key, field_name))

        logger.info(f"Configuring {index_name} metadata fields")
        metadata_index_fields = self._create_metadata_index_fields()
        fields.extend(metadata_index_fields)

        logger.info(f"Configuring {index_name} vector search")

        # Configure vector search
        if self._vector_search_config is not None:
            if isinstance(self._vector_search_config, str):
                # Create default vector search with specified profile name
                compressions = self._get_compressions()
                algorithms = [
                    HnswAlgorithmConfiguration(
                        name="myHnsw",
                        kind=VectorSearchAlgorithmKind.HNSW,
                        parameters=HnswParameters(
                            m=4,
                            ef_construction=400,
                            ef_search=500,
                            metric=VectorSearchAlgorithmMetric.COSINE,
                        ),
                    )
                ]
                profiles = [
                    VectorSearchProfile(
                        name=self._vector_search_config,
                        algorithm_configuration_name=algorithms[0].name,
                        compression_name=(
                            compressions[0].compression_name if compressions else None
                        ),
                    )
                ]
                vector_search = VectorSearch(
                    algorithms=algorithms,
                    compressions=compressions,
                    profiles=profiles,
                )
            else:
                # Use provided VectorSearch configuration directly
                vector_search = self._vector_search_config
        else:
            # Create default vector search configuration
            compressions = self._get_compressions()
            algorithms = [
                HnswAlgorithmConfiguration(
                    name="myHnsw",
                    kind=VectorSearchAlgorithmKind.HNSW,
                    parameters=HnswParameters(
                        m=4,
                        ef_construction=400,
                        ef_search=500,
                        metric=VectorSearchAlgorithmMetric.COSINE,
                    ),
                )
            ]
            profiles = [
                VectorSearchProfile(
                    name=self._vector_profile_name,
                    algorithm_configuration_name=algorithms[0].name,
                    compression_name=(
                        compressions[0].compression_name if compressions else None
                    ),
                )
            ]
            vector_search = VectorSearch(
                algorithms=algorithms,
                compressions=compressions,
                profiles=profiles,
            )

        logger.info(f"Configuring {index_name} semantic search")
        # Configure semantic search
        if self._semantic_search_config is not None:
            if isinstance(self._semantic_search_config, str):
                # Create default semantic search with specified configuration name
                semantic_config = SemanticConfiguration(
                    name=self._semantic_search_config,
                    prioritized_fields=SemanticPrioritizedFields(
                        content_fields=[
                            SemanticField(field_name=self._field_names["chunk"])
                        ]
                    ),
                )
                semantic_search = SemanticSearch(configurations=[semantic_config])
            else:
                # Use provided SemanticSearch configuration directly
                semantic_search = self._semantic_search_config
        else:
            # Create default semantic search configuration
            semantic_config = SemanticConfiguration(
                name=self._semantic_search_config_name,
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[
                        SemanticField(field_name=self._field_names["chunk"])
                    ],
                ),
            )
            semantic_search = SemanticSearch(configurations=[semantic_config])

        # Create and validate the index
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
        )

        # Validate required fields and configurations
        self._validate_index_configuration(index)

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
        search_or_index_client: Union[
            SearchClient, SearchIndexClient, AsyncSearchClient, AsyncSearchIndexClient
        ],
        id_field_key: Optional[str] = None,  # Kept for backwards compatibility
        id_field: Optional[
            Union[
                str,
                SearchField,
            ]
        ] = "id",
        chunk_field_key: Optional[str] = None,  # Kept for backwards compatibility
        chunk_field: Optional[
            Union[
                str,
                SearchField,
            ]
        ] = "chunk",
        embedding_field_key: Optional[str] = None,  # Kept for backwards compatibility
        embedding_field: Optional[
            Union[
                str,
                SearchField,
            ]
        ] = "embedding",
        metadata_string_field_key: Optional[
            str
        ] = None,  # Kept for backwards compatibility
        metadata_string_field: Optional[
            Union[
                str,
                SearchField,
            ]
        ] = "metadata",
        doc_id_field_key: Optional[str] = None,  # Kept for backwards compatibility
        doc_id_field: Optional[
            Union[
                str,
                SearchField,
            ]
        ] = "doc_id",
        async_search_or_index_client: Optional[
            Union[AsyncSearchClient, AsyncSearchIndexClient]
        ] = None,
        filterable_metadata_field_keys: Optional[
            Union[
                List[str],
                Dict[str, str],
                Dict[str, Tuple[str, MetadataIndexFieldType]],
                Dict[str, SearchField],
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
        vector_search_profile_name: Optional[str] = None,
        semantic_config_name: Optional[str] = None,
        semantic_search_config: Optional[
            Union[SemanticSearch, str, None]
        ] = "mySemanticConfig",
        vector_search_config: Optional[
            Union[VectorSearch, str, None]
        ] = "myHnswProfile",
        language_analyzer: str = "en.lucene",
        compression_type: str = "none",
        user_agent: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with Azure AI Search client.

        Args:
            search_or_index_client (Union[SearchClient, SearchIndexClient, AsyncSearchClient, AsyncSearchIndexClient]):
                Client for index to populate / query.
            id_field_key (str): DEPRECATED, use id_field instead
            id_field (str | SearchField): Index field storing the id
            chunk_field_key (str): DEPRECATED, use chunk_field instead
            chunk_field (str | SearchField): Index field storing the node text
            embedding_field_key (str): DEPRECATED, use embedding_field instead
            embedding_field (str | SearchField): Index field storing the embedding vector
            metadata_field_key (str): DEPRECATED, use metadata_field instead
            metadata_field (str | SearchField):
                Index field storing node metadata as a json string.
                Schema is arbitrary, to filter on metadata values they must be stored
                as separate fields in the index, use filterable_metadata_field_keys
                to specify the metadata values that should be stored in these filterable fields
            doc_id_field_key (str): DEPRECATED, use doc_id_field instead
            doc_id_field (str | SearchField): Index field storing doc_id
            hidden_field_keys (List[str]):
                List of index fields that should be hidden from the client.
                This is useful for fields that are not needed for retrieving,
                but are used for similarity search, like the embedding field.
            index_mapping (Optional[Callable[[Dict[str, str], Dict[str, Any]], Dict[str, str]]]):
                Optional function to map document fields to the AI search index fields.
                If none is specified a default mapping is provided.
            vector_search_profile_name (str | None): DEPRECATED use vector_search_profile instead
            semantic_search_config (SemanticSearch | str | None): Semantic configuration to use
            vector_search_config (VectorSearch | str | None): Vector search profile to use
            language_analyzer (str): Language analyzer to use for text fields
            compression_type (str): Type of vector compression to use
            user_agent (str | None): Optional user agent string
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
        self._embedding_dimensionality = embedding_dimensionality
        self._index_name = index_name
        self._vector_search_algorithm = vector_algorithm_type
        self._hidden_field_keys = hidden_field_keys or []
        self._index_mapping = (
            self._default_index_mapping if index_mapping is None else index_mapping
        )

        # Handle vector search profile configuration
        if vector_search_config is not None:
            self._vector_profile_name = (
                vector_search_config.name
                if isinstance(vector_search_config, VectorSearchProfile)
                else vector_search_config
            )
            self._vector_search_config = vector_search_config
        elif vector_search_profile_name is not None:
            logger.warning(
                "vector_search_profile_name is deprecated, use vector_search_profile instead"
            )
            self._vector_profile_name = vector_search_profile_name

        # Handle semantic search configuration
        self._semantic_config_name = (
            semantic_search_config.name
            if isinstance(semantic_search_config, SemanticSearch)
            else semantic_search_config
        )

        if semantic_search_config is not None:
            self._semantic_search_config = semantic_search_config

        # Set default vector profile based on algorithm type
        if vector_algorithm_type == "exhaustiveKnn":
            self._vector_profile_name = (
                self._vector_profile_name or "myExhaustiveKnnProfile"
            )
        elif vector_algorithm_type == "hnsw":
            self._vector_profile_name = self._vector_profile_name or "myHnswProfile"
        else:
            raise ValueError(
                "Only 'exhaustiveKnn' and 'hnsw' are supported for vector_algorithm_type"
            )

        self._vector_algorithm_type = vector_algorithm_type

        self._language_analyzer = language_analyzer
        self._compression_type = compression_type.lower()

        # Initialize clients
        self._index_client = None
        self._async_index_client = None
        self._search_client = None
        self._async_search_client = None

        if search_or_index_client and async_search_or_index_client is None:
            logger.warning(
                "async_search_or_index_client is None. Depending on the client type passed "
                "in, sync or async functions may not work."
            )

        # Configure clients and validate index name
        if search_or_index_client is not None:
            if isinstance(search_or_index_client, SearchIndexClient):
                self._index_client = search_or_index_client
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
            elif isinstance(search_or_index_client, SearchClient):
                self._search_client = search_or_index_client
                self._search_client._client._config.user_agent_policy.add_user_agent(
                    self._user_agent
                )
                if index_name:
                    raise ValueError(
                        "index_name cannot be supplied if search_or_index_client "
                        "is of type azure.search.documents.SearchClient"
                    )

        # Handle async client configuration
        async_search_or_index_client = (
            async_search_or_index_client or search_or_index_client
        )
        if async_search_or_index_client is not None:
            if isinstance(async_search_or_index_client, AsyncSearchIndexClient):
                self._async_index_client = async_search_or_index_client
                self._async_index_client._client._config.user_agent_policy.add_user_agent(
                    self._user_agent
                )
                if not index_name:
                    raise ValueError(
                        "index_name must be supplied if async_search_or_index_client is of "
                        "type azure.search.documents.aio.SearchIndexClient"
                    )
                self._async_search_client = self._async_index_client.get_search_client(
                    index_name=index_name
                )
                self._async_search_client._client._config.user_agent_policy.add_user_agent(
                    self._user_agent
                )
            elif isinstance(async_search_or_index_client, AsyncSearchClient):
                self._async_search_client = async_search_or_index_client
                self._async_search_client._client._config.user_agent_policy.add_user_agent(
                    self._user_agent
                )
                if index_name:
                    raise ValueError(
                        "index_name cannot be supplied if async_search_or_index_client "
                        "is of type azure.search.documents.aio.SearchClient"
                    )

        # Validate client configuration
        if not any(
            [
                self._search_client,
                self._async_search_client,
                self._index_client,
                self._async_index_client,
            ]
        ):
            raise ValueError(
                "Either search_or_index_client or async_search_or_index_client must be provided"
            )

        search_client = self._search_client or self._async_search_client

        # Validate index management requirements
        if index_management == IndexManagement.CREATE_IF_NOT_EXISTS and not (
            self._index_client or self._async_index_client
        ):
            raise ValueError(
                "index_management has value of IndexManagement.CREATE_IF_NOT_EXISTS "
                "but neither search_or_index_client nor async_search_or_index_client is of type "
                "azure.search.documents.SearchIndexClient or azure.search.documents.aio.SearchIndexClient"
            )

        self._index_management = index_management

        # Initialize field mappings
        self._field_mapping = {}
        self._field_names = {}

        # First determine the actual field names, with proper fallback to deprecated keys
        id_name = (
            id_field_key
            if id_field_key and len(id_field_key) > 0
            else id_field.name if id_field is not None else "id"
        )
        chunk_name = (
            chunk_field_key
            if chunk_field_key and len(chunk_field_key) > 0
            else chunk_field.name if chunk_field is not None else "chunk"
        )
        embedding_name = (
            embedding_field_key
            if embedding_field_key and len(embedding_field_key) > 0
            else embedding_field.name if embedding_field is not None else "embedding"
        )
        metadata_name = (
            metadata_string_field_key
            if metadata_string_field_key and len(metadata_string_field_key) > 0
            else (
                metadata_string_field.name
                if metadata_string_field is not None
                else "metadata"
            )
        )
        doc_id_name = (
            doc_id_field_key
            if doc_id_field_key and len(doc_id_field_key) > 0
            else doc_id_field.name if doc_id_field is not None else "doc_id"
        )

        self._field_names = {
            "id": id_name,
            "chunk": chunk_name,
            "embedding": embedding_name,
            "metadata": metadata_name,
            "doc_id": doc_id_name,
        }

        # Map the field variables directly
        self._field_mapping = {
            "id": (
                id_field
                if isinstance(id_field, SearchField)
                else self._get_default_field("id", self._field_names["id"])
            ),
            "chunk": (
                chunk_field
                if isinstance(chunk_field, SearchField)
                else self._get_default_field("chunk", self._field_names["chunk"])
            ),
            "embedding": (
                embedding_field
                if isinstance(embedding_field, SearchField)
                else self._get_default_field(
                    "embedding", self._field_names["embedding"]
                )
            ),
            "metadata": (
                metadata_string_field
                if isinstance(metadata_string_field, SearchField)
                else self._get_default_field("metadata", self._field_names["metadata"])
            ),
            "doc_id": (
                doc_id_field
                if isinstance(doc_id_field, SearchField)
                else self._get_default_field("doc_id", self._field_names["doc_id"])
            ),
        }

        # Now create the field mappings using the determined names
        for key, field in self._field_names.items():
            if isinstance(field, SearchField):
                self._field_mapping[key] = field
            else:
                self._field_mapping[key] = self._get_default_field(
                    key, self._field_names[key]
                )

        self._metadata_to_index_field_map = self._normalise_metadata_to_index_fields(
            filterable_metadata_field_keys
        )

        # Handle index initialization for non-async clients
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
        self, doc: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Default mapping function to map document fields to index fields."""
        # Start with the base document
        index_doc = {
            self._field_names[k]: v for k, v in doc.items() if k in self._field_names
        }

        # Add any configured metadata fields
        for meta_key, (field_name, _) in self._metadata_to_index_field_map.items():
            if meta_key in metadata:
                index_doc[field_name] = metadata[meta_key]

        return index_doc

    def add(
        self,
        nodes: List[BaseNode],
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

            index_document = self._create_index_document(node)
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

            index_document = self._create_index_document(node)
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

    def _create_index_document(self, node: BaseNode) -> Dict[str, Any]:
        """Create AI Search index document from embedding result."""
        doc: Dict[str, Any] = {}
        doc[self._field_names["id"]] = node.node_id
        doc[self._field_names["chunk"]] = (
            node.get_content(metadata_mode=MetadataMode.NONE) or ""
        )
        doc[self._field_names["embedding"]] = node.get_embedding()
        doc[self._field_names["doc_id"]] = node.ref_doc_id

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
        **delete_kwargs: Any,
    ) -> None:
        """
        Delete documents from the AI Search Index.
        """
        if node_ids is None and filters is None:
            raise ValueError("Either node_ids or filters must be provided")

        filter = self._build_filter_delete_query(node_ids, filters)

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
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """
        Delete documents from the AI Search Index.
        """
        if node_ids is None and filters is None:
            raise ValueError("Either node_ids or filters must be provided")

        filter = self._build_filter_delete_query(node_ids, filters)

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

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        odata_filter = None
        if query.filters is not None:
            odata_filter = self._create_odata_filter(query.filters)
        semantic_config = (
            self._semantic_search_config
            or self._semantic_config_name
            or "mySemanticConfig"
        )
        vector_profile = (
            self._vector_search_config.profiles[0].name
            if isinstance(self._vector_search_config, VectorSearch)
            else self._vector_profile_name or "mySearchProfile"
        )
        azure_query_result_search: AzureQueryResultSearchBase = (
            AzureQueryResultSearchDefault(
                query=query,
                field_names=self._field_names,
                metadata_to_index_field_map=self._metadata_to_index_field_map,
                odata_filter=odata_filter,
                search_client=self._search_client,
                async_search_client=self._async_search_client,
                semantic_search_config=semantic_config,
                vector_search_config=vector_profile,
            )
        )
        if query.mode == VectorStoreQueryMode.SPARSE:
            azure_query_result_search = AzureQueryResultSearchSparse(
                query=query,
                field_names=self._field_names,
                metadata_to_index_field_map=self._metadata_to_index_field_map,
                odata_filter=odata_filter,
                search_client=self._search_client,
                async_search_client=self._async_search_client,
                semantic_search_config=semantic_config,
                vector_search_config=vector_profile,
            )
        elif query.mode == VectorStoreQueryMode.HYBRID:
            azure_query_result_search = AzureQueryResultSearchHybrid(
                query=query,
                field_names=self._field_names,
                metadata_to_index_field_map=self._metadata_to_index_field_map,
                odata_filter=odata_filter,
                search_client=self._search_client,
                async_search_client=self._async_search_client,
                semantic_search_config=semantic_config,
                vector_search_config=vector_profile,
            )
        elif query.mode == VectorStoreQueryMode.SEMANTIC_HYBRID:
            azure_query_result_search = AzureQueryResultSearchSemanticHybrid(
                query=query,
                field_names=self._field_names,
                metadata_to_index_field_map=self._metadata_to_index_field_map,
                odata_filter=odata_filter,
                search_client=self._search_client,
                async_search_client=self._async_search_client,
                semantic_search_config=semantic_config,
                vector_search_config=vector_profile,
            )
        return azure_query_result_search.search()

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        odata_filter = None

        # NOTE: users can provide odata_filters directly to the query
        odata_filters = kwargs.get("odata_filters")
        if odata_filters is not None:
            odata_filter = odata_filter
        else:
            if query.filters is not None:
                odata_filter = self._create_odata_filter(query.filters)

        semantic_config = (
            self._semantic_search_config
            or self._semantic_config_name
            or "mySemanticConfig"
        )
        vector_profile = (
            self._vector_search_config.profiles[0].name
            if isinstance(self._vector_search_config, VectorSearch)
            else self._vector_profile_name or "mySearchProfile"
        )
        azure_query_result_search: AzureQueryResultSearchBase = (
            AzureQueryResultSearchDefault(
                query=query,
                field_names=self._field_names,
                metadata_to_index_field_map=self._metadata_to_index_field_map,
                odata_filter=odata_filter,
                search_client=self._search_client,
                async_search_client=self._async_search_client,
                semantic_search_config=semantic_config,
                vector_search_config=vector_profile,
            )
        )
        if query.mode == VectorStoreQueryMode.SPARSE:
            azure_query_result_search = AzureQueryResultSearchSparse(
                query=query,
                field_names=self._field_names,
                metadata_to_index_field_map=self._metadata_to_index_field_map,
                odata_filter=odata_filter,
                search_client=self._search_client,
                async_search_client=self._async_search_client,
                semantic_search_config=semantic_config,
                vector_search_config=vector_profile,
            )
        elif query.mode == VectorStoreQueryMode.HYBRID:
            azure_query_result_search = AzureQueryResultSearchHybrid(
                query=query,
                field_names=self._field_names,
                metadata_to_index_field_map=self._metadata_to_index_field_map,
                odata_filter=odata_filter,
                search_client=self._search_client,
                async_search_client=self._async_search_client,
                semantic_search_config=semantic_config,
                vector_search_config=vector_profile,
            )
        elif query.mode == VectorStoreQueryMode.SEMANTIC_HYBRID:
            azure_query_result_search = AzureQueryResultSearchSemanticHybrid(
                query=query,
                field_names=self._field_names,
                metadata_to_index_field_map=self._metadata_to_index_field_map,
                odata_filter=odata_filter,
                search_client=self._search_client,
                async_search_client=self._async_search_client,
                semantic_search_config=semantic_config,
                vector_search_config=vector_profile,
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

        filter_str = self._build_filter_str(self._field_names, node_ids, filters)
        nodes = []
        batch_size = 1000  # Azure Search batch size limit

        while True:
            try:
                search_request = create_search_request(
                    self._field_names, filter_str, batch_size, len(nodes)
                )
                results = self._search_client.search(**search_request)
            except Exception as e:
                handle_search_error(e)
                break

            batch_nodes = [
                create_node_from_result(result, self._field_names) for result in results
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

        filter_str = self._build_filter_str(self._field_names, node_ids, filters)
        nodes = []
        batch_size = 1000  # Azure Search batch size limit

        while True:
            try:
                search_request = create_search_request(
                    self._field_names, filter_str, batch_size, len(nodes)
                )
                results = await self._async_search_client.search(**search_request)
            except Exception as e:
                handle_search_error(e)
                break

            batch_nodes = []
            async for result in results:
                batch_nodes.append(create_node_from_result(result, self._field_names))

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
        field_names: Dict[str, str],
        metadata_to_index_field_map: Dict[str, Union[str, SearchField]],
        odata_filter: Optional[str],
        search_client: SearchClient,
        async_search_client: AsyncSearchClient,
        semantic_search_config: Optional[
            Union[SemanticSearch, str, None]
        ] = "mySemanticConfig",
        vector_search_config: Optional[
            Union[VectorSearch, str, None]
        ] = "myHnswProfile",
    ) -> None:
        self._query = query
        self._field_names = field_names
        self._metadata_to_index_field_map = metadata_to_index_field_map
        self._odata_filter = odata_filter
        self._search_client = search_client
        self._async_search_client = async_search_client

        # Store the full config objects or names
        self._semantic_search_config = semantic_search_config
        self._vector_search_config = vector_search_config

        # For backward compatibility, extract names if configs are objects
        self._semantic_config_name = (
            semantic_search_config.configurations[0].name
            if isinstance(semantic_search_config, SemanticSearch)
            else semantic_search_config or "mySemanticConfig"
        )
        self._vector_profile_name = (
            vector_search_config.profiles[0].name
            if isinstance(vector_search_config, VectorSearch)
            else vector_search_config or "myHnswProfile"
        )

    @property
    def _select_fields(self) -> list[str]:
        """Get the list of fields to select in queries."""
        return [
            self._field_names[field] for field in ["id", "chunk", "metadata", "doc_id"]
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
            node_id = result[self._field_names["id"]]
            chunk = result[self._field_names["chunk"]]
            score = result["@search.score"]

            # Try LlamaIndex metadata first
            metadata = {}
            if self._field_names["metadata"] in result:
                metadata_str = result[self._field_names["metadata"]]
                if metadata_str:
                    try:
                        metadata = json.loads(metadata_str)
                    except json.JSONDecodeError:
                        logger.debug(
                            "Could not parse metadata JSON, if chunk is not empty, we'll use it anyways"
                        )
                        if len(chunk) == 0:
                            raise json.JSONDecodeError(
                                "Could not parse metadata JSON, and chunk is empty"
                            )

            try:
                node = metadata_dict_to_node(metadata, text=chunk)
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                try:
                    metadata, node_info, relationships = legacy_metadata_dict_to_node(
                        metadata
                    )
                except Exception:
                    # If both metadata conversions fail, assume flat metadata structure
                    node_info = {}
                    relationships = {}

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
        results = await self._async_search_client.search(
            search_text=search_query,
            vector_queries=vectors,
            top=self._query.similarity_top_k,
            select=self._select_fields,
            filter=self._odata_filter,
        )

        id_result = []
        node_result = []
        score_result = []

        # Get set of default field names for later filtering
        default_fields = self._select_fields

        async for result in results:
            node_id = result[self._field_names["id"]]
            chunk = result[self._field_names["chunk"]]
            score = result["@search.score"]

            # Try LlamaIndex metadata first
            metadata = {}
            if self._field_names["metadata"] in result:
                metadata_str = result[self._field_names["metadata"]]
                if metadata_str:
                    try:
                        metadata = json.loads(metadata_str)
                    except json.JSONDecodeError:
                        logger.debug(
                            "Could not parse metadata JSON, if chunk is not empty, we'll use it anyways"
                        )
                        if len(chunk) == 0:
                            raise json.JSONDecodeError(
                                "Could not parse metadata JSON, and chunk is empty"
                            )

            try:
                node = metadata_dict_to_node(metadata, text=chunk)
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                try:
                    metadata, node_info, relationships = legacy_metadata_dict_to_node(
                        metadata
                    )
                except Exception:
                    # If both metadata conversions fail, assume flat metadata structure
                    node_info = {}
                    relationships = {}

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
            fields=self._field_names["embedding"],
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
            fields=self._field_names["embedding"],
        )
        vector_queries = [vectorized_query]
        logger.info("Vector search with supplied embedding")
        return vector_queries

    def _create_query_result(
        self, search_query: str, vectors: Optional[List[Any]]
    ) -> VectorStoreQueryResult:
        results = self._search_client.search(
            search_text=search_query,
            vector_queries=vectors,
            top=self._query.similarity_top_k,
            select=self._select_fields,
            filter=self._odata_filter,
            query_type="semantic",
            semantic_configuration_name=self._semantic_config_name,
        )

        id_result = []
        node_result = []
        score_result = []

        # Get set of default field names for later filtering
        default_fields = self._select_fields

        for result in results:
            node_id = result[self._field_names["id"]]
            chunk = result[self._field_names["chunk"]]
            # Use reranker_score if available (semantic search), otherwise use regular score
            score = (
                result.get("@search.reranker_score", result["@search.score"])
                if self._semantic_config_name and self._semantic_config_name != ""
                else result["@search.score"]
            )

            # Try LlamaIndex metadata first
            if self._field_names["metadata"] in result:
                metadata_str = result[self._field_names["metadata"]]
                if metadata_str:
                    try:
                        metadata = json.loads(metadata_str)
                    except json.JSONDecodeError:
                        logger.debug(
                            "Could not parse metadata JSON, if chunk is not empty, we'll use it anyways"
                        )
                        if len(chunk) == 0:
                            raise json.JSONDecodeError(
                                "Could not parse metadata JSON, and chunk is empty"
                            )

            try:
                node = metadata_dict_to_node(metadata, text=chunk)
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                try:
                    metadata, node_info, relationships = legacy_metadata_dict_to_node(
                        metadata
                    )
                except Exception:
                    # If both metadata conversions fail, assume flat metadata structure
                    node_info = {}
                    relationships = {}

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
        results = await self._async_search_client.search(
            search_text=search_query,
            vector_queries=vectors,
            top=self._query.similarity_top_k,
            select=self._select_fields,
            filter=self._odata_filter,
            query_type="semantic",
            semantic_configuration_name=self._semantic_config_name,
        )

        id_result = []
        node_result = []
        score_result = []

        async for result in results:
            node_id = result[self._field_names["id"]]
            chunk = result[self._field_names["chunk"]]
            # Use reranker_score if available (semantic search), otherwise use regular score
            score = (
                result.get("@search.reranker_score", result["@search.score"])
                if self._semantic_config_name and self._semantic_config_name != ""
                else result["@search.score"]
            )

            # Try LlamaIndex metadata first
            metadata = {}
            if self._field_names["metadata"] in result:
                metadata_str = result[self._field_names["metadata"]]
                if metadata_str:
                    try:
                        metadata = json.loads(metadata_str)
                    except json.JSONDecodeError:
                        logger.debug(
                            "Could not parse metadata JSON, if chunk is not empty, we'll use it anyways"
                        )
                        if len(chunk) == 0:
                            raise json.JSONDecodeError(
                                "Could not parse metadata JSON, and chunk is empty"
                            )

            try:
                node = metadata_dict_to_node(metadata, text=chunk)
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                try:
                    metadata, node_info, relationships = legacy_metadata_dict_to_node(
                        metadata
                    )
                except Exception:
                    # If both metadata conversions fail, assume flat metadata structure
                    node_info = {}
                    relationships = {}

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
