from llama_index.storage.docstore.cosmosdb_docstore import CosmosDBDocStore
from llama_index.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.storage.docstore.mongo_docstore import MongoDocumentStore
from llama_index.storage.docstore.redis_docstore import RedisDocumentStore

# alias for backwards compatibility
from llama_index.storage.docstore.simple_docstore import (
    DocumentStore,
    SimpleDocumentStore,
)
from llama_index.storage.docstore.types import BaseDocumentStore

__all__ = [
    "BaseDocumentStore",
    "DocumentStore",
    "SimpleDocumentStore",
    "MongoDocumentStore",
    "CosmosDBDocStore",
    "KVDocumentStore",
    "RedisDocumentStore",
]
