# LlamaIndex Chat Store Integration: MongoDB Chat Store

## Installation

```bash
pip install llama-index-storage-chat-store-mongodb
```

## Usage

Using `MongoChatStore` from `llama_index.storage.chat_store.mongo`
you can store chat history in MongoDB.

```python
from llama_index.storage.chat_store.mongo import MongoChatStore

# Initialize the MongoDB chat store with URI and database name and collection name
chat_store = MongoChatStore(
    mongodb_uri="mongodb://localhost:27017/",
    db_name="llama_index",
    collection_name="chat_sessions",
)
```

You can also initialize the chat store with a `MongoClient` or `AsyncIOMotorClient` and a database name and collection name.

```python
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient

client = MongoClient("mongodb://localhost:27017/")
async_client = AsyncIOMotorClient("mongodb://localhost:27017/")

chat_store = MongoChatStore(
    client=client,
    amongo_client=async_client,
    db_name="llama_index",
    collection_name="chat_sessions",
)
```

You can also initialize the chat store with a `Collection` or `AsyncIOMotorCollection`.

```python
from pymongo import Collection
from motor.motor_asyncio import AsyncIOMotorCollection

client = MongoClient("mongodb://localhost:27017/")
async_client = AsyncIOMotorClient("mongodb://localhost:27017/")

collection = client["llama_index"]["chat_sessions"]
async_collection = async_client["llama_index"]["chat_sessions"]

chat_store = MongoChatStore(
    collection=collection, async_collection=async_collection
)
```

## Usage with LlamaIndex

```python
from llama_index.core.chat_engine.types import ChatMessage

chat_memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    chat_store_key="user1",
)
```
