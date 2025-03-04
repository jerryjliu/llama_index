from typing import Any, List, Optional
from datetime import datetime

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store.base import BaseChatStore
from pymongo import MongoClient
from pymongo.collection import Collection
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection


def _message_to_dict(message: ChatMessage) -> dict:
    """Convert a ChatMessage to a dictionary for MongoDB storage."""
    return message.model_dump()


def _dict_to_message(d: dict) -> ChatMessage:
    """Convert a dictionary from MongoDB to a ChatMessage."""
    return ChatMessage.model_validate(d)


class MongoChatStore(BaseChatStore):
    """MongoDB chat store implementation."""

    mongo_uri: str = Field(
        default="mongodb://localhost:27017", description="MongoDB URI."
    )
    db_name: str = Field(default="default", description="MongoDB database name.")
    collection_name: str = Field(
        default="sessions", description="MongoDB collection name."
    )
    ttl_seconds: Optional[int] = Field(
        default=None, description="Time to live in seconds."
    )
    _mongo_client: Optional[MongoClient] = PrivateAttr()
    _async_client: Optional[AsyncIOMotorClient] = PrivateAttr()

    def __init__(
        self,
        mongo_uri: str = "mongodb://localhost:27017",
        db_name: str = "default",
        collection_name: str = "sessions",
        mongo_client: Optional[MongoClient] = None,
        amongo_client: Optional[AsyncIOMotorClient] = None,
        ttl_seconds: Optional[int] = None,
        collection: Optional[Collection] = None,
        async_collection: Optional[AsyncIOMotorCollection] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the MongoDB chat store.

        Args:
            mongo_uri: MongoDB connection URI
            db_name: Database name
            collection_name: Collection name for storing chat messages
            mongo_client: Optional pre-configured MongoDB client
            amongo_client: Optional pre-configured async MongoDB client
            ttl_seconds: Optional time-to-live for messages in seconds
            **kwargs: Additional arguments to pass to MongoDB client
        """
        super().__init__(ttl=ttl_seconds)

        self._mongo_client = mongo_client or MongoClient(mongo_uri, **kwargs)
        self._async_client = amongo_client or AsyncIOMotorClient(mongo_uri, **kwargs)

        if collection:
            self._collection = collection
        else:
            self._collection = self._mongo_client[db_name][collection_name]

        if async_collection:
            self._async_collection = async_collection
        else:
            self._async_collection = self._async_client[db_name][collection_name]

        # Create TTL index if ttl is specified
        if ttl_seconds:
            self._collection.create_index("created_at", expireAfterSeconds=ttl_seconds)

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "MongoChatStore"

    def set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """
        Set messages for a key.

        Args:
            key: Key to set messages for
            messages: List of ChatMessage objects
        """
        # Delete existing messages for this key
        self._collection.delete_many({"session_id": key})

        # Insert new messages
        if messages:
            current_time = datetime.now()
            message_dicts = [
                {
                    "session_id": key,
                    "index": i,
                    "message": _message_to_dict(msg),
                    "created_at": current_time,
                }
                for i, msg in enumerate(messages)
            ]
            self._collection.insert_many(message_dicts)

    async def aset_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """
        Set messages for a key asynchronously.

        Args:
            key: Key to set messages for
            messages: List of ChatMessage objects
        """
        # Delete existing messages for this key
        await self._async_collection.delete_many({"session_id": key})

        # Insert new messages
        if messages:
            current_time = datetime.now()
            message_dicts = [
                {
                    "session_id": key,
                    "index": i,
                    "message": _message_to_dict(msg),
                    "created_at": current_time,
                }
                for i, msg in enumerate(messages)
            ]
            await self._async_collection.insert_many(message_dicts)

    def get_messages(self, key: str) -> List[ChatMessage]:
        """
        Get messages for a key.

        Args:
            key: Key to get messages for
        """
        # Find all messages for this key, sorted by index
        docs = list(self._collection.find({"session_id": key}, sort=[("index", 1)]))

        # Convert to ChatMessage objects
        return [_dict_to_message(doc["message"]) for doc in docs]

    async def aget_messages(self, key: str) -> List[ChatMessage]:
        """
        Get messages for a key asynchronously.

        Args:
            key: Key to get messages for
        """
        # Find all messages for this key, sorted by index
        cursor = self._async_collection.find({"session_id": key}).sort("index", 1)

        # Convert to list and then to ChatMessage objects
        docs = await cursor.to_list(length=None)
        return [_dict_to_message(doc["message"]) for doc in docs]

    def add_message(
        self, key: str, message: ChatMessage, idx: Optional[int] = None
    ) -> None:
        """
        Add a message for a key.

        Args:
            key: Key to add message for
            message: ChatMessage object to add
        """
        if idx is None:
            # Get the current highest index
            highest_idx_doc = self._collection.find_one(
                {"session_id": key}, sort=[("index", -1)]
            )
            idx = 0 if highest_idx_doc is None else highest_idx_doc["index"] + 1

        # Insert the new message with current timestamp
        self._collection.insert_one(
            {
                "session_id": key,
                "index": idx,
                "message": _message_to_dict(message),
                "created_at": datetime.now(),
            }
        )

    async def async_add_message(
        self, key: str, message: ChatMessage, idx: Optional[int] = None
    ) -> None:
        """
        Add a message for a key asynchronously.

        Args:
            key: Key to add message for
            message: ChatMessage object to add
        """
        if idx is None:
            # Get the current highest index
            highest_idx_doc = await self._async_collection.find_one(
                {"session_id": key}, sort=[("index", -1)]
            )
            idx = 0 if highest_idx_doc is None else highest_idx_doc["index"] + 1

        # Insert the new message with current timestamp
        await self._async_collection.insert_one(
            {
                "session_id": key,
                "index": idx,
                "message": _message_to_dict(message),
                "created_at": datetime.now(),
            }
        )

    def delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        """
        Delete messages for a key.

        Args:
            key: Key to delete messages for
        """
        # Get messages before deleting
        messages = self.get_messages(key)

        # Delete all messages for this key
        self._collection.delete_many({"session_id": key})

        return messages

    async def adelete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        """
        Delete messages for a key asynchronously.

        Args:
            key: Key to delete messages for
        """
        # Get messages before deleting
        messages = await self.aget_messages(key)

        # Delete all messages for this key
        await self._async_collection.delete_many({"session_id": key})

        return messages

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """
        Delete specific message for a key.

        Args:
            key: Key to delete message for
            idx: Index of message to delete
        """
        # Find the message to delete
        doc = self._collection.find_one({"session_id": key, "index": idx})
        if doc is None:
            return None

        # Delete the message
        self._collection.delete_one({"session_id": key, "index": idx})

        # Reindex remaining messages
        self._collection.update_many(
            {"session_id": key, "index": {"$gt": idx}}, {"$inc": {"index": -1}}
        )

        return _dict_to_message(doc["message"])

    async def adelete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """
        Delete specific message for a key asynchronously.

        Args:
            key: Key to delete message for
            idx: Index of message to delete
        """
        # Find the message to delete
        doc = await self._async_collection.find_one({"session_id": key, "index": idx})
        if doc is None:
            return None

        # Delete the message
        await self._async_collection.delete_one({"session_id": key, "index": idx})

        # Reindex remaining messages
        await self._async_collection.update_many(
            {"session_id": key, "index": {"$gt": idx}}, {"$inc": {"index": -1}}
        )

        return _dict_to_message(doc["message"])

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        """
        Delete last message for a key.

        Args:
            key: Key to delete last message for
        """
        # Find the last message
        last_msg_doc = self._collection.find_one(
            {"session_id": key}, sort=[("index", -1)]
        )

        if last_msg_doc is None:
            return None

        # Delete the last message
        self._collection.delete_one({"_id": last_msg_doc["_id"]})

        return _dict_to_message(last_msg_doc["message"])

    async def adelete_last_message(self, key: str) -> Optional[ChatMessage]:
        """
        Delete last message for a key asynchronously.

        Args:
            key: Key to delete last message for
        """
        # Find the last message
        last_msg_doc = await self._async_collection.find_one(
            {"session_id": key}, sort=[("index", -1)]
        )

        if last_msg_doc is None:
            return None

        # Delete the last message
        await self._async_collection.delete_one({"_id": last_msg_doc["_id"]})

        return _dict_to_message(last_msg_doc["message"])

    def get_keys(self) -> List[str]:
        """
        Get all keys (session IDs).

        Returns:
            List of session IDs
        """
        # Get distinct session IDs
        return self._collection.distinct("session_id")

    async def aget_keys(self) -> List[str]:
        """
        Get all keys (session IDs) asynchronously.

        Returns:
            List of session IDs
        """
        # Get distinct session IDs
        return await self._async_collection.distinct("session_id")
