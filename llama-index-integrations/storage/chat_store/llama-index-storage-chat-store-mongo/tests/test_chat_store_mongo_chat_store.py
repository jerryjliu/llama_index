import time
import pytest
import docker
from typing import Dict, Generator, Union
from docker.models.containers import Container
from llama_index.core.storage.chat_store.base import BaseChatStore
from llama_index.storage.chat_store.mongo.base import MongoChatStore

from llama_index.core.llms import ChatMessage

try:
    import pymongo  # noqa: F401
    import motor.motor_asyncio  # noqa: F401

    no_packages = False
except ImportError:
    no_packages = True


def test_class():
    """Test that MongoChatStore inherits from BaseChatStore."""
    names_of_base_classes = [b.__name__ for b in MongoChatStore.__mro__]
    assert BaseChatStore.__name__ in names_of_base_classes


@pytest.fixture()
def mongo_container() -> Generator[Dict[str, Union[str, Container]], None, None]:
    """Fixture to create a MongoDB container for testing."""
    # Define MongoDB settings
    mongo_image = "mongo:latest"
    mongo_ports = {"27017/tcp": 27017}
    container = None
    try:
        # Initialize Docker client
        client = docker.from_env()

        # Run MongoDB container
        container = client.containers.run(mongo_image, ports=mongo_ports, detach=True)

        # Wait for MongoDB to start
        time.sleep(5)  # Give MongoDB time to initialize

        # Return connection information
        yield {
            "container": container,
            "mongodb_uri": "mongodb://localhost:27017/",
        }
    finally:
        # Stop and remove the container
        if container:
            container.stop()
            container.remove()
            client.close()


@pytest.fixture()
@pytest.mark.skipif(no_packages, reason="pymongo and motor not installed")
def mongo_chat_store(
    mongo_container: Dict[str, Union[str, Container]],
) -> Generator[MongoChatStore, None, None]:
    """Fixture to create a MongoChatStore instance connected to the test container."""
    chat_store = None
    try:
        chat_store = MongoChatStore(
            mongo_uri=mongo_container["mongodb_uri"],
            db_name="test_db",
            collection_name="test_chats",
        )
        yield chat_store
    finally:
        if chat_store and hasattr(chat_store, "_collection"):
            # Clean up by dropping the collection
            chat_store._collection.drop()


@pytest.mark.skipif(no_packages, reason="pymongo and motor not installed")
def test_create_chat(mongo_chat_store: MongoChatStore):
    """Test creating a chat session."""
    # Create a chat with metadata
    key = "test_key"
    messages = [
        ChatMessage(role="user", content="Hello, how are you?"),
        ChatMessage(role="assistant", content="I'm doing well, thank you!"),
    ]
    mongo_chat_store.set_messages(key=key, messages=messages)

    # Get all keys and verify our chat_id is there
    keys = mongo_chat_store.get_keys()
    assert key in keys


@pytest.mark.skipif(no_packages, reason="pymongo and motor not installed")
def test_get_messages(mongo_chat_store: MongoChatStore):
    """Test retrieving messages from a chat session."""
    # Create a chat with messages
    key = "test_get_messages"
    messages = [
        ChatMessage(role="user", content="Hello, MongoDB!"),
        ChatMessage(role="assistant", content="Hello, user! How can I help you?"),
        ChatMessage(role="user", content="I need information about databases."),
    ]
    mongo_chat_store.set_messages(key=key, messages=messages)

    # Retrieve the messages
    retrieved_messages = mongo_chat_store.get_messages(key=key)

    # Verify the messages were retrieved correctly
    assert len(retrieved_messages) == 3
    assert retrieved_messages[0].role == "user"
    assert retrieved_messages[0].content == "Hello, MongoDB!"
    assert retrieved_messages[1].role == "assistant"
    assert retrieved_messages[1].content == "Hello, user! How can I help you?"
    assert retrieved_messages[2].role == "user"
    assert retrieved_messages[2].content == "I need information about databases."


@pytest.mark.skipif(no_packages, reason="pymongo and motor not installed")
def test_add_message(mongo_chat_store: MongoChatStore):
    """Test adding a message to an existing chat session."""
    # Create a chat with initial messages
    key = "test_add_message"
    initial_messages = [ChatMessage(role="user", content="Initial message")]
    mongo_chat_store.set_messages(key=key, messages=initial_messages)

    # Add a new message
    new_message = ChatMessage(role="assistant", content="Response to initial message")
    mongo_chat_store.add_message(key=key, message=new_message)

    # Retrieve all messages
    messages = mongo_chat_store.get_messages(key=key)

    # Verify the new message was added
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "Initial message"
    assert messages[1].role == "assistant"
    assert messages[1].content == "Response to initial message"


@pytest.mark.skipif(no_packages, reason="pymongo and motor not installed")
def test_delete_messages(mongo_chat_store: MongoChatStore):
    """Test deleting all messages from a chat session."""
    # Create a chat with messages
    key = "test_delete_messages"
    messages = [
        ChatMessage(role="user", content="Message to be deleted"),
        ChatMessage(role="assistant", content="This will also be deleted"),
    ]
    mongo_chat_store.set_messages(key=key, messages=messages)

    # Delete all messages
    deleted_messages = mongo_chat_store.delete_messages(key=key)

    # Verify the messages were deleted
    assert len(deleted_messages) == 2
    assert deleted_messages[0].content == "Message to be deleted"

    # Verify the chat is empty
    remaining_messages = mongo_chat_store.get_messages(key=key)
    assert len(remaining_messages) == 0

    # Verify the key is not present in the store
    keys = mongo_chat_store.get_keys()
    assert key not in keys


@pytest.mark.skipif(no_packages, reason="pymongo and motor not installed")
def test_delete_message(mongo_chat_store: MongoChatStore):
    """Test deleting a specific message from a chat session."""
    # Create a chat with multiple messages
    key = "test_delete_specific"
    messages = [
        ChatMessage(role="user", content="First message"),
        ChatMessage(role="assistant", content="Middle message to delete"),
        ChatMessage(role="user", content="Last message"),
    ]
    mongo_chat_store.set_messages(key=key, messages=messages)

    # Delete the middle message
    deleted_message = mongo_chat_store.delete_message(key=key, idx=1)

    # Verify the correct message was deleted
    assert deleted_message.role == "assistant"
    assert deleted_message.content == "Middle message to delete"

    # Verify the remaining messages are correct and reindexed
    remaining = mongo_chat_store.get_messages(key=key)
    assert len(remaining) == 2
    assert remaining[0].content == "First message"
    assert remaining[1].content == "Last message"


@pytest.mark.skipif(no_packages, reason="pymongo and motor not installed")
def test_delete_last_message(mongo_chat_store: MongoChatStore):
    """Test deleting the last message from a chat session."""
    # Create a chat with messages
    key = "test_delete_last"
    messages = [
        ChatMessage(role="user", content="First message"),
        ChatMessage(role="assistant", content="Last message to delete"),
    ]
    mongo_chat_store.set_messages(key=key, messages=messages)

    # Delete the last message
    deleted = mongo_chat_store.delete_last_message(key=key)

    # Verify the correct message was deleted
    assert deleted.role == "assistant"
    assert deleted.content == "Last message to delete"

    # Verify only the first message remains
    remaining = mongo_chat_store.get_messages(key=key)
    assert len(remaining) == 1
    assert remaining[0].content == "First message"


@pytest.mark.skipif(no_packages, reason="pymongo and motor not installed")
@pytest.mark.asyncio()
async def test_async_get_messages(mongo_chat_store: MongoChatStore):
    """Test retrieving messages asynchronously."""
    # Create a chat with messages
    key = "test_async_get"
    messages = [
        ChatMessage(role="user", content="Async test message"),
        ChatMessage(role="assistant", content="Async response"),
    ]
    mongo_chat_store.set_messages(key=key, messages=messages)

    # Retrieve messages asynchronously
    retrieved = await mongo_chat_store.aget_messages(key=key)

    # Verify messages were retrieved correctly
    assert len(retrieved) == 2
    assert retrieved[0].content == "Async test message"
    assert retrieved[1].content == "Async response"


@pytest.mark.skipif(no_packages, reason="pymongo and motor not installed")
@pytest.mark.asyncio()
async def test_async_add_message(mongo_chat_store: MongoChatStore):
    """Test adding a message asynchronously."""
    key = "test_async_add"
    initial_message = ChatMessage(role="user", content="Initial async message")
    mongo_chat_store.set_messages(key=key, messages=[initial_message])

    # Add message asynchronously
    new_message = ChatMessage(role="assistant", content="Async response")
    await mongo_chat_store.async_add_message(key=key, message=new_message)

    # Verify message was added
    messages = mongo_chat_store.get_messages(key=key)
    assert len(messages) == 2
    assert messages[1].content == "Async response"


@pytest.mark.skipif(no_packages, reason="pymongo and motor not installed")
@pytest.mark.asyncio()
async def test_async_set_messages(mongo_chat_store: MongoChatStore):
    """Test setting messages asynchronously."""
    key = "test_async_set"
    messages = [
        ChatMessage(role="user", content="First async set message"),
        ChatMessage(role="assistant", content="Second async set message"),
    ]

    # Set messages asynchronously
    await mongo_chat_store.aset_messages(key=key, messages=messages)

    # Verify messages were set correctly
    retrieved = await mongo_chat_store.aget_messages(key=key)
    assert len(retrieved) == 2
    assert retrieved[0].content == "First async set message"
    assert retrieved[1].content == "Second async set message"


@pytest.mark.skipif(no_packages, reason="pymongo and motor not installed")
@pytest.mark.asyncio()
async def test_async_delete_messages(mongo_chat_store: MongoChatStore):
    """Test deleting all messages asynchronously."""
    key = "test_async_delete_all"
    messages = [
        ChatMessage(role="user", content="Async message to delete 1"),
        ChatMessage(role="assistant", content="Async message to delete 2"),
    ]
    mongo_chat_store.set_messages(key=key, messages=messages)

    # Delete messages asynchronously
    deleted = await mongo_chat_store.adelete_messages(key=key)

    # Verify messages were deleted
    assert len(deleted) == 2
    assert deleted[0].content == "Async message to delete 1"

    # Verify no messages remain
    remaining = await mongo_chat_store.aget_messages(key=key)
    assert len(remaining) == 0

    # Verify key is not in store
    keys = await mongo_chat_store.aget_keys()
    assert key not in keys


@pytest.mark.skipif(no_packages, reason="pymongo and motor not installed")
@pytest.mark.asyncio()
async def test_async_delete_message(mongo_chat_store: MongoChatStore):
    """Test deleting a specific message asynchronously."""
    key = "test_async_delete_specific"
    messages = [
        ChatMessage(role="user", content="Async first message"),
        ChatMessage(role="assistant", content="Async middle message to delete"),
        ChatMessage(role="user", content="Async last message"),
    ]
    mongo_chat_store.set_messages(key=key, messages=messages)

    # Delete middle message asynchronously
    deleted = await mongo_chat_store.adelete_message(key=key, idx=1)

    # Verify correct message was deleted
    assert deleted.role == "assistant"
    assert deleted.content == "Async middle message to delete"

    # Verify remaining messages and reindexing
    remaining = await mongo_chat_store.aget_messages(key=key)
    assert len(remaining) == 2
    assert remaining[0].content == "Async first message"
    assert remaining[1].content == "Async last message"


@pytest.mark.skipif(no_packages, reason="pymongo and motor not installed")
@pytest.mark.asyncio()
async def test_async_delete_last_message(mongo_chat_store: MongoChatStore):
    """Test deleting the last message asynchronously."""
    key = "test_async_delete_last"
    messages = [
        ChatMessage(role="user", content="Async first message"),
        ChatMessage(role="assistant", content="Async last message to delete"),
    ]
    mongo_chat_store.set_messages(key=key, messages=messages)

    # Delete last message asynchronously
    deleted = await mongo_chat_store.adelete_last_message(key=key)

    # Verify correct message was deleted
    assert deleted.role == "assistant"
    assert deleted.content == "Async last message to delete"

    # Verify only first message remains
    remaining = await mongo_chat_store.aget_messages(key=key)
    assert len(remaining) == 1
    assert remaining[0].content == "Async first message"


@pytest.mark.skipif(no_packages, reason="pymongo and motor not installed")
@pytest.mark.asyncio()
async def test_async_get_keys(mongo_chat_store: MongoChatStore):
    """Test getting all keys asynchronously."""
    # Create multiple chats
    await mongo_chat_store.aset_messages(
        key="async_keys_test1",
        messages=[ChatMessage(role="user", content="Test message 1")],
    )
    await mongo_chat_store.aset_messages(
        key="async_keys_test2",
        messages=[ChatMessage(role="user", content="Test message 2")],
    )

    # Get keys asynchronously
    keys = await mongo_chat_store.aget_keys()

    # Verify keys were retrieved
    assert "async_keys_test1" in keys
    assert "async_keys_test2" in keys


@pytest.mark.skipif(no_packages, reason="pymongo and motor not installed")
def test_nonexistent_key(mongo_chat_store: MongoChatStore):
    """Test behavior with nonexistent keys."""
    # Try to get messages for nonexistent key
    messages = mongo_chat_store.get_messages(key="nonexistent_key")

    # Verify empty list is returned
    assert messages == []

    # Try to delete a message from nonexistent chat
    deleted = mongo_chat_store.delete_message(key="nonexistent_key", idx=0)

    # Verify None is returned
    assert deleted is None

    # Try to delete last message from nonexistent chat
    deleted = mongo_chat_store.delete_last_message(key="nonexistent_key")

    # Verify None is returned
    assert deleted is None


@pytest.mark.skipif(no_packages, reason="pymongo and motor not installed")
def test_ttl_configuration(mongo_container):
    """Test TTL configuration is applied correctly."""
    # Create chat store with TTL
    chat_store = MongoChatStore(
        mongo_uri=mongo_container["mongodb_uri"],
        db_name="test_ttl_db",
        collection_name="test_ttl_chats",
        ttl_seconds=3600,  # 1 hour TTL
    )

    # Verify TTL index was created
    indexes = list(chat_store._collection.list_indexes())
    ttl_index = next((idx for idx in indexes if "created_at" in idx["key"]), None)

    assert ttl_index is not None
    assert ttl_index.get("expireAfterSeconds") == 3600

    # Clean up
    chat_store._collection.drop()


@pytest.mark.skipif(no_packages, reason="pymongo and motor not installed")
def test_invalid_message_index(mongo_chat_store: MongoChatStore):
    """Test behavior when trying to delete a message with invalid index."""
    key = "test_invalid_index"
    messages = [ChatMessage(role="user", content="Only message")]
    mongo_chat_store.set_messages(key=key, messages=messages)

    # Try to delete message with out-of-range index
    deleted = mongo_chat_store.delete_message(key=key, idx=5)

    # Verify None is returned
    assert deleted is None

    # Verify original message still exists
    remaining = mongo_chat_store.get_messages(key=key)
    assert len(remaining) == 1
    assert remaining[0].content == "Only message"


@pytest.mark.skipif(no_packages, reason="pymongo and motor not installed")
def test_multiple_clients(mongo_container):
    """Test using multiple chat store instances with the same database."""
    # Create two chat store instances
    chat_store1 = MongoChatStore(
        mongo_uri=mongo_container["mongodb_uri"],
        db_name="test_multi_client_db",
        collection_name="test_chats",
    )

    chat_store2 = MongoChatStore(
        mongo_uri=mongo_container["mongodb_uri"],
        db_name="test_multi_client_db",
        collection_name="test_chats",
    )

    # Add message with first client
    key = "test_multi_client"
    chat_store1.set_messages(
        key=key, messages=[ChatMessage(role="user", content="Message from client 1")]
    )

    # Add message with second client
    chat_store2.add_message(
        key=key, message=ChatMessage(role="assistant", content="Message from client 2")
    )

    # Verify both messages are visible to both clients
    messages1 = chat_store1.get_messages(key=key)
    messages2 = chat_store2.get_messages(key=key)

    assert len(messages1) == 2
    assert len(messages2) == 2
    assert messages1[0].content == "Message from client 1"
    assert messages1[1].content == "Message from client 2"

    # Clean up
    chat_store1._collection.drop()
