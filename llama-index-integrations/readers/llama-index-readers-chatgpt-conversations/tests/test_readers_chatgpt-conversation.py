from llama_index.readers.chatgpt_conversations import ChatGPTConversationsReader
import json
from pathlib import Path


def test_load_data():
    # Prepare a sample conversation JSON
    sample_conversation = {
        "title": "Sample Conversation",
        "create_time": 1690000000,
        "update_time": 1690003600,
        "conversation_id": "test_convo_1",
        "mapping": {
            "node1": {
                "parent": None,
                "children": ["node2"],
                "message": {
                    "id": "node1",
                    "author": {"role": "user"},
                    "create_time": 1690000000,
                    "content": {"parts": ["Hello, how can I export my data?"]},
                },
            },
            "node2": {
                "parent": "node1",
                "children": [],
                "message": {
                    "id": "node2",
                    "author": {"role": "assistant"},
                    "create_time": 1690000100,
                    "content": {"parts": ["You can export your data by..."]},
                },
            },
        },
    }

    # Write sample data to a temporary JSON file
    temp_file = Path("temp_conversation.json")
    with temp_file.open("w", encoding="utf-8") as f:
        json.dump([sample_conversation], f)

    # Initialize the reader
    reader = ChatGPTConversationsReader(input_file=str(temp_file))

    # Load data
    documents = reader.load_data()
    assert len(documents) == 1

    # Check document content
    doc = documents[0]
    assert doc.metadata["title"] == "Sample Conversation"
    assert "user: Hello, how can I export my data?" in doc.text
    assert "assistant: You can export your data by..." in doc.text

    # Clean up
    temp_file.unlink()
