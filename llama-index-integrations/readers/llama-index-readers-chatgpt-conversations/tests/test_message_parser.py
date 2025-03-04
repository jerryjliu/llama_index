from llama_index.readers.chatgpt_conversations import ChatGPTMessageNodeParser
from llama_index.core.schema import Document


def test_message_parser():
    # Create a sample document
    doc_text = "user: Hello, how can I export my data?\nassistant: You can export your data by..."
    metadata = {
        "conversation_id": "test_convo_1",
        "messages": [
            {
                "message_id": "node1",
                "author_role": "user",
                "text": "Hello, how can I export my data?",
            },
            {
                "message_id": "node2",
                "author_role": "assistant",
                "text": "You can export your data by...",
            },
        ],
    }
    doc = Document(text=doc_text, metadata=metadata)

    # Initialize the parser
    parser = ChatGPTMessageNodeParser()

    # Parse the document
    nodes = parser([doc])
    assert len(nodes) > 0

    # Check node content
    user_node = nodes[0]
    assert user_node.metadata["author_role"] == "user"
    assert "export my data" in user_node.text

    assistant_node = nodes[-1]
    assert assistant_node.metadata["author_role"] == "assistant"
    assert "You can export your data" in assistant_node.text
