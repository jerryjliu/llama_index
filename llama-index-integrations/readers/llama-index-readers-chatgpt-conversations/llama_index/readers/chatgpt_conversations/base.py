import json
from typing import Any, Dict, List, Tuple, Optional, Union
from datetime import datetime
from pathlib import Path
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class ChatGPTConversationsReader(BaseReader):
    """Custom JSON Reader for structured ChatGPT conversation data."""

    def __init__(
        self,
        input_file: Union[str, Path],
        ensure_ascii: bool = False,
        is_jsonl: Optional[bool] = False,
        max_items: Optional[int] = None,
    ) -> None:
        """Initialize with arguments."""
        super().__init__()
        self.input_file = Path(input_file)
        self.ensure_ascii = ensure_ascii
        self.is_jsonl = is_jsonl
        self.max_items = max_items

    def load_data(self, extra_info: Optional[Dict] = {}) -> List[Document]:
        """Load data from the input file."""
        with open(self.input_file, encoding="utf-8") as f:
            data = (
                json.load(f)
                if not self.is_jsonl
                else [json.loads(line.strip()) for line in f]
            )

            # Ensure data is a list of conversations
            if isinstance(data, dict):
                load_data = [data]
            elif isinstance(data, list):
                load_data = data
            else:
                raise ValueError("Invalid JSON format: Expected a dict or list.")

            documents = []
            processed_count = 0

            # Loop through each conversation in the JSON array
            for conversation in load_data:
                if self.max_items is not None and processed_count >= self.max_items:
                    break

                # Extract the conversation_id
                conversation_id = conversation.get(
                    "conversation_id", f"doc_{processed_count}"
                )

                # Basic metadata for each conversation
                metadata = {
                    "title": conversation.get("title", "No Title"),
                    "conversation_id": conversation_id,
                    "create_time": datetime.fromtimestamp(
                        conversation.get("create_time", 0)
                    ).isoformat(),
                    "update_time": datetime.fromtimestamp(
                        conversation.get("update_time", 0)
                    ).isoformat(),
                    **extra_info,
                }

                # Convert the conversation into plain text with speaker identities
                (
                    conversation_text,
                    messages_metadata,
                ) = self._convert_conversation_to_text(conversation)

                # Add messages metadata to the document metadata
                metadata["messages"] = messages_metadata

                # Add the conversation as a Document
                documents.append(
                    Document(
                        text=conversation_text,
                        metadata=metadata,
                        id_=conversation_id,  # Use conversation_id as id_
                    )
                )
                processed_count += 1

            print(f"Number of documents created: {len(documents)}")  # Debug statement

            return documents

    def _convert_conversation_to_text(
        self, conversation: Dict
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Convert conversation JSON to plain text with speaker identities and collect message metadata."""
        mapping = conversation.get("mapping", {})
        if not mapping:
            print("No 'mapping' found in conversation")
            return "", []  # Return empty string if no messages

        # Identify the root node and start processing messages from the root
        root_node_id = next(
            (
                node_id
                for node_id, node in mapping.items()
                if node.get("parent") is None
            ),
            None,
        )
        messages = []
        messages_metadata = []

        def traverse_messages(node_id):
            node_data = mapping.get(node_id)
            if not node_data:
                return
            message = node_data.get("message")
            if message and "content" in message and "parts" in message["content"]:
                message_content = message["content"]["parts"]
                # Adjusted this section to handle non-string items
                message_text = ""
                for part in message_content:
                    if isinstance(part, str):
                        message_text += part
                    elif isinstance(part, dict):
                        # Convert dict to string (e.g., via JSON)
                        message_text += json.dumps(part)
                    else:
                        # Convert other types to string
                        message_text += str(part)
                author_role = message.get("author", {}).get("role", "unknown")
                message_id = node_id  # Use node_id as the message_id

                create_time = message.get("create_time") or message.get("update_time")
                # Adjust for milliseconds if needed
                if create_time and create_time > 1e12:  # Timestamp in milliseconds
                    create_time = create_time / 1000.0
                timestamp = (
                    datetime.fromtimestamp(create_time).strftime("%H:%M")
                    if create_time
                    else "Unknown time"
                )
                message_date = (
                    datetime.fromtimestamp(create_time).strftime("%Y-%m-%d")
                    if create_time
                    else None
                )

                # Append message text with speaker identity
                messages.append(f"{author_role}: {message_text}")

                # Collect message metadata
                message_metadata = {
                    "message_id": message_id,
                    "author_role": author_role,
                    "timestamp": timestamp,
                    "date": message_date,
                    "text": message_text,
                    # Include other metadata as needed
                }
                messages_metadata.append(message_metadata)
            for child_id in node_data.get("children", []):
                traverse_messages(child_id)

        if root_node_id:
            traverse_messages(root_node_id)
        else:
            print("No root node found in conversation mapping")

        # Join messages into a single text
        conversation_text = "\n".join(messages)
        return conversation_text, messages_metadata
