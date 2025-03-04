from typing import Any, Dict, List, Sequence
from markdown_it import MarkdownIt
from llama_index.core.schema import (
    BaseNode,
    TextNode,
    NodeRelationship,
    TransformComponent,
)
from pydantic import Field, ConfigDict


class ChatGPTMessageNodeParser(TransformComponent):
    """Processes messages from the document text, leveraging speaker identities."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    md: MarkdownIt = Field(default_factory=MarkdownIt, exclude=True)

    def __call__(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> List[TextNode]:
        """Process nodes into TextNodes."""
        all_nodes = []
        for doc in nodes:
            if not isinstance(doc, BaseNode):
                continue
            # Split the document text into messages based on speaker identity
            messages = self._split_text_into_messages(doc.text)
            messages_metadata = doc.metadata.get("messages", [])
            # Process messages
            doc_nodes = self._process_messages(messages, messages_metadata, doc)
            all_nodes.extend(doc_nodes)
        return all_nodes

    def _split_text_into_messages(self, text: str) -> List[Dict[str, str]]:
        """Split text into messages using speaker identities."""
        # Assuming each message starts with 'user:' or 'assistant:'
        messages = []
        lines = text.strip().split("\n")
        for line in lines:
            if line.startswith(("user:", "assistant:")):
                author_role, message_text = line.split(":", 1)
                messages.append(
                    {"author_role": author_role.strip(), "text": message_text.strip()}
                )
        return messages

    def _process_messages(
        self,
        messages: List[Dict[str, str]],
        messages_metadata: List[Dict[str, Any]],
        doc: BaseNode,
    ) -> List[TextNode]:
        """Process messages and return TextNodes."""
        nodes = []
        prev_node = None
        for idx, message in enumerate(messages):
            message_text = message["text"]
            author_role = message["author_role"]
            # Find metadata for this message
            message_metadata = next(
                (
                    m
                    for m in messages_metadata
                    if m["author_role"] == author_role and m["text"] == message_text
                ),
                {},
            )
            metadata = {
                **doc.metadata,
                **message_metadata,
            }
            ref_doc_id = doc.id_
            source_node = doc

            # Process Markdown content
            tokens = self.md.parse(message_text)
            message_nodes = self._process_tokens(
                tokens, metadata, ref_doc_id, source_node
            )

            # Include previous user message for context in assistant messages
            if author_role == "assistant" and idx > 0:
                # Find the last user message
                context_message = messages[idx - 1]
                if context_message["author_role"] == "user":
                    context_text = context_message["text"]
                    context_metadata = {
                        **doc.metadata,
                        **next(
                            (
                                m
                                for m in messages_metadata
                                if m["author_role"] == "user"
                                and m["text"] == context_text
                            ),
                            {},
                        ),
                    }
                    # Process context message
                    context_tokens = self.md.parse(context_text)
                    context_nodes = self._process_tokens(
                        context_tokens, context_metadata, ref_doc_id, source_node
                    )
                    # Set up relationships between context and message nodes
                    if context_nodes and message_nodes:
                        for i in range(len(context_nodes) - 1):
                            context_nodes[i].relationships[
                                NodeRelationship.NEXT
                            ] = context_nodes[i + 1].as_related_node_info()
                            context_nodes[i + 1].relationships[
                                NodeRelationship.PREVIOUS
                            ] = context_nodes[i].as_related_node_info()
                        context_nodes[-1].relationships[
                            NodeRelationship.NEXT
                        ] = message_nodes[0].as_related_node_info()
                        message_nodes[0].relationships[
                            NodeRelationship.PREVIOUS
                        ] = context_nodes[-1].as_related_node_info()
                        message_nodes = context_nodes + message_nodes
                    elif context_nodes:
                        message_nodes = context_nodes

            # Set up relationships within message nodes
            for i in range(len(message_nodes) - 1):
                message_nodes[i].relationships[NodeRelationship.NEXT] = message_nodes[
                    i + 1
                ].as_related_node_info()
                message_nodes[i + 1].relationships[
                    NodeRelationship.PREVIOUS
                ] = message_nodes[i].as_related_node_info()

            # Set up relationships with previous node
            if prev_node and message_nodes:
                message_nodes[0].relationships[
                    NodeRelationship.PREVIOUS
                ] = prev_node.as_related_node_info()
                prev_node.relationships[NodeRelationship.NEXT] = message_nodes[
                    0
                ].as_related_node_info()

            if message_nodes:
                prev_node = message_nodes[-1]
                nodes.extend(message_nodes)
            else:
                # Optionally, handle the case where message_nodes is empty
                # For example, log a warning or skip to the next message
                pass

        return nodes

    def _process_tokens(
        self,
        tokens: List[Any],
        metadata: Dict[str, Any],
        ref_doc_id: str,
        source_node: BaseNode,
    ) -> List[TextNode]:
        """Process Markdown tokens into nodes with structure."""
        nodes = []
        current_text = ""
        header_path = []
        for idx, token in enumerate(tokens):
            # Header handling
            if token.type == "heading_open":
                if current_text.strip():
                    nodes.append(
                        self._build_text_node(
                            current_text,
                            metadata,
                            "/".join(header_path),
                            ref_doc_id,
                            source_node,
                        )
                    )
                    current_text = ""
                header_level = int(token.tag[1])
                next_token = tokens[idx + 1] if idx + 1 < len(tokens) else None
                header_text = next_token.content if next_token else ""
                header_path = header_path[: header_level - 1] + [header_text]
            # Code block handling
            elif token.type == "fence":
                if current_text.strip():
                    nodes.append(
                        self._build_text_node(
                            current_text,
                            metadata,
                            "/".join(header_path),
                            ref_doc_id,
                            source_node,
                        )
                    )
                    current_text = ""
                code_language = token.info.strip() or "plain"
                code_block = token.content
                code_node = TextNode(
                    text=code_block.strip(),
                    metadata={
                        **metadata,
                        "Content Type": "code",
                        "Language": code_language,
                        "Header Path": "/".join(header_path),
                    },
                    ref_doc_id=ref_doc_id,
                    source_node=source_node,
                )
                nodes.append(code_node)
            # Inline code
            elif token.type == "code_inline":
                inline_code = token.content
                current_text += f"`{inline_code}`"
            # Paragraph handling
            elif token.type == "paragraph_open":
                continue
            elif token.type == "paragraph_close":
                if current_text.strip():
                    nodes.append(
                        self._build_text_node(
                            current_text,
                            metadata,
                            "/".join(header_path),
                            ref_doc_id,
                            source_node,
                        )
                    )
                    current_text = ""
            elif token.type == "inline":
                current_text += token.content + " "
            # Text
            elif token.type == "text":
                current_text += token.content
            # Handle other token types as needed

        if current_text.strip():
            nodes.append(
                self._build_text_node(
                    current_text,
                    metadata,
                    "/".join(header_path),
                    ref_doc_id,
                    source_node,
                )
            )
        return nodes

    def _build_text_node(
        self,
        text: str,
        metadata: Dict[str, Any],
        header_path: str,
        ref_doc_id: str,
        source_node: BaseNode,
    ) -> TextNode:
        """Build a text node with metadata."""
        return TextNode(
            text=text.strip(),
            metadata={**metadata, "Content Type": "text", "Header Path": header_path},
            ref_doc_id=ref_doc_id,
            source_node=source_node,
        )
