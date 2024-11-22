import pytest
from llama_index.core.base.llms.types import (
    ChatMessage,
    ImageBlock,
    MessageRole,
    TextBlock,
)
from llama_index.core.bridge.pydantic import BaseModel


def test_chat_message_from_str():
    m = ChatMessage.from_str(content="test content")
    assert m.content == "test content"
    assert len(m.blocks) == 1
    assert type(m.blocks[0]) is TextBlock
    assert m.blocks[0].text == "test content"


def test_chat_message_content_legacy_get():
    m = ChatMessage(content="test content")
    assert m.content == "test content"
    assert len(m.blocks) == 1
    assert type(m.blocks[0]) is TextBlock
    assert m.blocks[0].text == "test content"

    m = ChatMessage(role="user", content="test content")
    assert m.role == "user"
    assert m.content == "test content"
    assert len(m.blocks) == 1
    assert type(m.blocks[0]) is TextBlock
    assert m.blocks[0].text == "test content"

    m = ChatMessage(content=[TextBlock(text="test content")])
    assert m.content == "test content"
    assert len(m.blocks) == 1
    assert type(m.blocks[0]) is TextBlock
    assert m.blocks[0].text == "test content"


def test_chat_message_content_legacy_set():
    m = ChatMessage()
    m.content = "test content"
    assert len(m.blocks) == 1
    assert type(m.blocks[0]) is TextBlock
    assert m.blocks[0].text == "test content"

    m = ChatMessage(content="some original content")
    m.content = "test content"
    assert len(m.blocks) == 1
    assert type(m.blocks[0]) is TextBlock
    assert m.blocks[0].text == "test content"

    m = ChatMessage(content=[TextBlock(text="test content"), ImageBlock()])
    with pytest.raises(ValueError):
        m.content = "test content"


def test_chat_message_content_returns_blocks():
    m = ChatMessage(content=[TextBlock(text="test content"), ImageBlock()])
    assert isinstance(m.content, list)
    assert isinstance(m.content[0], TextBlock)
    assert isinstance(m.content[1], ImageBlock)


def test__str__():
    assert str(ChatMessage(content="test content")) == "user: test content"


def test_serializer():
    class SimpleModel(BaseModel):
        some_field: str = ""

    m = ChatMessage(
        content="test content",
        additional_kwargs={"some_list": ["a", "b", "c"], "some_object": SimpleModel()},
    )
    assert m.model_dump() == {
        "role": MessageRole.USER,
        "additional_kwargs": {
            "some_list": ["a", "b", "c"],
            "some_object": {"some_field": ""},
        },
        "blocks": [{}],
    }


{
    "description": "Base component object to capture class names.",
    "properties": {},
    "title": "BaseComponent",
    "type": "object",
}
{
    "description": "Base component object to capture class names.",
    "properties": {
        "class_name": {
            "default": "base_component",
            "title": "Class Name",
            "type": "string",
        }
    },
    "title": "BaseComponent",
    "type": "object",
}
