import uuid
from typing import Any

from llama_index.core.base.llms.types import (
    ChatMessage,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.custom import CustomLLM


class TestLLM(CustomLLM):
    __test__ = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(callback_manager=None, **kwargs)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata()

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return CompletionResponse(
            text="test output",
            additional_kwargs={
                "prompt": prompt,
            },
        )

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        def gen() -> CompletionResponseGen:
            text = "test output"
            text_so_far = ""
            for ch in text:
                text_so_far += ch
                yield CompletionResponse(
                    text=text_so_far,
                    delta=ch,
                    additional_kwargs={
                        "prompt": prompt,
                    },
                )

        return gen()


def test_basic() -> None:
    llm = TestLLM()

    prompt = "test prompt"
    message = ChatMessage(role="user", content="test message")

    llm.complete(prompt)
    llm.chat([message])


def test_streaming() -> None:
    llm = TestLLM()

    prompt = "test prompt"
    message = ChatMessage(role="user", content="test message")

    llm.stream_complete(prompt)
    llm.stream_chat([message])


def test_llm_id():
    llm_id = uuid.uuid4().hex
    llm = TestLLM(id_=llm_id)
    assert llm.id_ == llm_id
