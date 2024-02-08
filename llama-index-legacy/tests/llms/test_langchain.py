from typing import List

import pytest
from llama_index.legacy.core.llms.types import ChatMessage, MessageRole

try:
    import cohere
except ImportError:
    cohere = None  # type: ignore

try:
    import langchain

    class LC:
        from llama_index.legacy.bridge.langchain import (
            AIMessage,
            BaseMessage,
            ChatMessage,
            ChatOpenAI,
            Cohere,
            FakeListLLM,
            FunctionMessage,
            HumanMessage,
            OpenAI,
            SystemMessage,
        )

    from llama_index.legacy.llms.langchain import LangChainLLM
    from llama_index.legacy.llms.langchain_utils import from_lc_messages, to_lc_messages

except ImportError:
    langchain = None  # type: ignore


@pytest.mark.skipif(langchain is None, reason="langchain not installed")
def test_basic() -> None:
    lc_llm = LC.FakeListLLM(responses=["test response 1", "test response 2"])
    llm = LangChainLLM(llm=lc_llm)

    prompt = "test prompt"
    message = ChatMessage(role="user", content="test message")

    llm.complete(prompt)
    llm.chat([message])


@pytest.mark.skipif(langchain is None, reason="langchain not installed")
def test_to_lc_messages() -> None:
    lc_messages: List[LC.BaseMessage] = [
        LC.SystemMessage(content="test system message"),
        LC.HumanMessage(content="test human message"),
        LC.AIMessage(content="test ai message"),
        LC.FunctionMessage(content="test function message", name="test function"),
        LC.ChatMessage(content="test function message", role="user"),
    ]

    messages = from_lc_messages(lc_messages)

    for i in range(len(messages)):
        assert messages[i].content == lc_messages[i].content


@pytest.mark.skipif(langchain is None, reason="langchain not installed")
def test_from_lc_messages() -> None:
    messages = [
        ChatMessage(content="test system message", role=MessageRole.SYSTEM),
        ChatMessage(content="test human message", role=MessageRole.USER),
        ChatMessage(content="test ai message", role=MessageRole.ASSISTANT),
        ChatMessage(
            content="test function message",
            role=MessageRole.FUNCTION,
            additional_kwargs={"name": "test function"},
        ),
        ChatMessage(
            content="test chat message",
            role=MessageRole.CHATBOT,
            additional_kwargs={"role": "user"},
        ),
    ]

    lc_messages = to_lc_messages(messages)

    for i in range(len(messages)):
        assert messages[i].content == lc_messages[i].content


@pytest.mark.skipif(
    cohere is None or langchain is None, reason="cohere or langchain not installed"
)
def test_metadata_sets_model_name() -> None:
    chat_gpt = LangChainLLM(
        llm=LC.ChatOpenAI(model="gpt-4-0613", openai_api_key="model-name-tests")
    )
    assert chat_gpt.metadata.model_name == "gpt-4-0613"

    gpt35 = LangChainLLM(
        llm=LC.OpenAI(model="gpt-3.5-turbo-0613", openai_api_key="model-name-tests")
    )
    assert gpt35.metadata.model_name == "gpt-3.5-turbo-0613"

    cohere_llm = LangChainLLM(
        llm=LC.Cohere(model="j2-jumbo-instruct", cohere_api_key="XXXXXXX")
    )
    assert cohere_llm.metadata.model_name == "j2-jumbo-instruct"
