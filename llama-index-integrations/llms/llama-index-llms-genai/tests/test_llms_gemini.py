import asyncio
import os
from typing import List

import pytest
from google.genai import types
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    ImageBlock,
    MessageRole,
    TextBlock,
)
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.program.function_program import get_function_tool
from llama_index.core.prompts import ChatPromptTemplate, PromptTemplate
from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field

from llama_index.llms.genai import Gemini
from llama_index.llms.genai.utils import convert_schema_to_function_declaration


class Poem(BaseModel):
    content: str


class Column(BaseModel):
    name: str = Field(description="Column field")
    data_type: str = Field(description="Data type field")


class Table(BaseModel):
    name: str = Field(description="Table name field")
    columns: List[Column] = Field(description="List of random Column objects")


class Schema(BaseModel):
    schema_name: str = Field(description="Schema name")
    tables: List[Table] = Field(description="List of random Table objects")


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_complete_and_acomplete() -> None:
    """Test both sync and async complete methods."""
    llm = Gemini(
        model="models/gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
    )

    prompt = "Write a poem about a magic backpack"

    # Test synchronous complete
    sync_response = llm.complete(prompt)
    assert sync_response is not None
    assert len(sync_response.text) > 0

    # Test async complete
    async_response = asyncio.run(llm.acomplete(prompt))
    assert async_response is not None
    assert len(async_response.text) > 0


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_chat_and_achat() -> None:
    """Test both sync and async chat methods."""
    llm = Gemini(
        model="models/gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
    )

    message = ChatMessage(content="Write a poem about a magic backpack")

    # Test synchronous chat
    sync_response = llm.chat(messages=[message])
    assert sync_response is not None
    assert sync_response.message.content and len(sync_response.message.content) > 0

    # Test async chat
    async_response = asyncio.run(llm.achat(messages=[message]))
    assert async_response is not None
    assert async_response.message.content and len(async_response.message.content) > 0


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_stream_chat_and_astream_chat() -> None:
    """Test both sync and async stream chat methods."""
    llm = Gemini(
        model="models/gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
    )

    message = ChatMessage(content="Write a poem about a magic backpack")

    # Test synchronous stream chat
    sync_chunks = list(llm.stream_chat(messages=[message]))
    assert len(sync_chunks) > 0
    assert all(isinstance(chunk.message.content, str) for chunk in sync_chunks)

    # Test async stream chat
    async def test_async_stream() -> List[ChatResponse]:
        chunks = []
        async for chunk in await llm.astream_chat(messages=[message]):
            chunks.append(chunk)
        return chunks

    async_chunks = asyncio.run(test_async_stream())
    assert len(async_chunks) > 0
    assert all(isinstance(chunk.message.content, str) for chunk in async_chunks)


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_stream_complete_and_astream_complete() -> None:
    """Test both sync and async stream complete methods."""
    llm = Gemini(
        model="models/gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
    )

    prompt = "Write a poem about a magic backpack"

    # Test synchronous stream complete
    sync_chunks = list(llm.stream_complete(prompt))
    assert len(sync_chunks) > 0
    assert all(isinstance(chunk.text, str) for chunk in sync_chunks)

    # Test async stream complete
    async def test_async_stream() -> List[CompletionResponse]:
        chunks = []
        async for chunk in await llm.astream_complete(prompt):
            chunks.append(chunk)
        return chunks

    async_chunks = asyncio.run(test_async_stream())
    assert len(async_chunks) > 0
    assert all(isinstance(chunk.text, str) for chunk in async_chunks)


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_simple_astructured_predict() -> None:
    """Test async structured prediction with a simple schema."""
    llm = Gemini(
        model="models/gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
    )

    response = asyncio.run(
        llm.astructured_predict(
            output_cls=Poem,
            prompt=PromptTemplate("Write a poem about a magic backpack"),
        )
    )

    assert response is not None
    assert isinstance(response, Poem)
    assert isinstance(response.content, str)
    assert len(response.content) > 0


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_simple_stream_structured_predict() -> None:
    """Test stream structured prediction with a simple schema."""
    llm = Gemini(
        model="models/gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
    )

    response = llm.stream_structured_predict(
        output_cls=Poem,
        prompt=PromptTemplate("Write a poem about a magic backpack"),
    )

    result = None
    for partial_response in response:
        assert partial_response.content is not None
        result = partial_response

    assert result is not None
    assert isinstance(result, Poem)
    assert len(result.content) > 0


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_simple_astream_structured_predict() -> None:
    """Test async stream structured prediction with a simple schema."""
    llm = Gemini(
        model="models/gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
    )

    response = llm.astream_structured_predict(
        output_cls=Poem,
        prompt=PromptTemplate("Write a poem about a magic backpack"),
    )

    async def run() -> None:
        result = None
        async for partial_response in await response:
            result = partial_response
            assert partial_response.content is not None

        assert result is not None
        assert isinstance(result, Poem)
        assert isinstance(result.content, str)

    asyncio.run(run())


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_simple_structured_predict() -> None:
    """Test structured prediction with a simple schema."""
    llm = Gemini(
        model="models/gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
    )

    response = llm.structured_predict(
        output_cls=Poem,
        prompt=PromptTemplate("Write a poem about a magic backpack"),
    )

    assert response is not None
    assert isinstance(response, Poem)
    assert isinstance(response.content, str)
    assert len(response.content) > 0


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_complex_structured_predict() -> None:
    """Test structured prediction with a complex nested schema."""
    llm = Gemini(
        model="models/gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
    )

    prompt = PromptTemplate("Generate a simple database structure")
    response = llm.structured_predict(output_cls=Schema, prompt=prompt)

    assert response is not None
    assert isinstance(response, Schema)
    assert isinstance(response.schema_name, str)
    assert len(response.schema_name) > 0
    assert len(response.tables) > 0
    assert all(isinstance(table, Table) for table in response.tables)
    assert all(len(table.columns) > 0 for table in response.tables)


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_as_structured_llm() -> None:
    llm = Gemini(
        model="models/gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
    )

    prompt = PromptTemplate("Generate content")

    # Test with simple schema
    poem_response = llm.as_structured_llm(output_cls=Poem, prompt=prompt).complete(
        "Write a poem about a magic backpack"
    )
    assert isinstance(poem_response.raw, Poem)
    assert len(poem_response.raw.content) > 0

    # Test with complex schema
    schema_response = llm.as_structured_llm(output_cls=Schema, prompt=prompt).complete(
        "Generate a simple database structure"
    )
    assert isinstance(schema_response.raw, Schema)
    assert len(schema_response.raw.schema_name) > 0
    assert len(schema_response.raw.tables) > 0


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_as_structured_llm_async() -> None:
    llm = Gemini(
        model="models/gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
    )

    prompt = PromptTemplate("Generate content")

    # Test with simple schema
    poem_response = asyncio.run(
        llm.as_structured_llm(output_cls=Poem, prompt=prompt).acomplete(
            "Write a poem about a magic backpack"
        )
    )
    assert isinstance(poem_response.raw, Poem)
    assert len(poem_response.raw.content) > 0

    # Test with complex schema
    schema_response = asyncio.run(
        llm.as_structured_llm(output_cls=Schema, prompt=prompt).acomplete(
            "Generate a simple database structure"
        )
    )
    assert isinstance(schema_response.raw, Schema)
    assert len(schema_response.raw.schema_name) > 0
    assert len(schema_response.raw.tables) > 0


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_as_structure_llm_with_config() -> None:
    llm = Gemini(
        model="models/gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
    )

    response = (
        llm.as_structured_llm(output_cls=Poem)
        .complete(
            prompt="Write a poem about a magic backpack",
            # here we want to change the temperature, but it must not override the whole config
            config={"temperature": 0.1},
        )
        .raw
    )

    assert isinstance(response, Poem)

    response = asyncio.run(
        llm.as_structured_llm(output_cls=Poem).acomplete(
            prompt="Write a poem about a magic backpack",
            # here we want to change the temperature, but it must not override the whole config
            config={"temperature": 0.1},
        )
    ).raw

    assert isinstance(response, Poem)


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_structured_predict_multiple_block() -> None:
    chat_messaages = [
        ChatMessage(
            content=[
                TextBlock(text="which logo is this?"),
                ImageBlock(
                    url="https://upload.wikimedia.org/wikipedia/commons/7/7a/Nohat-wiki-logo.png"
                ),
            ],
            role=MessageRole.USER,
        ),
    ]

    class Response(BaseModel):
        answer: str

    llm = Gemini(
        model="models/gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
    )
    support = llm.structured_predict(
        output_cls=Response, prompt=ChatPromptTemplate(message_templates=chat_messaages)
    )
    assert isinstance(support, Response)
    assert "wiki" in support.answer.lower()


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_get_tool_calls_from_response() -> None:
    def add(a: int, b: int) -> int:
        """Add two integers and returns the result integer."""
        return a + b

    llm = Gemini(
        model="models/gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
    )

    add_tool = FunctionTool.from_defaults(fn=add)
    msg = ChatMessage("What is the result of adding 2 and 3?")
    response = llm.chat_with_tools(
        user_msg=msg,
        tools=[add_tool],
    )

    tool_calls: List[ToolSelection] = llm.get_tool_calls_from_response(response)
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "add"
    assert tool_calls[0].tool_kwargs == {"a": 2, "b": 3}


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_convert_llama_index_schema_to_gemini_function_declaration() -> None:
    """Test conversion of a llama_index schema to a gemini function declaration."""
    llm = Gemini(
        model="models/gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
    )
    function_tool = get_function_tool(Poem)
    # this is our baseline, which is not working because:
    # 1. the descriptions are missing
    # 2. the required fields are not set
    google_openai_function = types.FunctionDeclaration.from_callable(
        client=llm._client,
        callable=function_tool.metadata.fn_schema,  # type: ignore
    )

    assert google_openai_function.description is None
    assert google_openai_function.parameters.required is None  # type: ignore

    # this is our custom conversion that can take a llama index: fn_schema and convert it to a gemini compatible
    # function declaration (subset of OpenAPI v3)
    converted = convert_schema_to_function_declaration(function_tool)

    assert converted.name == "Poem"
    assert converted.description is not None
    assert converted.parameters.required is not None  # type: ignore

    assert converted.parameters


def test_convert_llama_index_schema_to_gemini_function_declaration_nested_case() -> (
    None
):
    """Test conversion of a llama_index fn_schema to a gemini function declaration."""
    function_tool = get_function_tool(Schema)

    llama_index_model_json_schema = function_tool.metadata.fn_schema.model_json_schema()  # type: ignore
    # check that the model_json_schema contains a $defs key, which is not supported by Gemini
    assert "$defs" in llama_index_model_json_schema

    converted = convert_schema_to_function_declaration(function_tool)

    assert converted.name == "Schema"
    assert converted.description is not None
    assert converted.parameters.required is not None  # type: ignore

    assert converted.parameters
    assert list(converted.parameters.properties) == [
        "schema_name",
        "tables",
    ]

    assert converted.parameters.required == ["schema_name", "tables"]
