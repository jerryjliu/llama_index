"""Google's hosted Gemini API."""

import os
import typing
import uuid
from google import auth
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Union,
)

import llama_index.core.instrumentation as instrument
from google import genai
from google.genai import types
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_TEMPERATURE
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.prompts import PromptTemplate
from llama_index.core.types import Model
from llama_index.llms.genai.utils import (
    chat_from_gemini_response,
    chat_message_to_gemini,
    completion_from_gemini_response,
    convert_schema_to_function_declaration,
    prepare_chat_params,
)
from pydantic import PrivateAttr

dispatcher = instrument.get_dispatcher(__name__)

GEMINI_MODELS = (
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-thinking",
    "models/gemini-2.0-flash-thinking-exp-01-21",
    "models/gemini-2.0-flash-lite",
    "models/gemini-2.0-flash-lite-001",
    "models/gemini-2.0-pro-exp-02-05",
    "models/gemini-1.5-flash",
    "models/gemini-1.5-flash-8b",
    "models/gemini-1.0-pro",
)

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool


class VertexAIConfig(typing.TypedDict):
    credentials: Optional[auth.credentials.Credentials] = None
    project: Optional[str] = None
    location: Optional[str] = None


class Gemini(FunctionCallingLLM):
    """
    Gemini LLM.

    Examples:
        `pip install llama-index-llms-genai`

        ```python
        from llama_index.llms.gemini import Gemini

        llm = Gemini(model="models/gemini-ultra", api_key="YOUR_API_KEY")
        resp = llm.complete("Write a poem about a magic backpack")
        print(resp)
        ```
    """

    model: str = Field(default=GEMINI_MODELS[0], description="The Gemini model to use.")
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use during generation.",
        ge=0.0,
        le=2.0,
    )
    generate_kwargs: dict = Field(
        default_factory=dict, description="Kwargs for generation."
    )
    _max_tokens: int = PrivateAttr()
    _client: genai.Client = PrivateAttr()
    _generation_config: types.GenerateContentConfigDict = PrivateAttr()
    _model_meta: types.Model = PrivateAttr()

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GEMINI_MODELS[0],
        vertexai: bool = False,
        temperature: float = DEFAULT_TEMPERATURE,
        vertexai_config: Optional[VertexAIConfig] = None,
        max_tokens: Optional[int] = None,
        http_options: Optional[types.HttpOptions] = None,
        debug_config: Optional[genai.client.DebugConfig] = None,
        generation_config: Optional[types.GenerateContentConfig] = None,
        callback_manager: Optional[CallbackManager] = None,
        is_function_call_model: bool = True,
        **generate_kwargs: Any,
    ):
        # API keys are optional. The API can be authorised via OAuth (detected
        # environmentally) or by the GOOGLE_API_KEY environment variable.
        config_params: Dict[str, Any] = {
            **(vertexai_config if vertexai_config else {}),
            "api_key": api_key or os.getenv("GOOGLE_API_KEY"),
            "vertexai": vertexai,
        }

        if http_options:
            config_params["http_options"] = http_options

        if debug_config:
            config_params["debug_config"] = debug_config

        client = genai.Client(**config_params)
        model_meta = client.models.get(model=model)
        if not max_tokens:
            max_tokens = model_meta.output_token_limit
        else:
            max_tokens = min(max_tokens, model_meta.output_token_limit)

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            generate_kwargs=generate_kwargs,
            callback_manager=callback_manager,
        )

        self.model = model
        self._client = client
        self._model_meta = model_meta
        self._is_function_call_model = is_function_call_model
        # store this as a dict and not as a pydantic model so we can more easily
        # merge it later
        self._generation_config = (
            generation_config.model_dump() if generation_config else {}
        )
        self._max_tokens = max_tokens

    @classmethod
    def class_name(cls) -> str:
        return "GenAI"

    @property
    def metadata(self) -> LLMMetadata:
        total_tokens = (self._model_meta.input_token_limit or 0) + self._max_tokens
        return LLMMetadata(
            context_window=total_tokens,
            num_output=self._max_tokens,
            model_name=self.model,
            is_chat_model=True,
            is_function_calling_model=self._is_function_call_model,
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        generation_config = {
            **(self._generation_config or {}),
            **kwargs.pop("generation_config", {}),
        }
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=generation_config,
            **kwargs,
        )
        return completion_from_gemini_response(response)

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        generation_config = {
            **(self._generation_config or {}),
            **kwargs.pop("generation_config", {}),
        }
        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=generation_config,
            **kwargs,
        )

        return completion_from_gemini_response(response)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        generation_config = {
            **(self._generation_config or {}),
            **kwargs.pop("generation_config", {}),
        }

        def gen():
            it = self._client.models.generate_content_stream(
                model=self.model,
                contents=prompt,
                config=generation_config,
            )
            for response in it:
                yield completion_from_gemini_response(response)

        return gen()

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        generation_config = {
            **(self._generation_config or {}),
            **kwargs.pop("generation_config", {}),
        }

        async def gen():
            it = self._client.aio.models.generate_content_stream(
                model=self.model,
                contents=prompt,
                config=generation_config,
            )
            async for response in await it:  # type: ignore
                yield completion_from_gemini_response(response)

        return gen()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        generation_config = {
            **(self._generation_config or {}),
            **kwargs.pop("generation_config", {}),
        }
        params = {**kwargs, "generation_config": generation_config}
        next_msg, chat_kwargs = prepare_chat_params(self.model, messages, **params)
        chat = self._client.chats.create(**chat_kwargs)
        response = chat.send_message(next_msg.parts)  # type: ignore
        return chat_from_gemini_response(response)

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        generation_config = {
            **(self._generation_config or {}),
            **kwargs.pop("generation_config", {}),
        }
        params = {**kwargs, "generation_config": generation_config}
        next_msg, chat_kwargs = prepare_chat_params(self.model, messages, **params)
        chat = self._client.aio.chats.create(**chat_kwargs)
        response = await chat.send_message(next_msg.parts)  # type: ignore
        return chat_from_gemini_response(response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        generation_config = {
            **(self._generation_config or {}),
            **kwargs.pop("generation_config", {}),
        }
        params = {**kwargs, "generation_config": generation_config}
        next_msg, chat_kwargs = prepare_chat_params(self.model, messages, **params)
        chat = self._client.chats.create(**chat_kwargs)
        response = chat.send_message_stream(next_msg.parts)  # type: ignore

        def gen() -> ChatResponseGen:
            content = ""
            existing_tool_calls = []
            for r in response:
                top_candidate = r.candidates[0]
                content_delta = top_candidate.content.parts[0].text
                if content_delta:
                    content += content_delta
                llama_resp = chat_from_gemini_response(r)
                existing_tool_calls.extend(
                    llama_resp.message.additional_kwargs.get("tool_calls", [])
                )
                llama_resp.delta = content_delta
                llama_resp.message.content = content
                llama_resp.message.additional_kwargs["tool_calls"] = existing_tool_calls
                yield llama_resp

        return gen()

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        generation_config = {
            **(self._generation_config or {}),
            **kwargs.pop("generation_config", {}),
        }
        params = {**kwargs, "generation_config": generation_config}
        next_msg, chat_kwargs = prepare_chat_params(self.model, messages, **params)
        chat = self._client.aio.chats.create(**chat_kwargs)

        async def gen() -> ChatResponseAsyncGen:
            content = ""
            existing_tool_calls = []
            async for r in await chat.send_message_stream(next_msg.parts):  # type: ignore
                if candidates := r.candidates:
                    top_candidate = candidates[0]
                    if response_content := top_candidate.content:
                        if parts := response_content.parts:
                            content_delta = parts[0].text
                            if content_delta:
                                content += content_delta
                            llama_resp = chat_from_gemini_response(r)
                            existing_tool_calls.extend(
                                llama_resp.message.additional_kwargs.get(
                                    "tool_calls", []
                                )
                            )
                            llama_resp.delta = content_delta
                            llama_resp.message.content = content
                            llama_resp.message.additional_kwargs[
                                "tool_calls"
                            ] = existing_tool_calls
                            yield llama_resp

        return gen()

    def _prepare_chat_with_tools(
        self,
        tools: Sequence["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_choice: Union[str, dict] = "auto",
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Predict and call the tool."""
        if tool_choice == "auto":
            tool_mode = types.FunctionCallingConfigMode.AUTO
        elif tool_choice == "none":
            tool_mode = types.FunctionCallingConfigMode.NONE
        else:
            tool_mode = types.FunctionCallingConfigMode.ANY

        function_calling_config = types.FunctionCallingConfig(mode=tool_mode)

        if tool_choice not in ["auto", "none"]:
            if isinstance(tool_choice, dict):
                raise ValueError("Gemini does not support tool_choice as a dict")

            # assume that the user wants a tool call to be made
            # if the tool choice is not in the list of tools, then we will make a tool call to all tools
            # otherwise, we will make a tool call to the tool choice
            tool_names = [tool.metadata.name for tool in tools if tool.metadata.name]
            if tool_choice not in tool_names:
                function_calling_config.allowed_function_names = tool_names
            else:
                function_calling_config.allowed_function_names = [tool_choice]

        tool_config = types.ToolConfig(
            function_calling_config=function_calling_config,
        )

        tool_declarations = []
        for tool in tools:
            if tool.metadata.fn_schema:
                function_declaration = convert_schema_to_function_declaration(tool)
                tool_declarations.append(function_declaration)

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        return {
            "messages": messages,
            "tools": (
                [types.Tool(function_declarations=tool_declarations)]
                if tool_declarations
                else None
            ),
            "tool_config": tool_config,
            **kwargs,
        }

    def get_tool_calls_from_response(
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> List[ToolSelection]:
        """Predict and call the tool."""
        tool_calls = response.message.additional_kwargs.get("tool_calls", [])

        if len(tool_calls) < 1:
            if error_on_no_tool_call:
                raise ValueError(
                    f"Expected at least one tool call, but got {len(tool_calls)} tool calls."
                )
            else:
                return []

        tool_selections = []
        for tool_call in tool_calls:
            tool_selections.append(
                ToolSelection(
                    tool_id=str(uuid.uuid4()),
                    tool_name=tool_call.name,
                    tool_kwargs=dict(tool_call.args),
                )
            )

        return tool_selections

    @dispatcher.span
    def structured_predict_without_function_calling(
        self,
        output_cls: type[BaseModel],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Structured predict."""
        llm_kwargs = llm_kwargs or {}
        all_kwargs = {**llm_kwargs, **kwargs}

        messages = prompt.format_messages()
        response = self._client.models.generate_content(
            model=self.model,
            contents=list(map(chat_message_to_gemini, messages)),
            **{
                **all_kwargs,
                **{
                    "config": {
                        "response_mime_type": "application/json",
                        "response_schema": output_cls,
                    }
                },
            },
        )

        if isinstance(response.parsed, BaseModel):
            return response.parsed
        else:
            raise ValueError("Response is not a BaseModel")

    @dispatcher.span
    def structured_predict(
        self,
        output_cls: type[BaseModel],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Structured predict."""
        llm_kwargs = llm_kwargs or {}
        all_kwargs = {**llm_kwargs, **kwargs}

        if self._is_function_call_model:
            llm_kwargs["tool_choice"] = (
                "required"
                if "tool_choice" not in all_kwargs
                else all_kwargs["tool_choice"]
            )

        return super().structured_predict(
            output_cls, prompt, llm_kwargs=llm_kwargs, **kwargs
        )

    @dispatcher.span
    async def astructured_predict(
        self,
        output_cls: type[BaseModel],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Structured predict."""
        llm_kwargs = llm_kwargs or {}
        all_kwargs = {**llm_kwargs, **kwargs}

        if self._is_function_call_model:
            llm_kwargs["tool_choice"] = (
                "required"
                if "tool_choice" not in all_kwargs
                else all_kwargs["tool_choice"]
            )

        return await super().astructured_predict(
            output_cls, prompt, llm_kwargs=llm_kwargs, **kwargs
        )

    @dispatcher.span
    def stream_structured_predict(
        self,
        output_cls: type[BaseModel],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Generator[Union[Model, List[Model]], None, None]:
        """Stream structured predict."""
        llm_kwargs = llm_kwargs or {}
        all_kwargs = {**llm_kwargs, **kwargs}

        if self._is_function_call_model:
            llm_kwargs["tool_choice"] = (
                "required"
                if "tool_choice" not in all_kwargs
                else all_kwargs["tool_choice"]
            )
        return super().stream_structured_predict(
            output_cls, prompt, llm_kwargs=llm_kwargs, **kwargs
        )

    @dispatcher.span
    async def astream_structured_predict(
        self,
        output_cls: type[BaseModel],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Generator[Union[Model, List[Model]], None, None]:
        """Stream structured predict."""
        llm_kwargs = llm_kwargs or {}
        all_kwargs = {**llm_kwargs, **kwargs}

        if self._is_function_call_model:
            llm_kwargs["tool_choice"] = (
                "required"
                if "tool_choice" not in all_kwargs
                else all_kwargs["tool_choice"]
            )
        return await super().astream_structured_predict(
            output_cls, prompt, llm_kwargs=llm_kwargs, **kwargs
        )
