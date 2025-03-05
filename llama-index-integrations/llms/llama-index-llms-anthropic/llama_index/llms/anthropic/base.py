import json
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from llama_index.core.base.llms.generic_utils import (
    achat_to_completion_decorator,
    astream_chat_to_completion_decorator,
    chat_to_completion_decorator,
    stream_chat_to_completion_decorator,
)
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
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_TEMPERATURE
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM, ToolSelection
from llama_index.core.llms.utils import parse_partial_json
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.core.utils import Tokenizer
from llama_index.llms.anthropic.utils import (
    anthropic_modelname_to_contextsize,
    force_single_tool_call,
    is_function_calling_model,
    messages_to_anthropic_messages,
)

import anthropic
from anthropic.types import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    TextBlock,
    TextDelta,
)
from anthropic.types.tool_use_block import ToolUseBlock

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool


DEFAULT_ANTHROPIC_MODEL = "claude-2.1"
DEFAULT_ANTHROPIC_MAX_TOKENS = 512


class AnthropicTokenizer:
    def __init__(self, client, model) -> None:
        self._client = client
        self.model = model

    def encode(self, text: str, *args: Any, **kwargs: Any) -> List[int]:
        count = self._client.beta.messages.count_tokens(
            messages=[{"role": "user", "content": text}],
            model=self.model,
        ).input_tokens
        return [1] * count


class Anthropic(FunctionCallingLLM):
    """
    Anthropic LLM.

    Examples:
        `pip install llama-index-llms-anthropic`

        ```python
        from llama_index.llms.anthropic import Anthropic

        llm = Anthropic(model="claude-instant-1")
        resp = llm.stream_complete("Paul Graham is ")
        for r in resp:
            print(r.delta, end="")
        ```
    """

    model: str = Field(
        default=DEFAULT_ANTHROPIC_MODEL, description="The anthropic model to use."
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        ge=0.0,
        le=1.0,
    )
    max_tokens: int = Field(
        default=DEFAULT_ANTHROPIC_MAX_TOKENS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )

    base_url: Optional[str] = Field(default=None, description="The base URL to use.")
    timeout: Optional[float] = Field(
        default=None, description="The timeout to use in seconds.", ge=0
    )
    max_retries: int = Field(
        default=10, description="The maximum number of API retries.", ge=0
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the anthropic API."
    )

    _client: Union[
        anthropic.Anthropic, anthropic.AnthropicVertex, anthropic.AnthropicBedrock
    ] = PrivateAttr()
    _aclient: Union[
        anthropic.AsyncAnthropic,
        anthropic.AsyncAnthropicVertex,
        anthropic.AsyncAnthropicBedrock,
    ] = PrivateAttr()

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_ANTHROPIC_MAX_TOKENS,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: int = 10,
        api_key: Optional[str] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        default_headers: Optional[Dict[str, str]] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
        region: Optional[str] = None,
        project_id: Optional[str] = None,
        aws_region: Optional[str] = None,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            model=model,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

        if region and project_id and not aws_region:
            self._client = anthropic.AnthropicVertex(
                region=region,
                project_id=project_id,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
            )

            self._aclient = anthropic.AsyncAnthropicVertex(
                region=region,
                project_id=project_id,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
            )
        elif aws_region:
            # Note: this assumes you have AWS credentials configured.
            self._client = anthropic.AnthropicBedrock(
                aws_region=aws_region,
            )
            self._aclient = anthropic.AsyncAnthropicBedrock(
                aws_region=aws_region,
            )
        else:
            self._client = anthropic.Anthropic(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
            )
            self._aclient = anthropic.AsyncAnthropic(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
            )

    @classmethod
    def class_name(cls) -> str:
        return "Anthropic_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=anthropic_modelname_to_contextsize(self.model),
            num_output=self.max_tokens,
            is_chat_model=True,
            model_name=self.model,
            is_function_calling_model=is_function_calling_model(self.model),
        )

    @property
    def tokenizer(self) -> Tokenizer:
        return AnthropicTokenizer(self._client, self.model)

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        return {
            **base_kwargs,
            **self.additional_kwargs,
        }

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            **self._model_kwargs,
            **kwargs,
        }

    def _get_content_and_tool_calls(
        self, response: Any
    ) -> Tuple[str, List[ToolUseBlock]]:
        tool_calls = []
        content = ""
        for content_block in response.content:
            if isinstance(content_block, TextBlock):
                content += content_block.text
            elif isinstance(content_block, ToolUseBlock):
                tool_calls.append(content_block.dict())

        return content, tool_calls

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        anthropic_messages, system_prompt = messages_to_anthropic_messages(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = self._client.messages.create(
            messages=anthropic_messages,
            stream=False,
            system=system_prompt,
            **all_kwargs,
        )

        content, tool_calls = self._get_content_and_tool_calls(response)

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                additional_kwargs={"tool_calls": tool_calls},
            ),
            raw=dict(response),
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        complete_fn = chat_to_completion_decorator(self.chat)
        return complete_fn(prompt, **kwargs)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        anthropic_messages, system_prompt = messages_to_anthropic_messages(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = self._client.messages.create(
            messages=anthropic_messages, system=system_prompt, stream=True, **all_kwargs
        )

        def gen() -> ChatResponseGen:
            content = ""
            cur_tool_calls: List[ToolUseBlock] = []
            cur_tool_call: Optional[ToolUseBlock] = None
            cur_tool_json: str = ""
            role = MessageRole.ASSISTANT
            for r in response:
                if isinstance(r, ContentBlockDeltaEvent):
                    if isinstance(r.delta, TextDelta):
                        content_delta = r.delta.text
                        content += content_delta
                    else:
                        if not isinstance(cur_tool_call, ToolUseBlock):
                            raise ValueError("Tool call not started")
                        content_delta = r.delta.partial_json
                        cur_tool_json += content_delta
                        try:
                            argument_dict = parse_partial_json(cur_tool_json)
                            cur_tool_call.input = argument_dict
                        except ValueError:
                            pass

                    if cur_tool_call is not None:
                        tool_calls_to_send = [*cur_tool_calls, cur_tool_call]
                    else:
                        tool_calls_to_send = cur_tool_calls
                    yield ChatResponse(
                        message=ChatMessage(
                            role=role,
                            content=content,
                            additional_kwargs={
                                "tool_calls": [t.dict() for t in tool_calls_to_send]
                            },
                        ),
                        delta=content_delta,
                        raw=r,
                    )
                elif isinstance(r, ContentBlockStartEvent):
                    if isinstance(r.content_block, ToolUseBlock):
                        cur_tool_call = r.content_block
                        cur_tool_json = ""
                elif isinstance(r, ContentBlockStopEvent):
                    if isinstance(cur_tool_call, ToolUseBlock):
                        cur_tool_calls.append(cur_tool_call)

        return gen()

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        stream_complete_fn = stream_chat_to_completion_decorator(self.stream_chat)
        return stream_complete_fn(prompt, **kwargs)

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        anthropic_messages, system_prompt = messages_to_anthropic_messages(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = await self._aclient.messages.create(
            messages=anthropic_messages,
            system=system_prompt,
            stream=False,
            **all_kwargs,
        )

        content, tool_calls = self._get_content_and_tool_calls(response)

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                additional_kwargs={"tool_calls": tool_calls},
            ),
            raw=dict(response),
        )

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        acomplete_fn = achat_to_completion_decorator(self.achat)
        return await acomplete_fn(prompt, **kwargs)

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        anthropic_messages, system_prompt = messages_to_anthropic_messages(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = await self._aclient.messages.create(
            messages=anthropic_messages, system=system_prompt, stream=True, **all_kwargs
        )

        async def gen() -> ChatResponseAsyncGen:
            content = ""
            cur_tool_calls: List[ToolUseBlock] = []
            cur_tool_call: Optional[ToolUseBlock] = None
            cur_tool_json: str = ""
            role = MessageRole.ASSISTANT
            async for r in response:
                if isinstance(r, ContentBlockDeltaEvent):
                    if isinstance(r.delta, TextDelta):
                        content_delta = r.delta.text
                        content += content_delta
                    else:
                        if not isinstance(cur_tool_call, ToolUseBlock):
                            raise ValueError("Tool call not started")
                        content_delta = r.delta.partial_json
                        cur_tool_json += content_delta
                        try:
                            argument_dict = parse_partial_json(cur_tool_json)
                            cur_tool_call.input = argument_dict
                        except ValueError:
                            pass

                    if cur_tool_call is not None:
                        tool_calls_to_send = [*cur_tool_calls, cur_tool_call]
                    else:
                        tool_calls_to_send = cur_tool_calls
                    yield ChatResponse(
                        message=ChatMessage(
                            role=role,
                            content=content,
                            additional_kwargs={
                                "tool_calls": [t.dict() for t in tool_calls_to_send]
                            },
                        ),
                        delta=content_delta,
                        raw=r,
                    )
                elif isinstance(r, ContentBlockStartEvent):
                    if isinstance(r.content_block, ToolUseBlock):
                        cur_tool_call = r.content_block
                        cur_tool_json = ""
                elif isinstance(r, ContentBlockStopEvent):
                    if isinstance(cur_tool_call, ToolUseBlock):
                        cur_tool_calls.append(cur_tool_call)

        return gen()

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        astream_complete_fn = astream_chat_to_completion_decorator(self.astream_chat)
        return await astream_complete_fn(prompt, **kwargs)

    def _prepare_chat_with_tools(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare the chat with tools."""
        chat_history = chat_history or []

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)
            chat_history.append(user_msg)

        tool_dicts = []
        if tools:
            for tool in tools:
                tool_dicts.append(
                    {
                        "name": tool.metadata.name,
                        "description": tool.metadata.description,
                        "input_schema": tool.metadata.get_parameters_dict(),
                    }
                )
            if "prompt-caching" in kwargs.get("extra_headers", {}).get(
                "anthropic-beta", ""
            ):
                tool_dicts[-1]["cache_control"] = {"type": "ephemeral"}

        return {"messages": chat_history, "tools": tool_dicts, **kwargs}

    def _validate_chat_with_tools_response(
        self,
        response: ChatResponse,
        tools: List["BaseTool"],
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        """Validate the response from chat_with_tools."""
        if not allow_parallel_tool_calls:
            force_single_tool_call(response)
        return response

    def get_tool_calls_from_response(
        self,
        response: "ChatResponse",
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
            if (
                "input" not in tool_call
                or "id" not in tool_call
                or "name" not in tool_call
            ):
                raise ValueError("Invalid tool call.")
            if tool_call["type"] != "tool_use":
                raise ValueError("Invalid tool type. Unsupported by Anthropic")
            argument_dict = (
                json.loads(tool_call["input"])
                if isinstance(tool_call["input"], str)
                else tool_call["input"]
            )

            tool_selections.append(
                ToolSelection(
                    tool_id=tool_call["id"],
                    tool_name=tool_call["name"],
                    tool_kwargs=argument_dict,
                )
            )

        return tool_selections
