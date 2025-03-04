import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from huggingface_hub import (
    InferenceClient,
    AsyncInferenceClient,
    ChatCompletionInputMessage,
)
from llama_index.core.base.llms.generic_utils import (
    chat_to_completion_decorator,
    achat_to_completion_decorator,
    stream_chat_to_completion_decorator,
    astream_chat_to_completion_decorator,
    get_from_param_or_env,
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
from llama_index.core.constants import (
    DEFAULT_TEMPERATURE,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
)
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.tools.types import BaseTool
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from packaging import version

logger = logging.getLogger(__name__)


def to_tgi_messages(
    messages: Sequence[ChatMessage],
) -> List[ChatCompletionInputMessage]:
    return [
        ChatCompletionInputMessage(role=message.role, content=message.content)
        for message in messages
    ]


def force_single_tool_call(response: ChatResponse) -> None:
    tool_calls = response.message.additional_kwargs.get("tool_calls", [])
    if len(tool_calls) > 1:
        response.message.additional_kwargs["tool_calls"] = [tool_calls[0]]


def resolve_tool_choice(
    tools: Optional[List[dict]] = None, tool_choice: str = "none"
) -> Union[str, dict]:
    """Resolve tool choice.

    Check if tool_name exists in tools.
    Note that unlike in OpenAI specification, 'auto' will ALWAYS choose the tool for you.
    Set to 'none' explicitly if do not wish to use tool.
    """
    valid_tool_choices = ["none", "auto"] + [t["function"]["name"] for t in tools or []]

    if tool_choice not in valid_tool_choices:
        raise ValueError(
            f"{tool_choice} is not a valid tool_choice. Must be one of {valid_tool_choices}"
        )

    return tool_choice


class TextGenerationInference(FunctionCallingLLM):
    model_name: Optional[str] = Field(
        default=None,
        description="The name of the model served at the TGI endpoint",
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        ge=0.0,
        le=1.0,
    )
    max_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    token: Union[str, bool, None] = Field(
        default=None,
        description=(
            "Hugging Face token. Will default to the locally saved token. Pass "
            "token=False if you don’t want to send your token to the server."
        ),
    )
    timeout: float = Field(
        default=120, description="The timeout to use in seconds.", ge=0
    )
    max_retries: int = Field(
        default=5, description="The maximum number of API retries.", ge=0
    )
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "Additional headers to send to the server. By default only the"
            " authorization headers are sent. Values in this dictionary"
            " will override the default values."
        ),
    )
    cookies: Optional[Dict[str, str]] = Field(
        default=None, description="Additional cookies to send to the server."
    )
    seed: Optional[str] = Field(
        default=None, description="The random seed to use for sampling."
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the TGI API."
    )

    _sync_client: "InferenceClient" = PrivateAttr()
    _async_client: "AsyncInferenceClient" = PrivateAttr()

    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description=(
            LLMMetadata.model_fields["context_window"].description
            + " Maximum total tokens returned from TGI endpoint."
        ),
    )
    is_chat_model: bool = Field(
        default=True,
        description=(
            LLMMetadata.model_fields["is_chat_model"].description
            + " TGI makes use of chat templating,"
            " function call is available only for '/v1/chat/completions' route"
            " of TGI endpoint"
        ),
    )
    is_function_calling_model: bool = Field(
        default=False,
        description=(
            LLMMetadata.model_fields["is_function_calling_model"].description
            + " 'text-generation-inference' supports function call"
            " starting from v1.4.3"
        ),
    )

    def __init__(
        self,
        model_url,
        model_name: Optional[str] = None,
        cookies: Optional[dict] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_NUM_OUTPUTS,
        timeout: int = 120,
        max_retries: int = 5,
        seed: Optional[int] = None,
        token: Optional[str] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        token = get_from_param_or_env("token", token, "HF_TOKEN", "")

        headers = {}
        if token:
            headers.update({"Authorization": f"Bearer {token}"})

        self._sync_client = InferenceClient(
            base_url=model_url,
            headers=headers,
            cookies=cookies,
            timeout=timeout,
        )
        self._async_client = AsyncInferenceClient(
            base_url=model_url,
            headers=headers,
            cookies=cookies,
            timeout=timeout,
        )

        endpoint_info = self._sync_client.get_endpoint_info()

        context_window = endpoint_info.get("max_total_tokens") or DEFAULT_CONTEXT_WINDOW

        is_function_calling_model = True
        tgi_version = endpoint_info.get("version", None)
        if version.parse(tgi_version) < version.parse("2.0.1"):
            is_function_calling_model = False
            logger.warning(
                f"TGI client has no function call support: 'text-generation-inference' version "
                f"incompatible with function call: {tgi_version}. Function call support was added in v2.0.1"
            )

        super().__init__(
            context_window=context_window,
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs,
            timeout=timeout,
            max_retries=max_retries,
            seed=seed,
            model_name=model_name,
            is_function_calling_model=is_function_calling_model,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

    @classmethod
    def class_name(cls) -> str:
        return "TextGenerationInference"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            is_chat_model=True,
            model_name=self.model_name,
            random_seed=self.seed,
            is_function_calling_model=self.is_function_calling_model,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
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

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)
        messages = to_tgi_messages(messages)
        response = self._sync_client.chat_completion(messages=messages, **all_kwargs)
        tool_calls = response.choices[0].message.tool_calls

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response.choices[0].message.content,
                additional_kwargs=(
                    {"tool_calls": tool_calls} if tool_calls is not None else {}
                ),
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
        messages = to_tgi_messages(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)
        response = self._sync_client.chat_completion(messages=messages, **all_kwargs)

        def generator() -> ChatResponseGen:
            content = ""
            role = MessageRole.ASSISTANT
            for chunk in response:
                content_delta = chunk.choices[0].delta.content
                if content_delta is None:
                    continue
                content += content_delta
                yield ChatResponse(
                    message=ChatMessage(role=role, content=content),
                    delta=content_delta,
                    raw=chunk,
                )

        return generator()

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
        messages = to_tgi_messages(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)
        response = await self._async_client.chat_completion(
            messages=messages, **all_kwargs
        )
        tool_calls = response.choices[0].message.tool_calls

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response.choices[0].message.content,
                additional_kwargs=(
                    {"tool_calls": tool_calls} if tool_calls is not None else {}
                ),
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
        messages = to_tgi_messages(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)
        response = await self._async_client.chat_completion(
            messages=messages, stream=True, **all_kwargs
        )

        async def generator() -> ChatResponseAsyncGen:
            content = ""
            role = MessageRole.ASSISTANT
            async for chunk in response:
                content_delta = chunk.choices[0].delta.content
                if content_delta is None:
                    continue
                content += content_delta
                yield ChatResponse(
                    message=ChatMessage(role=role, content=content),
                    delta=content_delta,
                    raw=chunk,
                )

        return generator()

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
        tool_choice: str = "auto",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare the arguments needed to let the LLM chat with tools."""
        # use openai tool format
        tool_specs = [
            tool.metadata.to_openai_tool(skip_length_check=True) for tool in tools
        ]

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        return {
            "messages": messages,
            "tools": tool_specs or None,
            "tool_choice": resolve_tool_choice(tool_specs, tool_choice),
            **kwargs,
        }

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
        self, response: "ChatResponse", error_on_no_tool_call: bool = True, **kwargs
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
            # TODO Add typecheck with ToolCall from TGI once the client is updated
            if tool_call and (tc_type := tool_call["type"]) != "function":
                raise ValueError(
                    f"Invalid tool type: got {tc_type}, expect 'function'."
                )
            argument_dict = tool_call["function"]["parameters"]

            tool_selections.append(
                ToolSelection(
                    tool_id=tool_call["id"],
                    tool_name=tool_call["function"][
                        "name"
                    ],  # NOTE for now the tool_name is hardcoded 'tools' in TGI
                    tool_kwargs=argument_dict,
                )
            )

        return tool_selections
