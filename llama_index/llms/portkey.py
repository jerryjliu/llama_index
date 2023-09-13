"""
    Portkey intergation with Llama_index for enchanced monitoring
"""
from typing import Any, Optional, Sequence, Union, List, TYPE_CHECKING, cast

from llama_index.llms.custom import CustomLLM
from llama_index.llms.base import (
    ChatMessage,
    LLMMetadata,
    ChatResponse,
    CompletionResponse,
    ChatResponseGen,
    llm_completion_callback,
    llm_chat_callback,
    CompletionResponseGen,
)
from llama_index.llms.portkey_utils import (
    is_chat_model,
    generate_llm_metadata,
    get_llm,
    IMPORT_ERROR_MESSAGE,
)
from llama_index.llms.generic_utils import (
    completion_to_chat_decorator,
    chat_to_completion_decorator,
    stream_completion_to_chat_decorator,
    stream_chat_to_completion_decorator,
)

from llama_index.bridge.pydantic import Field, PrivateAttr

if TYPE_CHECKING:
    from portkey import (
        LLMOptions,
        ModesLiteral,
        Modes,
        PortkeyResponse,
    )


class Portkey(CustomLLM):
    """_summary_

    Args:
        LLM (_type_): _description_
    """

    mode: Optional[Union["Modes", "ModesLiteral"]] = Field(
        description="The mode for using the Portkey integration"
    )

    model: Optional[str] = Field(default="gpt-3.5-turbo")
    llm: "LLMOptions" = Field(
        description="LLM parameter", default_factory=dict)

    llms: List["LLMOptions"] = Field(
        description="LLM parameters", default_factory=list)

    _client: Any = PrivateAttr()

    def __init__(
        self,
        *,
        mode: Union["Modes", "ModesLiteral"],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """
        Initialize a Portkey instance.

        Args:
            api_key (Optional[str]): The API key to authenticate with Portkey.
            mode (Optional[Modes]): The mode for using the Portkey integration
            (default: Modes.SINGLE).
            provider (Optional[ProviderTypes]): The LLM provider to be used for the
                Portkey integration.
                Eg: openai, anthropic etc.
                NOTE: Check the ProviderTypes to see the supported list
                of LLMs.
            model (str): The name of the language model to use
            (default: "gpt-3.5-turbo").
            model_api_key (Optional[str]): The api key of the provider being used.
                Eg: api key of openai.
            temperature (float): The temperature parameter for text generation
            (default: 0.1).
            max_tokens (Optional[int]): The maximum number of tokens in the generated
            text.
            max_retries (int): The maximum number of retries for failed requests
            (default: 5).
            trace_id (Optional[str]): A unique identifier for tracing requests.
            cache_status (Optional[CacheType]): The type of cache to use
            (default: "").
                If cache_status is set, then cache is automatically set to True
            cache (Optional[bool]): Whether to use caching (default: False).
            metadata (Optional[Dict[str, Any]]): Metadata associated with the
            request (default: {}).
            weight (Optional[float]): The weight of the LLM in the ensemble
            (default: 1.0).
            **kwargs (Any): Additional keyword arguments.

        Raises:
            ValueError: If neither 'llm' nor 'llms' are provided during
            Portkey initialization.
        """
        try:
            import portkey
        except ImportError as exc:
            raise ImportError(IMPORT_ERROR_MESSAGE) from exc

        super().__init__(
            base_url=base_url,
            api_key=api_key,
        )
        if api_key is not None:
            portkey.api_key = api_key

        if base_url is not None:
            portkey.base_url = base_url

        portkey.mode = mode

        self._client = portkey
        self.model = None
        self.mode = mode

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return generate_llm_metadata(self.llms[0])

    def add_llms(
        self, llm_params: Union["LLMOptions", List["LLMOptions"]]
    ) -> "Portkey":
        """
        Adds the specified LLM parameters to the list of LLMs. This may be used for
        fallbacks or load-balancing as specified in the mode.

        Args:
            llm_params (Union[LLMOptions, List[LLMOptions]]): A single LLM parameter \
            set or a list of LLM parameter sets. Each set should be an instance of \
            LLMOptions with
            the specified attributes.
                > provider: Optional[ProviderTypes]
                > model: str
                > temperature: float
                > max_tokens: Optional[int]
                > max_retries: int
                > trace_id: Optional[str]
                > cache_status: Optional[CacheType]
                > cache: Optional[bool]
                > metadata: Dict[str, Any]
                > weight: Optional[float]

            NOTE: User may choose to pass additional params as well.
        Returns:
            self
        """
        try:
            from portkey import LLMOptions
        except ImportError as exc:
            raise ImportError(IMPORT_ERROR_MESSAGE) from exc
        if isinstance(llm_params, LLMOptions):
            llm_params = [llm_params]
        self.llms.extend(llm_params)
        if self.model is None:
            self.model = self.llms[0].model
        return self

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Completion endpoint for LLM."""
        if self._is_chat_model:
            complete_fn = chat_to_completion_decorator(self._chat)
        else:
            complete_fn = self._complete
        return complete_fn(prompt, **kwargs)

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        if self._is_chat_model:
            chat_fn = self._chat
        else:
            chat_fn = completion_to_chat_decorator(self._complete)
        return chat_fn(messages, **kwargs)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Completion endpoint for LLM."""
        if self._is_chat_model:
            complete_fn = stream_chat_to_completion_decorator(
                self._stream_chat)
        else:
            complete_fn = self._stream_complete
        return complete_fn(prompt, **kwargs)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        if self._is_chat_model:
            stream_chat_fn = self._stream_chat
        else:
            stream_chat_fn = stream_completion_to_chat_decorator(
                self._stream_complete)
        return stream_chat_fn(messages, **kwargs)

    def _chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        try:
            from portkey import Message, Config
        except ImportError as exc:
            raise ImportError(IMPORT_ERROR_MESSAGE) from exc
        _messages = cast(
            List[Message],
            [{"role": i.role.value, "content": i.content} for i in messages],
        )
        config = Config(llms=self.llms)
        response = self._client.ChatCompletions.create(
            messages=_messages, config=config
        )
        self.llm = self._get_llm(response)

        message = response.choices[0].message
        return ChatResponse(message=message, raw=response)

    def _complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        try:
            from portkey import Config
        except ImportError as exc:
            raise ImportError(IMPORT_ERROR_MESSAGE) from exc

        config = Config(llms=self.llms)
        response = self._client.Completions.create(
            prompt=prompt, config=config)
        text = response.choices[0].text
        return CompletionResponse(text=text, raw=response)

    def _stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        try:
            from portkey import Message, Config
        except ImportError as exc:
            raise ImportError(IMPORT_ERROR_MESSAGE) from exc
        _messages = cast(
            List[Message],
            [{"role": i.role.value, "content": i.content} for i in messages],
        )
        config = Config(llms=self.llms)
        response = self._client.ChatCompletions.create(
            messages=_messages, config=config, stream=True, **kwargs
        )

        def gen() -> ChatResponseGen:
            content = ""
            function_call: Optional[dict] = {}
            for resp in response:
                if resp.choices is None:
                    continue
                delta = resp.choices[0].delta
                role = delta.get("role", "assistant")
                content_delta = delta.get("content", "") or ""
                content += content_delta

                function_call_delta = delta.get("function_call", None)
                if function_call_delta is not None:
                    if function_call is None:
                        function_call = function_call_delta
                        # ensure we do not add a blank function call
                        if (
                            function_call
                            and function_call.get("function_name", "") is None
                        ):
                            del function_call["function_name"]
                    else:
                        function_call["arguments"] += function_call_delta["arguments"]

                additional_kwargs = {}
                if function_call is not None:
                    additional_kwargs["function_call"] = function_call

                yield ChatResponse(
                    message=ChatMessage(
                        role=role,
                        content=content,
                        additional_kwargs=additional_kwargs,
                    ),
                    delta=content_delta,
                    raw=resp,
                )

        return gen()

    def _stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        try:
            from portkey import Config
        except ImportError as exc:
            raise ImportError(IMPORT_ERROR_MESSAGE) from exc

        config = Config(llms=self.llms)
        response = self._client.Completions.create(
            prompt=prompt, config=config, stream=True, **kwargs
        )

        def gen() -> CompletionResponseGen:
            text = ""
            for resp in response:
                delta = resp.choices[0].text or ""
                text += delta
                yield CompletionResponse(
                    delta=delta,
                    text=text,
                    raw=resp,
                )

        return gen()

    @property
    def _is_chat_model(self) -> bool:
        """Check if a given model is a chat-based language model.

        Returns:
            bool: True if the provided model is a chat-based language model,
            False otherwise.
        """
        return is_chat_model(self.model or "")

    def _get_llm(self, response: "PortkeyResponse") -> "LLMOptions":
        return get_llm(response, self.llms)
