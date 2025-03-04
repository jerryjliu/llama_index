from typing import Any, Dict, List, Optional, Sequence
import requests

from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    MessageRole,
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    ChatResponseAsyncGen,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import (
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from llama_index.core.multi_modal_llms import (
    MultiModalLLM,
    MultiModalLLMMetadata,
)
from llama_index.core.schema import ImageNode
from llama_index.core.base.llms.generic_utils import (
    get_from_param_or_env,
)

from llama_index.core.base.llms.generic_utils import get_from_param_or_env

from llama_index.multi_modal_llms.nvidia.utils import (
    BASE_URL,
    DEFAULT_MODEL,
    KNOWN_URLS,
    NVIDIA_MULTI_MODAL_MODELS,
    generate_nvidia_multi_modal_chat_message,
    aggregate_msgs,
    process_response,
)
import aiohttp
import json
import warnings

from llama_index.core.bridge.pydantic import BaseModel


class Model(BaseModel):
    id: str


class NVIDIAClient:
    def __init__(
        self,
        api_key: str,
        timeout: Optional[float] = None,
    ):
        self.api_key = api_key
        self.timeout = timeout

    def _get_headers(self, stream: bool) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
            "User-Agent": "langchain-nvidia-ai-endpoints",
        }
        headers["accept"] = "text/event-stream" if stream else "application/json"
        return headers

    def get_model_details(self) -> List[Model]:
        """
        Get model details.

        Returns:
            List of models
        """
        return [Model(id=model) for model in NVIDIA_MULTI_MODAL_MODELS]

    def request(
        self,
        endpoint: str,
        stream: bool,
        messages: Dict[str, Any],
        extra_headers: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Perform a synchronous request to the NVIDIA API.

        Args:
            endpoint (str): The API endpoint to send the request to.
            messages (Dict[str, Any]): The request payload.

        Returns:
            Dict[str, Any]: The API response.
        """

        def perform_request():
            payload = {"messages": messages, "stream": stream, **kwargs}
            headers = {
                **self._get_headers(stream=stream),
                **extra_headers,
            }
            response = requests.post(
                endpoint, json=payload, headers=headers, stream=stream
            )
            response.raise_for_status()
            return response

        return perform_request()

    async def request_async(
        self,
        endpoint: str,
        stream: bool,
        messages: Dict[str, Any],
        extra_headers: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Perform an asynchronous request to the NVIDIA API.

        Args:
            endpoint (str): The API endpoint to send the request to.
            messages (Dict[str, Any]): The request payload.

        Returns:
            Dict[str, Any]: The API response.
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint,
                json={"messages": messages, "stream": stream, **kwargs},
                headers={**self._get_headers(stream=stream), **extra_headers},
            ) as response:
                response.raise_for_status()
                return await response.json()


class NVIDIAMultiModal(MultiModalLLM):
    model: str = Field(description="The Multi-Modal model to use from NVIDIA.")
    temperature: float = Field(description="The temperature to use for sampling.")
    max_tokens: Optional[int] = Field(
        description=" The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.",
        gt=0,
    )
    timeout: float = Field(
        default=60.0,
        description="The timeout, in seconds, for API requests.",
        ge=0,
    )
    api_key: str = Field(default=None, description="The NVIDIA API key.", exclude=True)
    base_url: str = Field(default=BASE_URL, description="The base URL for NVIDIA API.")
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the NVIDIA API."
    )

    def __init__(
        self,
        model: str = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = 300,
        nvidia_api_key: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = BASE_URL,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        api_key = get_from_param_or_env(
            "api_key",
            nvidia_api_key or api_key,
            "NVIDIA_API_KEY",
            "NO_API_KEY_PROVIDED",
        )

        is_hosted = base_url in KNOWN_URLS

        if is_hosted and api_key == "NO_API_KEY_PROVIDED":
            raise ValueError(
                "An API key is required for the hosted NIM. This will become an error in 0.2.0."
            )

        ## set default model
        if not model:
            model = DEFAULT_MODEL
            warnings.warn(
                f"Default model is set as: {model}. \n"
                "Set model using model parameter. \n"
                "To get available models use available_models property.",
                UserWarning,
            )

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            api_base=base_url,
            callback_manager=callback_manager,
            **kwargs,
        )

    @property
    def _client(self) -> NVIDIAClient:
        return NVIDIAClient(**self._get_credential_kwargs())

    @classmethod
    def class_name(cls) -> str:
        return "nvidia_multi_modal_llm"

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        """Multi Modal LLM metadata."""
        return MultiModalLLMMetadata(
            num_output=self.max_tokens or DEFAULT_NUM_OUTPUTS,
            model_name=self.model,
        )

    @property
    def available_models(self) -> List[Model]:
        return self._client.get_model_details()

    def _get_credential_kwargs(self) -> Dict[str, Any]:
        return {"api_key": self.api_key}

    # Model Params for NVIDIA Multi Modal model.
    def _get_model_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        if self.model not in NVIDIA_MULTI_MODAL_MODELS:
            raise ValueError(
                f"Invalid model {self.model}. "
                f"Available models are: {list(NVIDIA_MULTI_MODAL_MODELS.keys())}"
            )
        base_kwargs = {"model": self.model, "temperature": self.temperature, **kwargs}
        if self.max_tokens is not None:
            base_kwargs["max_tokens"] = self.max_tokens
        return {**base_kwargs, **self.additional_kwargs}

    def _get_response_token_counts(self, raw_response: Any) -> dict:
        """Get the token usage reported by the response."""
        if not isinstance(raw_response, dict):
            return {}

        usage = raw_response.get("usage", {})
        # NOTE: other model providers that use the NVIDIA client may not report usage
        if usage is None:
            return {}

        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    def _complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        all_kwargs = self._get_model_kwargs(**kwargs)
        content, extra_headers = generate_nvidia_multi_modal_chat_message(
            prompt=prompt, image_documents=image_documents, model=self.model
        )
        message_dict = [{"role": MessageRole.USER, "content": content}]

        response = self._client.request(
            endpoint=NVIDIA_MULTI_MODAL_MODELS[self.model]["endpoint"],
            stream=False,
            messages=message_dict,
            extra_headers=extra_headers,
            **all_kwargs,
        )
        response = response.json()
        text = response["choices"][0]["message"]["content"]
        return CompletionResponse(
            text=text,
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
        )

    def _stream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponseGen:
        all_kwargs = self._get_model_kwargs(**kwargs)
        content, extra_headers = generate_nvidia_multi_modal_chat_message(
            prompt=prompt, image_documents=image_documents, model=self.model
        )
        message_dict = [{"role": MessageRole.USER, "content": content}]

        def gen() -> CompletionResponseGen:
            response = self._client.request(
                messages=message_dict,
                stream=True,
                endpoint=NVIDIA_MULTI_MODAL_MODELS[self.model]["endpoint"],
                extra_headers=extra_headers,
                **all_kwargs,
            )
            for line in response.iter_lines():
                if line and line.strip() != b"data: [DONE]":
                    line = line.decode("utf-8")
                    line = line[5:]

                    msg, final_line = aggregate_msgs(process_response(line))

                    yield CompletionResponse(
                        **msg,
                        additional_kwargs=self._get_response_token_counts(line),
                    )

                    if final_line:
                        break

        return gen()

    def complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        return self._complete(prompt, image_documents, **kwargs)

    def stream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponseGen:
        return self._stream_complete(prompt, image_documents, **kwargs)

    def _chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        all_kwargs = self._get_model_kwargs(**kwargs)
        content, extra_headers = generate_nvidia_multi_modal_chat_message(
            inputs=messages, model=self.model
        )

        response = self._client.request(
            endpoint=NVIDIA_MULTI_MODAL_MODELS[self.model]["endpoint"],
            stream=False,
            messages=content,
            extra_headers=extra_headers,
            **all_kwargs,
        )
        response = response.json()
        text = response["choices"][0]["message"]["content"]

        return ChatResponse(
            delta=text,
            message=ChatMessage(
                role=response["choices"][0]["message"]["role"], content=text
            ),
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
        )

    def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        return self._chat(messages, **kwargs)

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseGen:
        all_kwargs = self._get_model_kwargs(**kwargs)
        content, extra_headers = generate_nvidia_multi_modal_chat_message(
            inputs=messages, model=self.model
        )

        def gen() -> CompletionResponseGen:
            response = self._client.request(
                messages=content,
                stream=True,
                endpoint=NVIDIA_MULTI_MODAL_MODELS[self.model]["endpoint"],
                extra_headers=extra_headers,
                **all_kwargs,
            )
            for line in response.iter_lines():
                if line and line.strip() != b"data: [DONE]":
                    line = line.decode("utf-8")
                    line = line[5:]

                    msg, final_line = aggregate_msgs(process_response(line))

                    role = msg.get("role", MessageRole.ASSISTANT)
                    additional_kwargs = {}

                    yield ChatResponse(
                        message=ChatMessage(
                            role=role,
                            content=msg.get("content"),
                            additional_kwargs=additional_kwargs,
                        ),
                        delta=msg.get("content"),
                        raw=response,
                        additional_kwargs=self._get_response_token_counts(line),
                    )

                    if final_line:
                        break

        return gen()

    # ===== Async Endpoints =====

    async def _acomplete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        all_kwargs = self._get_model_kwargs(**kwargs)
        content, extra_headers = generate_nvidia_multi_modal_chat_message(
            prompt=prompt, image_documents=image_documents, model=self.model
        )
        message_dict = [{"role": MessageRole.USER, "content": content}]

        response_json = await self._client.request_async(
            endpoint=NVIDIA_MULTI_MODAL_MODELS[self.model]["endpoint"],
            stream=False,
            messages=message_dict,
            extra_headers=extra_headers,
            **all_kwargs,
        )
        text = response_json["choices"][0]["message"]["content"]
        return CompletionResponse(
            text=text,
            raw=response_json,
            additional_kwargs=self._get_response_token_counts(response_json),
        )

    async def acomplete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        return await self._acomplete(prompt, image_documents, **kwargs)

    async def astream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        all_kwargs = self._get_model_kwargs(**kwargs)
        content, extra_headers = generate_nvidia_multi_modal_chat_message(
            prompt=prompt, image_documents=image_documents, model=self.model
        )
        payload = {
            "messages": [{"role": MessageRole.USER, "content": content}],
            "stream": True,
            **all_kwargs,
        }
        headers = {
            **self._client._get_headers(stream=True),
            **extra_headers,
        }

        async def gen() -> CompletionResponseAsyncGen:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    NVIDIA_MULTI_MODAL_MODELS[self.model]["endpoint"],
                    json=payload,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    text = ""
                    async for line in response.content:
                        if line and line.strip() != b"data: [DONE]":
                            line = line.decode("utf-8").strip()
                            if line.startswith("data:"):
                                data = json.loads(line[5:])

                                delta = data["choices"][0]["delta"]["content"]
                                text += delta

                                yield CompletionResponse(
                                    text=text,
                                    raw=data,
                                    delta=text,
                                    additional_kwargs=self._get_response_token_counts(
                                        line
                                    ),
                                )

        return gen()

    async def _achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        all_kwargs = self._get_model_kwargs(**kwargs)
        content, extra_headers = generate_nvidia_multi_modal_chat_message(
            inputs=messages, model=self.model
        )

        response_json = await self._client.request_async(
            endpoint=NVIDIA_MULTI_MODAL_MODELS[self.model]["endpoint"],
            stream=False,
            messages=content,
            extra_headers=extra_headers,
            **all_kwargs,
        )

        text = response_json["choices"][0]["message"]["content"]

        return ChatResponse(
            delta=text,
            message=ChatMessage(
                role=response_json["choices"][0]["message"]["role"], content=text
            ),
            raw=response_json,
            additional_kwargs=self._get_response_token_counts(response_json),
        )

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        return await self._achat(messages, **kwargs)

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        all_kwargs = self._get_model_kwargs(**kwargs)
        content, extra_headers = generate_nvidia_multi_modal_chat_message(
            inputs=messages, model=self.model
        )
        payload = {"messages": content, "stream": True, **all_kwargs}
        headers = {
            **self._client._get_headers(stream=True),
            **extra_headers,
        }

        async def gen() -> ChatResponseAsyncGen:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    NVIDIA_MULTI_MODAL_MODELS[self.model]["endpoint"],
                    json=payload,
                    headers=headers,
                ) as response:
                    response.raise_for_status()

                    text = ""

                    async for line in response.content:
                        if line and line.strip() != b"data: [DONE]":
                            line_text = line.decode("utf-8").strip()

                            if line_text.startswith("data:"):
                                data = json.loads(line_text[5:])
                                delta = data["choices"][0]["delta"]["content"]
                                role = data["choices"][0]["delta"].get(
                                    "role", MessageRole.ASSISTANT
                                )
                                text += delta

                                yield ChatResponse(
                                    message=ChatMessage(
                                        role=role,
                                        content=delta,
                                        additional_kwargs={},
                                    ),
                                    delta=delta,
                                    raw=data,
                                    additional_kwargs=self._get_response_token_counts(
                                        data
                                    ),
                                )

        return gen()
