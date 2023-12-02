import json
from typing import Any, Dict, List, Optional, Sequence

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.llms.base import (
    LLM,
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.llms.vllm_utils import get_response, post_http_request


class Vllm(LLM):
    model: Optional[str] = Field(description="The HuggingFace Model to use.")

    temperature: float = Field(description="The temperature to use for sampling.")

    tensor_parallel_size: Optional[int] = Field(
        default=1,
        description="The number of GPUs to use for distributed execution with tensor parallelism.",
    )

    trust_remote_code: Optional[bool] = Field(
        default=True,
        description="Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer.",
    )

    n: int = Field(
        default=1,
        description="Number of output sequences to return for the given prompt.",
    )

    best_of: Optional[int] = Field(
        default=None,
        description="Number of output sequences that are generated from the prompt.",
    )

    presence_penalty: float = Field(
        default=0.0,
        description="Float that penalizes new tokens based on whether they appear in the generated text so far.",
    )

    frequency_penalty: float = Field(
        default=0.0,
        description="Float that penalizes new tokens based on their frequency in the generated text so far.",
    )

    top_p: float = Field(
        default=1.0,
        description="Float that controls the cumulative probability of the top tokens to consider.",
    )

    top_k: int = Field(
        default=-1,
        description="Integer that controls the number of top tokens to consider.",
    )

    use_beam_search: bool = Field(
        default=False, description="Whether to use beam search instead of sampling."
    )

    stop: Optional[List[str]] = Field(
        default=None,
        description="List of strings that stop the generation when they are generated.",
    )

    ignore_eos: bool = Field(
        default=False,
        description="Whether to ignore the EOS token and continue generating tokens after the EOS token is generated.",
    )

    max_new_tokens: int = Field(
        default=512,
        description="Maximum number of tokens to generate per output sequence.",
    )

    logprobs: Optional[int] = Field(
        default=None,
        description="Number of log probabilities to return per output token.",
    )

    dtype: str = Field(
        default="auto",
        description="The data type for the model weights and activations.",
    )

    download_dir: Optional[str] = Field(
        default=None,
        description="Directory to download and load the weights. (Default to the default cache dir of huggingface)",
    )

    vllm_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Holds any model parameters valid for `vllm.LLM` call not explicitly specified.",
    )

    api_url: str = Field(description="The api url for vllm server")

    _client: Any = PrivateAttr()

    def __init__(
        self,
        model: str = "facebook/opt-125m",
        temperature: float = 1.0,
        tensor_parallel_size: Optional[int] = 1,
        trust_remote_code: Optional[bool] = True,
        n: int = 1,
        best_of: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        use_beam_search: bool = False,
        stop: Optional[List[str]] = None,
        ignore_eos: bool = False,
        max_new_tokens: int = 512,
        logprobs: Optional[int] = None,
        dtype: str = "auto",
        download_dir: Optional[str] = None,
        vllm_kwargs: Dict[str, Any] = {},
        api_url: Optional[str] = "",
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        try:
            from vllm import LLM as VLLModel
        except ImportError:
            raise ImportError(
                "Could not import vllm python package. "
                "Please install it with `pip install vllm`."
            )
        if model == "":
            self._client = VLLModel(
                model=model,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
                download_dir=download_dir,
                **vllm_kwargs
            )
        else:
            self._client = None
        callback_manager = callback_manager or CallbackManager([])

        super().__init__(
            model=model,
            temperature=temperature,
            n=n,
            best_of=best_of,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            top_p=top_p,
            top_k=top_k,
            use_beam_search=use_beam_search,
            stop=stop,
            ignore_eos=ignore_eos,
            max_new_tokens=max_new_tokens,
            logprobs=logprobs,
            dtype=dtype,
            download_dir=download_dir,
            vllm_kwargs=vllm_kwargs,
            api_url=api_url,
        )

    @classmethod
    def class_name(cls) -> str:
        return "Vllm"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model)

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "n": self.n,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "use_beam_search": self.use_beam_search,
            "ignore_eos": self.ignore_eos,
            "stop": self.stop,
            "logprobs": self.logprobs,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "stop": self.stop,
        }
        return {**base_kwargs, **self.vllm_kwargs}

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            **self._model_kwargs,
            **kwargs,
        }

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        raise (ValueError("Not Implemented"))

    @llm_completion_callback()
    def complete(self, prompts: List[str], **kwargs: Any) -> List[CompletionResponse]:
        # return CompletionResponse(text="hello")
        # print("here")
        kwargs = kwargs if kwargs else {}
        params = {**self._model_kwargs, **kwargs}

        from vllm import SamplingParams

        responses = []
        # build sampling parameters
        sampling_params = SamplingParams(**params)
        outputs = self._client.generate(prompts, sampling_params)
        for output in outputs:
            # prompt = output.prompt
            responses.append(CompletionResponse(text=output.outputs[0].text))

        return responses

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        raise (ValueError("Not Implemented"))

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise (ValueError("Not Implemented"))

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        raise (ValueError("Not Implemented"))

    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        raise (ValueError("Not Implemented"))

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        raise (ValueError("Not Implemented"))

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise (ValueError("Not Implemented"))


class VllmServer(Vllm):
    def __init__(
        self,
        model: str = "facebook/opt-125m",
        api_url: str = "http://localhost:8000",
        temperature: float = 1.0,
        tensor_parallel_size: Optional[int] = 1,
        trust_remote_code: Optional[bool] = True,
        max_retries: int = 10,
        n: int = 1,
        best_of: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        use_beam_search: bool = False,
        stop: Optional[List[str]] = None,
        ignore_eos: bool = False,
        max_new_tokens: int = 512,
        logprobs: Optional[int] = None,
        dtype: str = "auto",
        download_dir: Optional[str] = None,
        vllm_kwargs: Dict[str, Any] = {},
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._client = None

        callback_manager = callback_manager or CallbackManager([])

        model = ""
        super().__init__(
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            n=n,
            best_of=best_of,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            top_p=top_p,
            top_k=top_k,
            use_beam_search=use_beam_search,
            stop=stop,
            ignore_eos=ignore_eos,
            max_new_tokens=max_new_tokens,
            logprobs=logprobs,
            dtype=dtype,
            download_dir=download_dir,
            vllm_kwargs=vllm_kwargs,
            api_url=api_url,
        )

    @classmethod
    def class_name(cls) -> str:
        return "VllmServer"

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> List[CompletionResponse]:
        # return CompletionResponse(text="hello")
        # print("here")
        kwargs = kwargs if kwargs else {}
        params = {**self._model_kwargs, **kwargs}

        from vllm import SamplingParams

        responses = []
        # build sampling parameters
        sampling_params = SamplingParams(**params).__dict__
        sampling_params["prompt"] = prompt
        response = post_http_request(self.api_url, sampling_params, stream=False)
        outputs = get_response(response)
        for output in outputs:
            # prompt = output.prompt
            responses.append(CompletionResponse(text=output))

        return responses

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        kwargs = kwargs if kwargs else {}
        params = {**self._model_kwargs, **kwargs}

        from vllm import SamplingParams

        # build sampling parameters
        sampling_params = SamplingParams(**params).__dict__
        sampling_params["prompt"] = prompt
        response = post_http_request(self.api_url, sampling_params, stream=True)

        def gen() -> CompletionResponseGen:
            for chunk in response.iter_lines(
                chunk_size=8192, decode_unicode=False, delimiter=b"\0"
            ):
                if chunk:
                    data = json.loads(chunk.decode("utf-8"))

                    yield CompletionResponse(text=data["text"][0])

        return gen()
