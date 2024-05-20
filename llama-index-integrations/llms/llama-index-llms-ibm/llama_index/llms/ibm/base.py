from typing import Any, Dict, Optional, Sequence, Union, Tuple

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import (
    Field,
    PrivateAttr,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.constants import (
    DEFAULT_TEMPERATURE,
)
from llama_index.core.base.llms.generic_utils import (
    completion_to_chat_decorator,
    stream_completion_to_chat_decorator,
)
from llama_index.core.llms.custom import CustomLLM
from llama_index.llms.ibm.utils import (
    retrive_attributes_from_model,
    resolve_watsonx_credentials,
)

DEFAULT_MAX_TOKENS = 20


class WatsonxLLM(CustomLLM):
    """
    IBM watsonx.ai large language models.

    Example:
        `pip install llama-index-llms-ibm`

        ```python

        from llama_index.llms.ibm import WatsonxLLM
        watsonx_llm = WatsonxLLM(
            model_id="google/flan-ul2",
            url="https://us-south.ml.cloud.ibm.com",
            apikey="*****",
            project_id="*****",
        )
        ```
    """

    model_id: Optional[str] = Field(
        default=None, description="Type of model to use.", allow_mutation=False
    )
    deployment_id: Optional[str] = Field(
        default=None, description="Id of deployed model to use.", allow_mutation=False
    )

    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        gte=0.0,
        lte=2.0,
    )
    max_new_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    additional_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional generation params for the watsonx.ai models.",
    )

    project_id: Optional[str] = Field(
        default=None,
        description="ID of the Watson Studio project.",
        allow_mutation=False,
    )

    space_id: Optional[str] = Field(
        default=None, description="ID of the Watson Studio space.", allow_mutation=False
    )

    url: Optional[str] = Field(
        default=None,
        description="Url to Watson Machine Learning or CPD instance",
        allow_mutation=False,
    )

    apikey: Optional[str] = Field(
        default=None,
        description="Apikey to Watson Machine Learning or CPD instance",
        allow_mutation=False,
    )

    token: Optional[str] = Field(
        default=None, description="Token to CPD instance", allow_mutation=False
    )

    password: Optional[str] = Field(
        default=None, description="Password to CPD instance", allow_mutation=False
    )

    username: Optional[str] = Field(
        default=None, description="Username to CPD instance", allow_mutation=False
    )

    instance_id: Optional[str] = Field(
        default=None, description="Instance_id of CPD instance", allow_mutation=False
    )

    version: Optional[str] = Field(
        default=None, description="Version of CPD instance", allow_mutation=False
    )

    verify: Union[str, bool, None] = Field(
        default=None,
        description="""
        User can pass as verify one of following:
        the path to a CA_BUNDLE file
        the path of directory with certificates of trusted CAs
        True - default path to truststore will be taken
        False - no verification will be made
        """,
        allow_mutation=False,
    )

    _model: ModelInference = PrivateAttr()
    _model_info: Optional[Dict[str, Any]] = PrivateAttr()
    _text_generation_params: Dict[str, Any] = PrivateAttr()

    def __init__(
        self,
        model_id: Optional[str] = None,
        deployment_id: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
        additional_params: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None,
        space_id: Optional[str] = None,
        url: Optional[str] = None,
        apikey: Optional[str] = None,
        token: Optional[str] = None,
        password: Optional[str] = None,
        username: Optional[str] = None,
        instance_id: Optional[str] = None,
        version: Optional[str] = None,
        verify: Union[str, bool, None] = None,
        watsonx_model: Optional[ModelInference] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize LLM and watsonx.ai ModelInference.
        """
        callback_manager = callback_manager or CallbackManager([])
        additional_params = additional_params or {}

        attrs = retrive_attributes_from_model(watsonx_model) or {}
        creds = (
            resolve_watsonx_credentials(
                url=url,
                apikey=apikey,
                token=token,
                username=username,
                password=password,
                instance_id=instance_id,
            )
            if not attrs
            else {}
        )

        super().__init__(
            model_id=(attrs.get("model_id") or model_id),
            deployment_id=(attrs.get("deployment_id") or deployment_id),
            temperature=(attrs.get("temperature") or temperature),
            max_new_tokens=(attrs.get("max_new_tokens") or max_new_tokens),
            additional_params=(attrs.get("additional_params") or additional_params),
            project_id=(attrs.get("project_id") or project_id),
            space_id=(attrs.get("space_id") or space_id),
            url=creds.get("url"),
            apikey=creds.get("apikey"),
            token=creds.get("token"),
            password=creds.get("password"),
            username=creds.get("username"),
            instance_id=creds.get("instance_id"),
            version=version,
            verify=verify,
            _model=watsonx_model,
            callback_manager=callback_manager,
            **kwargs,
        )

        self._text_generation_params, _ = self._split_generation_params(
            {
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,
                **additional_params,
            }
        )

        if watsonx_model is not None:
            self._model = watsonx_model
        else:
            self._model = ModelInference(
                model_id=model_id,
                deployment_id=deployment_id,
                credentials=Credentials.from_dict(
                    self._get_credential_kwargs(), _verify=self.verify
                ),
                params=self._text_generation_params,
                project_id=self.project_id,
                space_id=self.space_id,
            )
        self._model_info = None

    class Config:
        validate_assignment = True

    @property
    def model_info(self):
        if self._model_info is None:
            self._model_info = self._model.get_details()
        return self._model_info

    @classmethod
    def class_name(cls) -> str:
        """Get Class Name."""
        return "WatsonxLLM"

    def _get_credential_kwargs(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "apikey": self.apikey,
            "token": self.token,
            "password": self.password,
            "username": self.username,
            "instance_id": self.instance_id,
            "version": self.version,
        }

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=(
                (self.model_info or {})
                .get("model_limits", {})
                .get("max_sequence_length")
                if self.model_id
                else None
            ),
            num_output=self.max_new_tokens,
            model_name=self.model_id
            or self.model_info.get("entity", {}).get("base_model_id"),
        )

    @property
    def sample_generation_text_params(self) -> Dict[str, Any]:
        """Example of Model generation text kwargs that a user can pass to the model."""
        return GenTextParamsMetaNames().get_example_values()

    def _split_generation_params(
        self, data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        params = {}
        kwargs = {}
        sample_generation_kwargs_keys = set(self.sample_generation_text_params.keys())
        sample_generation_kwargs_keys.add("prompt_variables")
        for key, value in data.items():
            if key in sample_generation_kwargs_keys:
                params.update({key: value})
            else:
                kwargs.update({key: value})
        return params, kwargs

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        params, generation_kwargs = self._split_generation_params(kwargs)
        response = self._model.generate(
            prompt=prompt,
            params=self._text_generation_params | params,
            **generation_kwargs,
        )

        return CompletionResponse(
            text=self._model._return_guardrails_stats(response).get("generated_text"),
            raw=response,
        )

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        params, generation_kwargs = self._split_generation_params(kwargs)

        stream_response = self._model.generate_text_stream(
            prompt=prompt,
            params=self._text_generation_params | params,
            **generation_kwargs,
        )

        def gen() -> CompletionResponseGen:
            content = ""
            if kwargs.get("raw_response"):
                raw_stream_deltas: Dict[str, Any] = {"stream_deltas": []}
                for stream_delta in stream_response:
                    stream_delta_text = self._model._return_guardrails_stats(
                        stream_delta
                    ).get("generated_text", "")
                    content += stream_delta_text
                    raw_stream_deltas["stream_deltas"].append(stream_delta)
                    yield CompletionResponse(
                        text=content, delta=stream_delta_text, raw=raw_stream_deltas
                    )
            else:
                for stream_delta in stream_response:
                    content += stream_delta
                    yield CompletionResponse(text=content, delta=stream_delta)

        return gen()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        chat_fn = completion_to_chat_decorator(self.complete)

        return chat_fn(messages, **kwargs)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        chat_stream_fn = stream_completion_to_chat_decorator(self.stream_complete)

        return chat_stream_fn(messages, **kwargs)
