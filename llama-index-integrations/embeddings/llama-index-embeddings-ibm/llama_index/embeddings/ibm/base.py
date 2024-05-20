from typing import Any, List, Optional, Union, Dict

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.embeddings.ibm.utils import (
    resolve_watsonx_credentials,
)

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models.embeddings import Embeddings

DEFAULT_EMBED_MODEL = "ibm/slate-125m-english-rtrvr"


class WatsonxEmbeddings(BaseEmbedding):
    """
    IBM watsonx.ai embeddings.

    Example:
        `pip install llama-index-embeddings-ibm`

        ```python

        from llama_index.embeddings.ibm import WatsonxEmbeddings
        watsonx_llm = WatsonxLLM(
            model_id="ibm/slate-125m-english-rtrvr",
            url="https://us-south.ml.cloud.ibm.com",
            apikey="*****",
            project_id="*****",
        )
        ```
    """

    model_id: str = Field(
        default=DEFAULT_EMBED_MODEL,
        description="""Type of model to use.""",
        allow_mutation=False,
    )

    truncate_input_tokens: Optional[int] = Field(
        default=None,
        description="""Represents the maximum number of input tokens accepted.
        If the text is truncated, then it truncates from the left.""",
    )
    additional_kwargs: Optional[Dict[str, Any]] = Field(
        default_factory=None,
        description="Additional kwargs for the embeddings generation.",
    )

    project_id: Optional[str] = Field(
        default=None,
        description="ID of the Watson Studio project.",
        allow_mutation=False,
    )

    space_id: Optional[str] = Field(
        default=None,
        description="""ID of the Watson Studio space.""",
        allow_mutation=False,
    )

    url: Optional[str] = Field(
        default=None,
        description="""Url to Watson Machine Learning or CPD instance""",
        allow_mutation=False,
    )

    apikey: Optional[str] = Field(
        default=None,
        description="""Apikey to Watson Machine Learning or CPD instance""",
        allow_mutation=False,
    )

    token: Optional[str] = Field(
        default=None, description="""Token to CPD instance""", allow_mutation=False
    )

    password: Optional[str] = Field(
        default=None, description="""Password to CPD instance""", allow_mutation=False
    )

    username: Optional[str] = Field(
        default=None, description="""Username to CPD instance""", allow_mutation=False
    )

    instance_id: Optional[str] = Field(
        default=None,
        description="""Instance_id of CPD instance""",
        allow_mutation=False,
    )

    version: Optional[str] = Field(
        default=None, description="""Version of CPD instance""", allow_mutation=False
    )

    verify: Union[str, bool, None] = Field(
        default=None,
        description="""User can pass as verify one of following:
        the path to a CA_BUNDLE file
        the path of directory with certificates of trusted CAs
        True - default path to truststore will be taken
        False - no verification will be made""",
        allow_mutation=False,
    )

    _embed_model: Embeddings = PrivateAttr()

    def __init__(
        self,
        model_id: str,
        truncate_input_tokens: Optional[None] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        additional_kwargs: Optional[Dict[str, Any]] = None,
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
        api_client: Optional[APIClient] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ):
        callback_manager = callback_manager or CallbackManager([])

        if isinstance(api_client, APIClient):
            project_id = api_client.default_project_id or project_id
            space_id = api_client.default_space_id or space_id
            creds = {}
        else:
            creds = resolve_watsonx_credentials(
                url=url,
                apikey=apikey,
                token=token,
                username=username,
                password=password,
                instance_id=instance_id,
            )

        super().__init__(
            model_id=model_id,
            truncate_input_tokens=truncate_input_tokens,
            additional_kwargs=additional_kwargs,
            project_id=project_id,
            space_id=space_id,
            url=creds.get("url"),
            apikey=creds.get("apikey"),
            token=creds.get("token"),
            password=creds.get("password"),
            username=creds.get("username"),
            instance_id=creds.get("instance_id"),
            version=version,
            verify=verify,
            callback_manager=callback_manager,
            embed_batch_size=embed_batch_size,
            **kwargs,
        )

        self._embed_model = Embeddings(
            model_id=model_id,
            params=self.params,
            credentials=Credentials.from_dict(creds) if creds else None,
            project_id=self.project_id,
            space_id=self.space_id,
            api_client=api_client,
        )

    class Config:
        validate_assignment = True

    @classmethod
    def class_name(cls) -> str:
        return "WatsonxEmbedding"

    @property
    def params(self):
        return {"truncate_input_tokens": self.truncate_input_tokens}

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._embed_model.embed_query(text=query, params=self.params)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._get_query_embedding(query=text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self._embed_model.embed_documents(texts=texts, params=self.params)

    ### Async methods
    # Asynchronous evaluation is not yet supported for watsonx.ai embeddings
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        return self._get_text_embeddings(texts)
