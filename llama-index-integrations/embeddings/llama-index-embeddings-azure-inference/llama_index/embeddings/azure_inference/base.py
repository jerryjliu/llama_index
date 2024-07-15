"""Azure AI model inference embeddings client."""

from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.base.llms.generic_utils import get_from_param_or_env

if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential

from azure.ai.inference import EmbeddingsClient
from azure.ai.inference.aio import EmbeddingsClient as EmbeddingsClientAsync
from azure.core.credentials import AzureKeyCredential


class AzureAIEmbeddingsModel(BaseEmbedding):
    """Azure AI model inference for embeddings.

    Examples:
        ```python
        from llama_index.core import Settings
        from llama_index.embeddings.azure_inference import AzureAIEmbeddingsModel

        llm = AzureAIEmbeddingsModel(
            endpoint="https://[your-endpoint].inference.ai.azure.com",
            credential="your-api-key",
        )

        # If using Microsoft Entra ID authentication, you can create the
        # client as follows
        #
        # from azure.identity import DefaultAzureCredential
        #
        # embed_model = AzureAIEmbeddingsModel(
        #     endpoint="https://[your-endpoint].inference.ai.azure.com",
        #     credential=DefaultAzureCredential()
        # )

        # Once the client is instantiated, you can set the context to use the model
        Settings.embed_model = embed_model

        documents = SimpleDirectoryReader("./data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        ```
    """

    model_extras: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs model parameters."
    )

    _client: EmbeddingsClient = PrivateAttr()
    _async_client: EmbeddingsClientAsync = PrivateAttr()

    def __init__(
        self,
        endpoint: str = None,
        credential: Union[str, AzureKeyCredential, "TokenCredential"] = None,
        model_name: str = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        num_workers: Optional[int] = None,
        client_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ):
        client_kwargs = client_kwargs or {}

        endpoint = get_from_param_or_env(
            "endpoint", endpoint, "AZURE_INFERENCE_ENDPOINT_URL", None
        )
        credential = get_from_param_or_env(
            "credential", credential, "AZURE_INFERENCE_ENDPOINT_CREDENTIAL", None
        )
        credential = (
            AzureKeyCredential(credential)
            if isinstance(credential, str)
            else credential
        )

        if not endpoint:
            raise ValueError(
                "You must provide an endpoint to use the Azure AI model inference LLM."
                "Pass the endpoint as a parameter or set the AZURE_INFERENCE_ENDPOINT_URL"
                "environment variable."
            )

        if not credential:
            raise ValueError(
                "You must provide an credential to use the Azure AI model inference LLM."
                "Pass the credential as a parameter or set the AZURE_INFERENCE_ENDPOINT_CREDENTIAL"
            )

        self._client = EmbeddingsClient(
            endpoint=endpoint,
            credential=credential,
            user_agent="llamaindex",
            **client_kwargs,
        )

        self._async_client = EmbeddingsClientAsync(
            endpoint=endpoint,
            credential=credential,
            user_agent="llamaindex",
            **client_kwargs,
        )

        super().__init__(
            model_name=model_name or "unknown",
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            num_workers=num_workers,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "AzureAIEmbeddingsModel"

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        additional_kwargs = {}
        if self.model_name and self.model_name != "unknown":
            additional_kwargs["model"] = self.model_name
        if self.model_extras:
            # pass any extra model parameter as model extra
            additional_kwargs["model_extras"] = self.model_extras

        return additional_kwargs

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._client.embed(input=[query], **self._model_kwargs).data[0].embedding

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return (
            (await self._async_client.embed(input=[query], **self._model_kwargs))
            .data[0]
            .embedding
        )

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._client.embed(input=[text], **self._model_kwargs).data[0].embedding

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        return (
            (await self._async_client.embed(input=[text], **self._model_kwargs))
            .data[0]
            .embedding
        )

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        embedding_response = self._client.embed(input=texts, **self._model_kwargs).data
        return [embed.embedding for embed in embedding_response]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        embedding_response = await self._async_client.embed(
            input=texts, **self._model_kwargs
        )
        return [embed.embedding for embed in embedding_response.data]
