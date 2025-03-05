from typing import Any, Optional

from llama_index.llms.openai_like import OpenAILike
from pydantic import Field
from typing import List
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.base.llms.types import (
    CompletionResponse,
    ChatResponse,
    ChatResponseGen,
    MessageRole,
    ChatMessage,
)


from contextual import ContextualAI


class Contextual(OpenAILike):
    """
    Generate a response using Contextual's Grounded Language Model (GLM), an LLM engineered specifically to prioritize faithfulness to in-context retrievals over parametric knowledge to reduce hallucinations in Retrieval-Augmented Generation.

    The total request cannot exceed 32,000 tokens. Email glm-feedback@contextual.ai with any feedback or questions.

    Examples:
        `pip install llama-index-llms-contextual`

        ```python
        from llama_index.llms.contextual import Contextual

        # Set up the Contextual class with the required model and API key
        llm = Contextual(model="contextual-clm", api_key="your_api_key")

        # Call the complete method with a query
        response = llm.complete("Explain the importance of low latency LLMs")

        print(response)
        ```
    """

    model: str = Field(
        description="The model to use. Currently only supports `v1`.", default="v1"
    )
    api_key: str = Field(description="The API key to use.", default=None)
    base_url: str = Field(
        description="The base URL to use.",
        default="https://api.contextual.ai/v1/generate",
    )
    avoid_commentary: bool = Field(
        description="Flag to indicate whether the model should avoid providing additional commentary in responses. Commentary is conversational in nature and does not contain verifiable claims; therefore, commentary is not strictly grounded in available context. However, commentary may provide useful context which improves the helpfulness of responses.",
        default=False,
    )
    client: Any = Field(default=None, description="Contextual AI Client")

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = None,
        avoid_commentary: bool = False,
        **openai_llm_kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            api_base=base_url,
            is_chat_model=openai_llm_kwargs.pop("is_chat_model", True),
            **openai_llm_kwargs,
        )

        try:
            self.client = ContextualAI(api_key=api_key, base_url=base_url)
        except Exception as e:
            raise ValueError(f"Error initializing ContextualAI client: {e}")

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "contextual-clm"

    # Synchronous Methods
    @llm_completion_callback()
    def complete(
        self, prompt: str, knowledge: Optional[List[str]] = None, **kwargs
    ) -> CompletionResponse:
        """
        Generate completion for the given prompt.

        Args:
            prompt (str): The input prompt to generate completion for.
            **kwargs: Additional keyword arguments for the API request.

        Returns:
            str: The generated text completion.
        """
        messages_list = [{"role": MessageRole.USER, "content": prompt}]
        response = self._generate(
            knowledge=knowledge,
            messages=messages_list,
            model=self.model,
            system_prompt=self.system_prompt,
            **kwargs,
        )
        return CompletionResponse(text=response)

    @llm_chat_callback()
    def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        """
        Generate a chat response for the given messages.
        """
        messages_list = [
            {"role": msg.role, "content": msg.blocks[0].text} for msg in messages
        ]
        response = self._generate(
            knowledge=kwargs.get("knowledge_base", None),
            messages=messages_list,
            model=self.model,
            system_prompt=self.system_prompt,
            **kwargs,
        )
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=response)
        )

    @llm_chat_callback()
    def stream_chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponseGen:
        """
        Generate a chat response for the given messages.
        """
        raise NotImplementedError("stream methods not implemented in Contextual")

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs) -> ChatResponseGen:
        """
        Generate a chat response for the given messages.
        """
        raise NotImplementedError("stream methods not implemented in Contextual")

    def _generate(
        self, knowledge, messages, system_prompt, **kwargs
    ) -> CompletionResponse:
        """
        Generate completion for the given prompt.
        """
        raw_message = self.client.generate.create(
            messages=messages,
            knowledge=knowledge or [],
            model=self.model,
            system_prompt=system_prompt,
            avoid_commentary=self.avoid_commentary,
            temperature=kwargs.get("temperature", 0.0),
            max_new_tokens=kwargs.get("max_tokens", 1024),
            top_p=kwargs.get("top_p", 1),
        )
        return raw_message.response
