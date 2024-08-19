"""NVIDIA Agent.

Simple wrapper around AgentRunner + OpenAIAgentWorker.
"""

from typing import (
    Any,
    Dict,
    List,
    Callable,
    Optional,
    Type,
)

from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.callbacks import CallbackManager
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.memory.types import BaseMemory
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.tools import BaseTool
from llama_index.llms.nvidia import NVIDIA
from llama_index.llms.openai.utils import OpenAIToolCall

# as an agent
from llama_index.agent.openai import OpenAIAgentWorker

DEFAULT_MAX_FUNCTION_CALLS = 5


class NVIDIAAgent(AgentRunner):
    """NVIDIA agent.

    Subclasses AgentRunner with a OpenAIAgentWorker.\
    ```

    """

    def __init__(
        self,
        tools: List[BaseTool],
        memory: BaseMemory,
        prefix_messages: List[ChatMessage],
        verbose: bool = False,
        llm: Optional[NVIDIA] = None,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        default_tool_choice: str = "auto",
        callback_manager: Optional[CallbackManager] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        tool_call_parser: Optional[Callable[[OpenAIToolCall], Dict]] = None,
    ) -> None:
        """Init params."""
        if not llm:
            llm = NVIDIA("meta/llama-3.1-70b-instruct", is_function_calling_model=True)
        callback_manager = callback_manager or llm.callback_manager
        step_engine = OpenAIAgentWorker.from_tools(
            tools=tools,
            tool_retriever=tool_retriever,
            llm=llm,
            verbose=verbose,
            max_function_calls=max_function_calls,
            callback_manager=callback_manager,
            prefix_messages=prefix_messages,
            tool_call_parser=tool_call_parser,
        )
        super().__init__(
            step_engine,
            memory=memory,
            llm=llm,
            callback_manager=callback_manager,
            default_tool_choice=default_tool_choice,
        )

    @classmethod
    def from_tools(
        cls,
        tools: Optional[List[BaseTool]] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        llm: Optional[NVIDIA] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        default_tool_choice: str = "auto",
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        prefix_messages: Optional[List[ChatMessage]] = None,
        tool_call_parser: Optional[Callable[[OpenAIToolCall], Dict]] = None,
        **kwargs: Any,
    ) -> "NVIDIAAgent":
        """Create an NVIDIAAgent from a list of tools.

        Similar to `from_defaults` in other classes, this method will
        infer defaults for a variety of parameters, including the LLM,
        if they are not specified.

        """
        tools = tools or []

        chat_history = chat_history or []
        if not llm:  # default model
            llm = NVIDIA("meta/llama-3.1-70b-instruct", is_function_calling_model=True)

        if callback_manager is not None:
            llm.callback_manager = callback_manager

        memory = memory or memory_cls.from_defaults(chat_history, llm=llm)

        # only llama3 endpoints support tool calling
        # TODO: update filtering to support future models
        if not llm.model.startswith("meta/llama-3.1-"):
            raise ValueError(
                f"Model name {llm.model} does not support function calling API. "
            )

        if system_prompt is not None:
            if prefix_messages is not None:
                raise ValueError(
                    "Cannot specify both system_prompt and prefix_messages"
                )
            prefix_messages = [ChatMessage(content=system_prompt, role="system")]

        prefix_messages = prefix_messages or []

        return cls(
            tools=tools,
            tool_retriever=tool_retriever,
            llm=llm,
            memory=memory,
            prefix_messages=prefix_messages,
            verbose=verbose,
            max_function_calls=max_function_calls,
            callback_manager=callback_manager,
            default_tool_choice=default_tool_choice,
            tool_call_parser=tool_call_parser,
        )
