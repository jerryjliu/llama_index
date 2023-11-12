import json
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from deprecated import deprecated

from llama_index.bridge.langchain import StructuredTool, Tool
from llama_index.bridge.pydantic import BaseModel


class DefaultToolFnSchema(BaseModel):
    """Default tool function Schema."""

    input: str


@dataclass
class ToolMetadata:
    description: str
    name: Optional[str] = None
    fn_schema: Optional[Type[BaseModel]] = DefaultToolFnSchema

    def get_parameters_dict(self) -> dict:
        if self.fn_schema is None:
            parameters = {
                "type": "object",
                "properties": {
                    "input": {"title": "input query string", "type": "string"},
                },
                "required": ["input"],
            }
        else:
            parameters = self.fn_schema.schema()
            parameters = {
                k: v
                for k, v in parameters.items()
                if k in ["type", "properties", "required"]
            }
        return parameters

    @property
    def fn_schema_str(self) -> str:
        """Get fn schema as string."""
        if self.fn_schema is None:
            raise ValueError("fn_schema is None.")
        parameters = self.get_parameters_dict()
        return json.dumps(parameters)

    def get_name(self) -> str:
        """Get name."""
        if self.name is None:
            raise ValueError("name is None.")
        return self.name

    @deprecated(
        "Deprecated in favor of `to_openai_tool`, which should be used instead."
    )
    def to_openai_function(self) -> Dict[str, Any]:
        """Deprecated and replaced by `to_openai_tool`.
        The name and arguments of a function that should be called, as generated by the
        model.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_parameters_dict(),
        }

    def to_openai_tool(self) -> Dict[str, Any]:
        """To OpenAI tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_parameters_dict(),
            },
        }


class ToolOutput(BaseModel):
    """Tool output."""

    content: str
    tool_name: str
    raw_input: Dict[str, Any]
    raw_output: Any

    def __str__(self) -> str:
        """String."""
        return str(self.content)


class BaseTool:
    @property
    @abstractmethod
    def metadata(self) -> ToolMetadata:
        pass

    @abstractmethod
    def __call__(self, input: Any) -> ToolOutput:
        pass

    def _process_langchain_tool_kwargs(
        self,
        langchain_tool_kwargs: Any,
    ) -> Dict[str, Any]:
        """Process langchain tool kwargs."""
        if "name" not in langchain_tool_kwargs:
            langchain_tool_kwargs["name"] = self.metadata.name or ""
        if "description" not in langchain_tool_kwargs:
            langchain_tool_kwargs["description"] = self.metadata.description
        if "fn_schema" not in langchain_tool_kwargs:
            langchain_tool_kwargs["args_schema"] = self.metadata.fn_schema
        return langchain_tool_kwargs

    def to_langchain_tool(
        self,
        **langchain_tool_kwargs: Any,
    ) -> Tool:
        """To langchain tool."""
        langchain_tool_kwargs = self._process_langchain_tool_kwargs(
            langchain_tool_kwargs
        )
        return Tool.from_function(
            func=self.__call__,
            **langchain_tool_kwargs,
        )

    def to_langchain_structured_tool(
        self,
        **langchain_tool_kwargs: Any,
    ) -> StructuredTool:
        """To langchain structured tool."""
        langchain_tool_kwargs = self._process_langchain_tool_kwargs(
            langchain_tool_kwargs
        )
        return StructuredTool.from_function(
            func=self.__call__,
            **langchain_tool_kwargs,
        )


class AsyncBaseTool(BaseTool):
    """
    Base-level tool class that is backwards compatible with the old tool spec but also
    supports async.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> ToolOutput:
        return self.call(*args, **kwargs)

    @abstractmethod
    def call(self, input: Any) -> ToolOutput:
        """
        This is the method that should be implemented by the tool developer.
        """

    @abstractmethod
    async def acall(self, input: Any) -> ToolOutput:
        """
        This is the async version of the call method.
        Should also be implemented by the tool developer as an
        async-compatible implementation.
        """


class BaseToolAsyncAdapter(AsyncBaseTool):
    """
    Adapter class that allows a synchronous tool to be used as an async tool.
    """

    def __init__(self, tool: BaseTool):
        self.base_tool = tool

    @property
    def metadata(self) -> ToolMetadata:
        return self.base_tool.metadata

    def call(self, input: Any) -> ToolOutput:
        return self.base_tool(input)

    async def acall(self, input: Any) -> ToolOutput:
        return self.call(input)


def adapt_to_async_tool(tool: BaseTool) -> AsyncBaseTool:
    """
    Converts a synchronous tool to an async tool.
    """
    if isinstance(tool, AsyncBaseTool):
        return tool
    else:
        return BaseToolAsyncAdapter(tool)
