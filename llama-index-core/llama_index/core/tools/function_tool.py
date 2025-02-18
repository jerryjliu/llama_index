import asyncio
import inspect
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional, Type

if TYPE_CHECKING:
    from llama_index.core.bridge.langchain import StructuredTool, Tool

from llama_index.core.async_utils import asyncio_run
from llama_index.core.bridge.pydantic import BaseModel, FieldInfo
from llama_index.core.tools.types import AsyncBaseTool, ToolMetadata, ToolOutput
from llama_index.core.tools.utils import create_schema_from_function
from llama_index.core.workflow.context import Context

AsyncCallable = Callable[..., Awaitable[Any]]


def sync_to_async(fn: Callable[..., Any]) -> AsyncCallable:
    """Sync to async."""

    async def _async_wrapped_fn(*args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    return _async_wrapped_fn


def async_to_sync(func_async: AsyncCallable) -> Callable:
    """Async from sync."""

    def _sync_wrapped_fn(*args: Any, **kwargs: Any) -> Any:
        return asyncio_run(func_async(*args, **kwargs))  # type: ignore[arg-type]

    return _sync_wrapped_fn



class FunctionTool(AsyncBaseTool):
    """Function Tool.

    A tool that takes in a function and optionally handles workflow context.
    """

    def __init__(
        self,
        fn: Optional[Callable[..., Any]] = None,
        metadata: Optional[ToolMetadata] = None,
        async_fn: Optional[AsyncCallable] = None,
    ) -> None:
        if fn is None and async_fn is None:
            raise ValueError("fn must be provided")

        # If async_fn is provided explicitly, use it. Otherwise check if fn is async
        if async_fn is not None:
            self._async_fn = async_fn
            self._fn = fn or async_to_sync(async_fn)
        else:
            assert fn is not None
            if inspect.iscoroutinefunction(fn):
                self._async_fn = fn
                self._fn = async_to_sync(fn)
            else:
                self._fn = fn
                self._async_fn = sync_to_async(fn)

        # Determine if function requires context by inspecting signature
        fn_to_inspect = fn or async_fn
        assert fn_to_inspect is not None
        sig = inspect.signature(fn_to_inspect)

        self.requires_context = any(
            param.annotation == Context for param in sig.parameters.values()
        )

        if metadata is None:
            raise ValueError("metadata must be provided")

        self._metadata = metadata

    @classmethod
    def from_defaults(
        cls,
        fn: Optional[Callable[..., Any]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        return_direct: bool = False,
        fn_schema: Optional[Type[BaseModel]] = None,
        async_fn: Optional[AsyncCallable] = None,
        tool_metadata: Optional[ToolMetadata] = None,
    ) -> "FunctionTool":
        if tool_metadata is None:
            tool_metadata = cls.tool_metadata_from_defaults(
                fn=fn or async_fn,
                name=name,
                description=description,
                return_direct=return_direct,
                fn_schema=fn_schema
            )
        assert (fn or async_fn) is not None, "fn must be provided"
        return cls(
            fn=fn,
            metadata=tool_metadata,
            async_fn=async_fn,
        )
    @classmethod
    def tool_metadata_from_defaults(
        cls,
        fn: Optional[Callable[..., Any] | AsyncCallable] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        return_direct: bool = False,
        fn_schema: Optional[Type[BaseModel]] = None
    ) -> ToolMetadata:
        """Creates tool metadata from default values and returns it.
        Args:
            fn: Function to parse. Optional if name, description, and fn_schema are provided.
            name: Name of the tool. Optional if fn is provided.
            description: Description of the tool. Optional if fn is provided.
            return_direct: Return tool output directly, ending agent loop.
            fn_schema: Schema of the function. Optional if fn is provided.
        """
        fn_to_parse = fn
        if fn_to_parse:
            name = name or fn_to_parse.__name__
            docstring = fn_to_parse.__doc__

            # Get function signature
            fn_sig = inspect.signature(fn_to_parse)

            # Remove ctx parameter from schema if present
            has_ctx = any(
                param.annotation == Context for param in fn_sig.parameters.values()
            )
            ctx_param_name = None
            if has_ctx:
                ctx_param_name = next(
                    param.name
                    for param in fn_sig.parameters.values()
                    if param.annotation == Context
                )
                fn_sig = fn_sig.replace(
                    parameters=[
                        param
                        for param in fn_sig.parameters.values()
                        if param.annotation != Context
                    ]
                )

            # Handle FieldInfo defaults
            fn_sig = fn_sig.replace(
                parameters=[
                    param.replace(default=inspect.Parameter.empty)
                    if isinstance(param.default, FieldInfo)
                    else param
                    for param in fn_sig.parameters.values()
                ]
            )

            description = description or f"{name}{fn_sig}\n{docstring}"
            if fn_schema is None:
                fn_schema = create_schema_from_function(
                    f"{name}",
                    fn_to_parse,
                    additional_fields=None,
                    ignore_fields=[ctx_param_name]
                    if ctx_param_name is not None
                    else None,
                )
        else:
            assert name is not None, "name or fn must be provided"
            assert description is not None, "description or fn must be provided"
            assert fn_schema is not None, "fn_schema or fn must be provided"
        return ToolMetadata(
            name=name,
            description=description,
            fn_schema=fn_schema,
            return_direct=return_direct,
        )
    @property
    def metadata(self) -> ToolMetadata:
        """Metadata."""
        return self._metadata

    @property
    def fn(self) -> Callable[..., Any]:
        """Function."""
        return self._fn

    @property
    def async_fn(self) -> AsyncCallable:
        """Async function."""
        return self._async_fn

    def call(
        self, *args: Any, ctx: Optional[Context] = None, **kwargs: Any
    ) -> ToolOutput:
        """Call."""
        if self.requires_context:
            if ctx is None:
                raise ValueError("Context is required for this tool")
            tool_output = self._fn(ctx, *args, **kwargs)
        else:
            tool_output = self._fn(*args, **kwargs)

        return ToolOutput(
            content=str(tool_output),
            tool_name=self.metadata.name,
            raw_input={"args": args, "kwargs": kwargs},
            raw_output=tool_output,
        )

    async def acall(
        self, *args: Any, ctx: Optional[Context] = None, **kwargs: Any
    ) -> ToolOutput:
        """Call."""
        if self.requires_context:
            if ctx is None:
                raise ValueError("Context is required for this tool")
            tool_output = await self._async_fn(ctx, *args, **kwargs)
        else:
            tool_output = await self._async_fn(*args, **kwargs)

        return ToolOutput(
            content=str(tool_output),
            tool_name=self.metadata.name,
            raw_input={"args": args, "kwargs": kwargs},
            raw_output=tool_output,
        )

    def to_langchain_tool(
        self,
        **langchain_tool_kwargs: Any,
    ) -> "Tool":
        """To langchain tool."""
        from llama_index.core.bridge.langchain import Tool

        langchain_tool_kwargs = self._process_langchain_tool_kwargs(
            langchain_tool_kwargs
        )
        return Tool.from_function(
            func=self.fn,
            coroutine=self.async_fn,
            **langchain_tool_kwargs,
        )

    def to_langchain_structured_tool(
        self,
        **langchain_tool_kwargs: Any,
    ) -> "StructuredTool":
        """To langchain structured tool."""
        from llama_index.core.bridge.langchain import StructuredTool

        langchain_tool_kwargs = self._process_langchain_tool_kwargs(
            langchain_tool_kwargs
        )
        return StructuredTool.from_function(
            func=self.fn,
            coroutine=self.async_fn,
            **langchain_tool_kwargs,
        )
