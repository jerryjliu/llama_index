import asyncio
from functools import partial
from contextlib import contextmanager
from contextvars import Context, ContextVar, Token, copy_context
from typing import Any, Callable, Generator, List, Optional, Dict, Protocol
import inspect
import logging
import uuid
from deprecated import deprecated
from llama_index.core.bridge.pydantic import BaseModel, Field, ConfigDict
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.span import active_span_id
from llama_index.core.instrumentation.span_handlers import (
    BaseSpanHandler,
    NullSpanHandler,
)
from llama_index.core.instrumentation.events.base import BaseEvent
from llama_index.core.instrumentation.events.span import SpanDropEvent
import wrapt

DISPATCHER_SPAN_DECORATED_ATTR = "__dispatcher_span_decorated__"

_logger = logging.getLogger(__name__)

# ContextVar for managing active instrument tags
active_instrument_tags: ContextVar[Dict[str, Any]] = ContextVar(
    "instrument_tags", default={}
)


@contextmanager
def instrument_tags(new_tags: Dict[str, Any]) -> Generator[None, None, None]:
    token = active_instrument_tags.set(new_tags)
    try:
        yield
    finally:
        active_instrument_tags.reset(token)


# Keep for backwards compatibility
class EventDispatcher(Protocol):
    def __call__(self, event: BaseEvent, **kwargs: Any) -> None:
        ...


class Dispatcher(BaseModel):
    """Dispatcher class.

    Responsible for dispatching BaseEvent (and its subclasses) as well as
    sending signals to enter/exit/drop a BaseSpan. It does so by sending
    event and span signals to its attached BaseEventHandler as well as
    BaseSpanHandler.

    Concurrency:
        - Dispatcher is async-task and thread safe in the sense that
        spans of async coros will maintain its hieararchy or trace-trees and
        spans which emanate from various threads will also maintain its
        hierarchy.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = Field(default_factory=str, description="Name of dispatcher")
    event_handlers: List[BaseEventHandler] = Field(
        default=[], description="List of attached handlers"
    )
    span_handlers: List[BaseSpanHandler] = Field(
        default=[NullSpanHandler()], description="Span handler."
    )
    parent_name: str = Field(
        default_factory=str, description="Name of parent Dispatcher."
    )
    manager: Optional["Manager"] = Field(
        default=None, description="Dispatcher manager."
    )
    root_name: str = Field(default="root", description="Root of the Dispatcher tree.")
    propagate: bool = Field(
        default=True,
        description="Whether to propagate the event to parent dispatchers and their handlers",
    )
    current_span_ids: Optional[Dict[Any, str]] = Field(
        default_factory=dict,  # type: ignore
        description="Id of current enclosing span. Used for creating `dispatch_event` partials.",
    )

    def __init__(
        self,
        name: str = "",
        event_handlers: List[BaseEventHandler] = [],
        span_handlers: List[BaseSpanHandler] = [],
        parent_name: str = "",
        manager: Optional["Manager"] = None,
        root_name: str = "root",
        propagate: bool = True,
    ):
        super().__init__(
            name=name,
            event_handlers=event_handlers,
            span_handlers=span_handlers,
            parent_name=parent_name,
            manager=manager,
            root_name=root_name,
            propagate=propagate,
        )

    @property
    def parent(self) -> "Dispatcher":
        assert self.manager is not None
        return self.manager.dispatchers[self.parent_name]

    @property
    def root(self) -> "Dispatcher":
        assert self.manager is not None
        return self.manager.dispatchers[self.root_name]

    def add_event_handler(self, handler: BaseEventHandler) -> None:
        """Add handler to set of handlers."""
        self.event_handlers += [handler]

    def add_span_handler(self, handler: BaseSpanHandler) -> None:
        """Add handler to set of handlers."""
        self.span_handlers += [handler]

    def event(self, event: BaseEvent, **kwargs: Any) -> None:
        """Dispatch event to all registered handlers."""
        c: Optional["Dispatcher"] = self

        # Attach tags from the active context
        event.tags.update(active_instrument_tags.get())

        while c:
            for h in c.event_handlers:
                try:
                    h.handle(event, **kwargs)
                except BaseException:
                    pass
            if not c.propagate:
                c = None
            else:
                c = c.parent

    @deprecated(
        version="0.10.41",
        reason=(
            "`get_dispatch_event()` has been deprecated in favor of using `event()` directly."
            " If running into this warning through an integration package, then please "
            "update your integration to the latest version."
        ),
    )
    def get_dispatch_event(self) -> EventDispatcher:
        """Keep for backwards compatibility.

        In llama-index-core v0.10.41, we removed this method and made changes to
        integrations or packs that relied on this method. Adding back this method
        in case any integrations or apps have not been upgraded. That is, they
        still rely on this method.
        """
        return self.event

    def span_enter(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Send notice to handlers that a span with id_ has started."""
        c: Optional["Dispatcher"] = self
        while c:
            for h in c.span_handlers:
                try:
                    h.span_enter(
                        id_=id_,
                        bound_args=bound_args,
                        instance=instance,
                        parent_id=parent_id,
                        tags=tags,
                        **kwargs,
                    )
                except BaseException:
                    pass
            if not c.propagate:
                c = None
            else:
                c = c.parent

    def span_drop(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> None:
        """Send notice to handlers that a span with id_ is being dropped."""
        c: Optional["Dispatcher"] = self
        while c:
            for h in c.span_handlers:
                try:
                    h.span_drop(
                        id_=id_,
                        bound_args=bound_args,
                        instance=instance,
                        err=err,
                        **kwargs,
                    )
                except BaseException:
                    pass
            if not c.propagate:
                c = None
            else:
                c = c.parent

    def span_exit(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Send notice to handlers that a span with id_ is exiting."""
        c: Optional["Dispatcher"] = self
        while c:
            for h in c.span_handlers:
                try:
                    h.span_exit(
                        id_=id_,
                        bound_args=bound_args,
                        instance=instance,
                        result=result,
                        **kwargs,
                    )
                except BaseException:
                    pass
            if not c.propagate:
                c = None
            else:
                c = c.parent

    def span(self, func: Callable) -> Any:
        # The `span` decorator should be idempotent.
        try:
            if hasattr(func, DISPATCHER_SPAN_DECORATED_ATTR):
                return func
            setattr(func, DISPATCHER_SPAN_DECORATED_ATTR, True)
        except AttributeError:
            # instance methods can fail with:
            # AttributeError: 'method' object has no attribute '__dispatcher_span_decorated__'
            pass

        @wrapt.decorator
        def wrapper(func: Callable, instance: Any, args: list, kwargs: dict) -> Any:
            bound_args = inspect.signature(func).bind(*args, **kwargs)
            id_ = f"{func.__qualname__}-{uuid.uuid4()}"
            tags = active_instrument_tags.get()
            result = None

            # Copy the current context
            context = copy_context()

            token = active_span_id.set(id_)
            parent_id = None if token.old_value is Token.MISSING else token.old_value
            self.span_enter(
                id_=id_,
                bound_args=bound_args,
                instance=instance,
                parent_id=parent_id,
                tags=tags,
            )

            def handle_future_result(
                future: asyncio.Future,
                span_id: str,
                bound_args: inspect.BoundArguments,
                instance: Any,
                context: Context,
            ) -> None:
                from llama_index.core.workflow.errors import (
                    WorkflowCancelledByUser,
                    WorkflowTimeoutError,
                    WorkflowRuntimeError,
                )

                try:
                    # Get exception first to avoid calling result() on failed future
                    exception = future.exception()
                    if exception is not None:
                        # Handle expected workflow exceptions
                        if isinstance(
                            exception,
                            (
                                WorkflowCancelledByUser,
                                WorkflowTimeoutError,
                                WorkflowRuntimeError,
                            ),
                        ):
                            self.span_exit(
                                id_=span_id,
                                bound_args=bound_args,
                                instance=instance,
                                result=None,
                            )
                            # Don't re-raise, just return None to avoid callback exception
                            return None

                        # Handle unexpected exceptions
                        self.event(
                            SpanDropEvent(span_id=span_id, err_str=str(exception))
                        )
                        self.span_drop(
                            id_=span_id,
                            bound_args=bound_args,
                            instance=instance,
                            err=exception,
                        )
                        # Don't re-raise, just return None to avoid callback exception
                        return None

                    # Handle successful result
                    result = future.result()
                    self.span_exit(
                        id_=span_id,
                        bound_args=bound_args,
                        instance=instance,
                        result=result,
                    )
                    return result

                except asyncio.CancelledError:
                    # Handle cancellation cleanly
                    self.span_exit(
                        id_=span_id,
                        bound_args=bound_args,
                        instance=instance,
                        result=None,
                    )
                    # Don't re-raise, just return None to avoid callback exception
                    return None
                finally:
                    try:
                        context.run(active_span_id.reset, token)
                    except ValueError as e:
                        _logger.debug(f"Failed to reset active_span_id: {e}")

            try:
                result = func(*args, **kwargs)
                if isinstance(result, asyncio.Future):
                    # If the result is a Future, wrap it
                    new_future = asyncio.ensure_future(result)
                    new_future.add_done_callback(
                        partial(
                            handle_future_result,
                            span_id=id_,
                            bound_args=bound_args,
                            instance=instance,
                            context=context,
                        )
                    )
                    return new_future
                else:
                    # For non-Future results, proceed as before
                    self.span_exit(
                        id_=id_, bound_args=bound_args, instance=instance, result=result
                    )
                    return result
            except BaseException as e:
                self.event(SpanDropEvent(span_id=id_, err_str=str(e)))
                self.span_drop(id_=id_, bound_args=bound_args, instance=instance, err=e)
                raise
            finally:
                if not isinstance(result, asyncio.Future):
                    active_span_id.reset(token)

        @wrapt.decorator
        async def async_wrapper(
            func: Callable, instance: Any, args: list, kwargs: dict
        ) -> Any:
            bound_args = inspect.signature(func).bind(*args, **kwargs)
            id_ = f"{func.__qualname__}-{uuid.uuid4()}"
            tags = active_instrument_tags.get()

            token = active_span_id.set(id_)
            parent_id = None if token.old_value is Token.MISSING else token.old_value
            self.span_enter(
                id_=id_,
                bound_args=bound_args,
                instance=instance,
                parent_id=parent_id,
                tags=tags,
            )
            try:
                result = await func(*args, **kwargs)
            except BaseException as e:
                self.event(SpanDropEvent(span_id=id_, err_str=str(e)))
                self.span_drop(id_=id_, bound_args=bound_args, instance=instance, err=e)
                raise
            else:
                self.span_exit(
                    id_=id_, bound_args=bound_args, instance=instance, result=result
                )
                return result
            finally:
                # clean up
                active_span_id.reset(token)

        if inspect.iscoroutinefunction(func):
            return async_wrapper(func)
        else:
            return wrapper(func)

    @property
    def log_name(self) -> str:
        """Name to be used in logging."""
        if self.parent:
            return f"{self.parent.name}.{self.name}"
        else:
            return self.name


class Manager:
    def __init__(self, root: Dispatcher) -> None:
        self.dispatchers: Dict[str, Dispatcher] = {root.name: root}

    def add_dispatcher(self, d: Dispatcher) -> None:
        if d.name in self.dispatchers:
            pass
        else:
            self.dispatchers[d.name] = d


Dispatcher.model_rebuild()
