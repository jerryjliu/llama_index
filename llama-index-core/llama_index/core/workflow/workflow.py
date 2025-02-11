import asyncio
import functools
import time
import uuid
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    AsyncGenerator,
    Set,
    Tuple,
)

from llama_index.core.instrumentation import get_dispatcher

from .decorators import StepConfig, step
from .context import Context
from .events import (
    InputRequiredEvent,
    HumanResponseEvent,
    Event,
    StartEvent,
    StopEvent,
)
from .errors import *
from .service import ServiceManager
from .utils import (
    get_steps_from_class,
    get_steps_from_instance,
    ServiceDefinition,
)
from .checkpointer import Checkpoint, CheckpointCallback
from .handler import WorkflowHandler
from .context_serializers import JsonSerializer, BaseSerializer


dispatcher = get_dispatcher(__name__)


class WorkflowMeta(type):
    def __init__(cls, name: str, bases: Tuple[type, ...], dct: Dict[str, Any]) -> None:
        super().__init__(name, bases, dct)
        cls._step_functions: Dict[str, Callable] = {}


class Workflow(metaclass=WorkflowMeta):
    """An event-driven abstraction used to orchestrate the execution of different components called "steps".

    Each step is responsible for handling certain event types and possibly emitting new events. Steps can be "bound"
    when they are defined as methods of the `Workflow` class itself, or "unbound" when they are defined as free
    functions. To define a step, the method or function must be decorated with the `@step` decorator.

    Workflows provide basic validation to catch potential runtime errors as soon as possible. Validation happens once,
    when the workflow starts, and does not produce much overhead. It can be disabled in any case.

    Use an instance of a `Workflow` class to run a workflow and stream events produced during execution. Workflows
    can be run step-by-step, by calling the `run_step` function multiple times until completion.
    """

    def __init__(
        self,
        timeout: Optional[float] = 10.0,
        disable_validation: bool = False,
        verbose: bool = False,
        service_manager: Optional[ServiceManager] = None,
        num_concurrent_runs: Optional[int] = None,
    ) -> None:
        """Create an instance of the workflow.

        Args:
            timeout:
                Number of seconds after the workflow execution will be halted, raising a `WorkflowTimeoutError`
                exception. If set to `None`, the timeout will be disabled.
            disable_validaton:
                Whether or not the workflow should be validated before running. In case the workflow is
                misconfigured, a call to `run` will raise a `WorkflowValidationError` exception explaining the details
                of the problem.
            verbose:
                Whether or not the workflow should print additional informative messages during execution.
            service_manager:
                The instance of the `ServiceManager` used to make nested workflows available to this
                workflow instance. The default value is the best choice unless you're customizing the workflow runtime.
            num_concurrent_runs:
                maximum number of .run() executions occurring simultaneously. If set to `None`, there
                is no limit to this number.
        """
        # Configuration
        self._timeout = timeout
        self._verbose = verbose
        self._disable_validation = disable_validation
        self._num_concurrent_runs = num_concurrent_runs
        self._sem = (
            asyncio.Semaphore(num_concurrent_runs) if num_concurrent_runs else None
        )
        # Broker machinery
        self._contexts: Set[Context] = set()
        self._stepwise_context: Optional[Context] = None
        # Services management
        self._service_manager = service_manager or ServiceManager()

    async def stream_events(self) -> AsyncGenerator[Event, None]:
        """Returns an async generator to consume any event that workflow steps decide to stream.

        To be able to use this generator, the usual pattern is to wrap the `run` call in a background task using
        `asyncio.create_task`, then enter a for loop like this:

            wf = StreamingWorkflow()
            r = asyncio.create_task(wf.run())

            async for ev in wf.stream_events():
                print(ev)

            await r
        """
        # In the typical streaming use case, `run()` is not awaited but wrapped in a asyncio.Task. Since we'll be
        # consuming events produced by `run()`, we must give its Task the chance to run before entering the dequeueing
        # loop.
        await asyncio.sleep(0)

        if len(self._contexts) > 1:
            # We can't possibly know from what session we should stream events, raise an error.
            msg = (
                "This workflow has multiple concurrent runs in progress and cannot stream events. "
                "To be able to stream events, make sure you call `run()` on this workflow only once."
            )
            raise WorkflowRuntimeError(msg)

        # Enter the dequeuing loop.
        ctx = next(iter(self._contexts))
        while True:
            ev = await ctx.streaming_queue.get()
            if type(ev) is StopEvent:
                break

            yield ev

        # remove context to free up room for the next stream_events call
        self._contexts.remove(ctx)

    @classmethod
    def add_step(cls, func: Callable) -> None:
        """Adds a free function as step for this workflow instance.

        It raises an exception if a step with the same name was already added to the workflow.
        """
        step_config: Optional[StepConfig] = getattr(func, "__step_config", None)
        if not step_config:
            msg = f"Step function {func.__name__} is missing the `@step` decorator."
            raise WorkflowValidationError(msg)

        if func.__name__ in {**get_steps_from_class(cls), **cls._step_functions}:
            msg = f"A step {func.__name__} is already part of this workflow, please choose another name."
            raise WorkflowValidationError(msg)

        cls._step_functions[func.__name__] = func

    def add_workflows(self, **workflows: "Workflow") -> None:
        """Adds one or more nested workflows to this workflow.

        This method only accepts keyword arguments, and the name of the parameter
        will be used as the name of the workflow.
        """
        for name, wf in workflows.items():
            self._service_manager.add(name, wf)

    def _get_steps(self) -> Dict[str, Callable]:
        """Returns all the steps, whether defined as methods or free functions."""
        return {**get_steps_from_instance(self), **self._step_functions}  # type: ignore[attr-defined]

    def _start(
        self,
        stepwise: bool = False,
        ctx: Optional[Context] = None,
        checkpoint_callback: Optional[CheckpointCallback] = None,
    ) -> Tuple[Context, str]:
        """Sets up the queues and tasks for each declared step.

        This method also launches each step as an async task.
        """
        run_id = str(uuid.uuid4())
        if ctx is None:
            ctx = Context(self, stepwise=stepwise)
            self._contexts.add(ctx)
        else:
            # clean up the context from the previous run
            ctx._tasks = set()
            ctx._retval = None
            ctx._step_event_holding = None
            ctx._cancel_flag.clear()

        for name, step_func in self._get_steps().items():
            if name not in ctx._queues:
                ctx._queues[name] = asyncio.Queue()

            if name not in ctx._step_flags:
                ctx._step_flags[name] = asyncio.Event()

            # At this point, step_func is guaranteed to have the `__step_config` attribute
            step_config: StepConfig = getattr(step_func, "__step_config")

            async def _task(
                name: str,
                queue: asyncio.Queue,
                step: Callable,
                config: StepConfig,
            ) -> None:
                while True:
                    ev = await queue.get()
                    if type(ev) not in config.accepted_events:
                        continue

                    # do we need to wait for the step flag?
                    if stepwise:
                        await ctx._step_flags[name].wait()

                        # clear all flags so that we only run one step
                        for flag in ctx._step_flags.values():
                            flag.clear()

                    if self._verbose and name != "_done":
                        print(f"Running step {name}")

                    # run step
                    kwargs: Dict[str, Any] = {}
                    if config.context_parameter:
                        kwargs[config.context_parameter] = ctx
                    for service_definition in config.requested_services:
                        service = self._service_manager.get(
                            service_definition.name, service_definition.default_value
                        )
                        kwargs[service_definition.name] = service
                    kwargs[config.event_name] = ev

                    # wrap the step with instrumentation
                    instrumented_step = dispatcher.span(step)

                    # - check if its async or not
                    # - if not async, run it in an executor
                    if asyncio.iscoroutinefunction(step):
                        retry_start_at = time.time()
                        attempts = 0
                        while True:
                            await ctx.mark_in_progress(name=name, ev=ev)
                            try:
                                new_ev = await instrumented_step(**kwargs)
                                break  # exit the retrying loop
                            except WorkflowDone:
                                await ctx.remove_from_in_progress(name=name, ev=ev)
                                raise
                            except Exception as e:
                                if config.retry_policy is None:
                                    raise WorkflowRuntimeError(
                                        f"Error in step '{name}': {e!s}"
                                    ) from e

                                delay = config.retry_policy.next(
                                    retry_start_at + time.time(), attempts, e
                                )
                                if delay is None:
                                    # We're done retrying
                                    raise WorkflowRuntimeError(
                                        f"Error in step '{name}': {e!s}"
                                    ) from e

                                attempts += 1
                                if self._verbose:
                                    print(
                                        f"Step {name} produced an error, retry in {delay} seconds"
                                    )
                                await asyncio.sleep(delay)

                    else:
                        try:
                            run_task = functools.partial(instrumented_step, **kwargs)
                            new_ev = await asyncio.get_event_loop().run_in_executor(
                                None, run_task
                            )
                        except WorkflowDone:
                            await ctx.remove_from_in_progress(name=name, ev=ev)
                            raise
                        except Exception as e:
                            raise WorkflowRuntimeError(
                                f"Error in step '{name}': {e!s}"
                            ) from e

                    if self._verbose and name != "_done":
                        if new_ev is not None:
                            print(f"Step {name} produced event {type(new_ev).__name__}")
                        else:
                            print(f"Step {name} produced no event")

                    # handle the return value
                    if new_ev is None:
                        await ctx.remove_from_in_progress(name=name, ev=ev)
                        continue

                    # Store the accepted event for the drawing operations
                    ctx._accepted_events.append((name, type(ev).__name__))

                    if not isinstance(new_ev, Event):
                        warnings.warn(
                            f"Step function {name} returned {type(new_ev).__name__} instead of an Event instance."
                        )
                    elif isinstance(new_ev, InputRequiredEvent):
                        ctx.write_event_to_stream(new_ev)
                    else:
                        if stepwise:
                            async with ctx._step_condition:
                                await ctx._step_condition.wait()
                                ctx._step_event_holding = new_ev
                                ctx._step_event_written.notify()  # shares same lock

                                await ctx.remove_from_in_progress(name=name, ev=ev)

                                # for stepwise Checkpoint after handler.run_step() call
                                if checkpoint_callback:
                                    await checkpoint_callback(
                                        run_id=run_id,
                                        ctx=ctx,
                                        last_completed_step=name,
                                        input_ev=ev,
                                        output_ev=new_ev,
                                    )
                        else:
                            # for regular execution, Checkpoint just before firing the next event
                            await ctx.remove_from_in_progress(name=name, ev=ev)
                            if checkpoint_callback:
                                await checkpoint_callback(
                                    run_id=run_id,
                                    ctx=ctx,
                                    last_completed_step=name,
                                    input_ev=ev,
                                    output_ev=new_ev,
                                )
                            ctx.send_event(new_ev)

            for _ in range(step_config.num_workers):
                ctx._tasks.add(
                    asyncio.create_task(
                        _task(name, ctx._queues[name], step_func, step_config),
                        name=name,
                    )
                )

            # add dedicated cancel task
            async def _cancel_workflow_task() -> None:
                await ctx._cancel_flag.wait()
                raise WorkflowCancelledByUser

            ctx._tasks.add(
                asyncio.create_task(
                    _cancel_workflow_task(), name="cancel_workflow_task"
                )
            )

        return ctx, run_id

    def send_event(self, message: Event, step: Optional[str] = None) -> None:
        msg = (
            "Use a Context instance to send events from a step. "
            "Make sure your step method or function takes a parameter of type Context like `ctx: Context` and "
            "replace `self.send_event(...)` with `ctx.send_event(...)` in your code."
        )

        if len(self._contexts) > 1:
            # We can't possibly know to what session we should send this event, raise an error.
            raise WorkflowRuntimeError(msg)

        # Emit a warning as this won't work for multiple run()s.
        warnings.warn(msg)
        ctx = next(iter(self._contexts))
        ctx.send_event(message=message, step=step)

    @dispatcher.span
    def run(
        self,
        ctx: Optional[Context] = None,
        stepwise: bool = False,
        checkpoint_callback: Optional[CheckpointCallback] = None,
        **kwargs: Any,
    ) -> WorkflowHandler:
        """Runs the workflow until completion."""
        # Validate the workflow if needed
        uses_hitl = self._validate()
        if uses_hitl and stepwise:
            raise WorkflowRuntimeError(
                "Human-in-the-loop is not supported with stepwise execution"
            )

        # Start the machinery in a new Context or use the provided one
        ctx, run_id = self._start(
            ctx=ctx, stepwise=stepwise, checkpoint_callback=checkpoint_callback
        )

        result = WorkflowHandler(ctx=ctx, run_id=run_id)

        async def _run_workflow() -> None:
            if self._sem:
                await self._sem.acquire()
            try:
                if not ctx.is_running:
                    # Send the first event
                    ctx.send_event(StartEvent(**kwargs))

                    # the context is now running
                    ctx.is_running = True
                else:
                    # resend in-progress events if already running
                    for name, evs in ctx._in_progress.items():
                        for ev in evs:
                            ctx.send_event(ev, step=name)

                done, unfinished = await asyncio.wait(
                    ctx._tasks,
                    timeout=self._timeout,
                    return_when=asyncio.FIRST_EXCEPTION,
                )

                # Handle task completion
                exception_raised = None
                for task in done:
                    try:
                        e = task.exception()
                        if isinstance(e, WorkflowDone):
                            result.set_result(ctx._retval)
                            return
                        elif e is not None:
                            exception_raised = e
                            break
                    except asyncio.CancelledError:
                        pass

                # Clean up unfinished tasks
                for t in unfinished:
                    t.cancel()

                # wait for cancelled tasks to cleanup
                await asyncio.gather(*unfinished, return_exceptions=True)

                # Handle errors
                ctx.is_running = False
                ctx.write_event_to_stream(StopEvent())

                if exception_raised:
                    if isinstance(
                        exception_raised,
                        (
                            WorkflowTimeoutError,
                            WorkflowRuntimeError,
                            WorkflowCancelledByUser,
                        ),
                    ):
                        result.set_exception(exception_raised)
                    else:
                        result.set_exception(
                            WorkflowRuntimeError(
                                f"Unexpected error: {exception_raised!s}"
                            )
                        )
                else:
                    result.set_exception(
                        WorkflowTimeoutError(
                            f"Operation timed out after {self._timeout} seconds"
                        )
                    )

            except Exception as e:
                if not result.cancelled():
                    result.set_exception(WorkflowRuntimeError(f"Workflow error: {e!s}"))
            finally:
                if self._sem:
                    self._sem.release()

        asyncio.create_task(_run_workflow())
        return result

    @dispatcher.span
    def run_from(
        self,
        checkpoint: Checkpoint,
        ctx_serializer: Optional[BaseSerializer] = None,
        checkpoint_callback: Optional[CheckpointCallback] = None,
        **kwargs: Any,
    ) -> WorkflowHandler:
        """Run from a specified Checkpoint.

        The `Context` snapshot contained in the checkpoint is loaded and used
        to execute the `Workflow`.
        """
        # load the `Context` from the checkpoint
        ctx_serializer = ctx_serializer or JsonSerializer()
        ctx = Context.from_dict(self, checkpoint.ctx_state, serializer=ctx_serializer)
        handler: WorkflowHandler = self.run(
            ctx=ctx, checkpoint_callback=checkpoint_callback, **kwargs
        )

        # only kick off the workflow if there are no in-progress events
        # in-progress events are already started in self.run()
        num_in_progress = sum(len(v) for v in ctx._in_progress.values())
        if num_in_progress == 0 and handler.ctx is not None:
            handler.ctx.send_event(checkpoint.output_event)

        return handler

    def is_done(self) -> bool:
        """Checks if the workflow is done."""
        return self._stepwise_context is None

    @step
    async def _done(self, ctx: Context, ev: StopEvent) -> None:
        """Tears down the whole workflow and stop execution."""
        ctx._retval = ev.result
        ctx.write_event_to_stream(ev)

        # Signal we want to stop the workflow
        raise WorkflowDone

    def _validate(self) -> bool:
        """Validate the workflow to ensure it's well-formed.

        Returns True if the workflow uses human-in-the-loop, False otherwise.
        """
        if self._disable_validation:
            return False

        produced_events: Set[type] = {StartEvent}
        consumed_events: Set[type] = set()
        requested_services: Set[ServiceDefinition] = set()

        for step_func in self._get_steps().values():
            step_config: Optional[StepConfig] = getattr(step_func, "__step_config")
            # At this point we know step config is not None, let's make the checker happy
            assert step_config is not None

            for event_type in step_config.accepted_events:
                consumed_events.add(event_type)

            for event_type in step_config.return_types:
                if event_type == type(None):
                    # some events may not trigger other events
                    continue

                produced_events.add(event_type)

            requested_services.update(step_config.requested_services)

        # Check if all consumed events are produced (except specific built-in events)
        unconsumed_events = consumed_events - produced_events
        unconsumed_events = {
            x
            for x in unconsumed_events
            if not issubclass(x, (InputRequiredEvent, HumanResponseEvent))
        }
        if unconsumed_events:
            names = ", ".join(ev.__name__ for ev in unconsumed_events)
            raise WorkflowValidationError(
                f"The following events are consumed but never produced: {names}"
            )

        # Check if there are any unused produced events (except specific built-in events)
        unused_events = produced_events - consumed_events
        unused_events = {
            x
            for x in unused_events
            if not issubclass(x, (InputRequiredEvent, HumanResponseEvent))
        }
        if unused_events:
            names = ", ".join(ev.__name__ for ev in unused_events)
            raise WorkflowValidationError(
                f"The following events are produced but never consumed: {names}"
            )

        # Check all the requested services are available
        required_service_names = {
            sd.name for sd in requested_services if sd.default_value is None
        }
        if required_service_names:
            avail_service_names = set(self._service_manager._services.keys())
            missing = required_service_names - avail_service_names
            if missing:
                msg = f"The following services are not available: {', '.join(str(m) for m in missing)}"
                raise WorkflowValidationError(msg)

        # Check if the workflow uses human-in-the-loop
        return (
            InputRequiredEvent in produced_events
            or HumanResponseEvent in consumed_events
        )
