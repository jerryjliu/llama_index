import asyncio
import json
import uuid
import warnings
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    DefaultDict,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
)

from .context_serializers import BaseSerializer, JsonSerializer
from .decorators import StepConfig
from .errors import WorkflowRuntimeError
from .events import Event
from .types import RunResultT

if TYPE_CHECKING:  # pragma: no cover
    from .workflow import Workflow

T = TypeVar("T", bound=Event)


class Context:
    """A global object representing a context for a given workflow run.

    The Context object can be used to store data that needs to be available across iterations during a workflow
    execution, and across multiple workflow runs.
    Every context instance offers two type of data storage: a global one, that's shared among all the steps within a
    workflow, and private one, that's only accessible from a single step.

    Both `set` and `get` operations on global data are governed by a lock, and considered coroutine-safe.
    """

    def __init__(
        self,
        workflow: "Workflow",
        stepwise: bool = False,
    ) -> None:
        self.stepwise = stepwise
        self.is_running = False

        self._workflow = workflow
        # Broker machinery
        self._waiter_id = str(uuid.uuid4())
        self._queues: Dict[str, asyncio.Queue] = {self._waiter_id: asyncio.Queue()}
        self._tasks: Set[asyncio.Task] = set()
        self._broker_log: List[Event] = []
        self._cancel_flag: asyncio.Event = asyncio.Event()
        self._step_flags: Dict[str, asyncio.Event] = {}
        self._step_event_holding: Optional[Event] = None
        self._step_lock: asyncio.Lock = asyncio.Lock()
        self._step_condition: asyncio.Condition = asyncio.Condition(
            lock=self._step_lock
        )
        self._step_event_written: asyncio.Condition = asyncio.Condition(
            lock=self._step_lock
        )
        self._accepted_events: List[Tuple[str, str]] = []
        self._retval: RunResultT = None

        # Map the step names that were executed to a list of events they received.
        # This will be serialized, and is needed to resume a Workflow run passing
        # an existing context.
        self._in_progress: Dict[str, List[Event]] = defaultdict(list)
        # Keep track of the steps currently running. This is only valid when a
        # workflow is running and won't be serialized. Note that a single step
        # might have multiple workers, so we keep a counter.
        self._currently_running_steps: DefaultDict[str, int] = defaultdict(int)
        # Streaming machinery
        self._streaming_queue: asyncio.Queue = asyncio.Queue()
        # Global data storage
        self._lock = asyncio.Lock()
        self._globals: Dict[str, Any] = {}
        # Step-specific instance
        self._events_buffer: Dict[str, List[Event]] = defaultdict(list)

    def _serialize_queue(self, queue: asyncio.Queue, serializer: BaseSerializer) -> str:
        queue_items = list(queue._queue)  # type: ignore
        queue_objs = [serializer.serialize(obj) for obj in queue_items]
        return json.dumps(queue_objs)  # type: ignore

    def _deserialize_queue(
        self,
        queue_str: str,
        serializer: BaseSerializer,
        prefix_queue_objs: List[Any] = [],
    ) -> asyncio.Queue:
        queue_objs = json.loads(queue_str)
        queue_objs = prefix_queue_objs + queue_objs
        queue = asyncio.Queue()  # type: ignore
        for obj in queue_objs:
            event_obj = serializer.deserialize(obj)
            queue.put_nowait(event_obj)
        return queue

    def _serialize_globals(self, serializer: BaseSerializer) -> Dict[str, Any]:
        serialized_globals = {}
        for key, value in self._globals.items():
            try:
                serialized_globals[key] = serializer.serialize(value)
            except Exception as e:
                raise ValueError(f"Failed to serialize value for key {key}: {e}")
        return serialized_globals

    def _deserialize_globals(
        self, serialized_globals: Dict[str, Any], serializer: BaseSerializer
    ) -> Dict[str, Any]:
        deserialized_globals = {}
        for key, value in serialized_globals.items():
            try:
                deserialized_globals[key] = serializer.deserialize(value)
            except Exception as e:
                raise ValueError(f"Failed to deserialize value for key {key}: {e}")
        return deserialized_globals

    def to_dict(self, serializer: Optional[BaseSerializer] = None) -> Dict[str, Any]:
        serializer = serializer or JsonSerializer()

        return {
            "globals": self._serialize_globals(serializer),
            "streaming_queue": self._serialize_queue(self._streaming_queue, serializer),
            "queues": {
                k: self._serialize_queue(v, serializer) for k, v in self._queues.items()
            },
            "stepwise": self.stepwise,
            "events_buffer": {
                k: [serializer.serialize(ev) for ev in v]
                for k, v in self._events_buffer.items()
            },
            "in_progress": {
                k: [serializer.serialize(ev) for ev in v]
                for k, v in self._in_progress.items()
            },
            "accepted_events": self._accepted_events,
            "broker_log": [serializer.serialize(ev) for ev in self._broker_log],
            "waiter_id": self._waiter_id,
            "is_running": self.is_running,
        }

    @classmethod
    def from_dict(
        cls,
        workflow: "Workflow",
        data: Dict[str, Any],
        serializer: Optional[BaseSerializer] = None,
    ) -> "Context":
        serializer = serializer or JsonSerializer()

        context = cls(workflow, stepwise=data["stepwise"])
        context._globals = context._deserialize_globals(data["globals"], serializer)
        context._streaming_queue = context._deserialize_queue(
            data["streaming_queue"], serializer
        )
        context._events_buffer = {
            k: [serializer.deserialize(ev) for ev in v]
            for k, v in data["events_buffer"].items()
        }
        if len(context._events_buffer) == 0:
            context._events_buffer = defaultdict(list)
        context._accepted_events = data["accepted_events"]
        context._waiter_id = data.get("waiter_id", str(uuid.uuid4()))
        context._broker_log = [serializer.deserialize(ev) for ev in data["broker_log"]]
        context.is_running = data["is_running"]
        # load back up whatever was in the queue as well as the events whose steps
        # were in progress when the serialization of the Context took place
        context._queues = {
            k: context._deserialize_queue(
                v, serializer, prefix_queue_objs=data["in_progress"].get(k, [])
            )
            for k, v in data["queues"].items()
        }
        context._in_progress = defaultdict(list)
        return context

    async def set(self, key: str, value: Any, make_private: bool = False) -> None:
        """Store `value` into the Context under `key`.

        Args:
            key: A unique string to identify the value stored.
            value: The data to be stored.

        Raises:
            ValueError: When make_private is True but a key already exists in the global storage.
        """
        if make_private:
            warnings.warn(
                "`make_private` is deprecated and will be ignored", DeprecationWarning
            )

        async with self.lock:
            self._globals[key] = value

    async def mark_in_progress(self, name: str, ev: Event) -> None:
        """Add input event to in_progress dict.

        Args:
            name (str): The name of the step that is in progress.
            ev (Event): The input event that kicked off this step.
        """
        async with self.lock:
            self._in_progress[name].append(ev)

    async def remove_from_in_progress(self, name: str, ev: Event) -> None:
        """Remove input event from active steps.

        Args:
            name (str): The name of the step that has been completed.
            ev (Event): The associated input event that kicked of this completed step.
        """
        async with self.lock:
            events = [e for e in self._in_progress[name] if e != ev]
            self._in_progress[name] = events

    async def add_running_step(self, name: str) -> None:
        async with self.lock:
            self._currently_running_steps[name] += 1

    async def remove_running_step(self, name: str) -> None:
        async with self.lock:
            self._currently_running_steps[name] -= 1
            if self._currently_running_steps[name] == 0:
                del self._currently_running_steps[name]

    async def running_steps(self) -> List[str]:
        async with self.lock:
            return list(self._currently_running_steps)

    async def get(self, key: str, default: Optional[Any] = Ellipsis) -> Any:
        """Get the value corresponding to `key` from the Context.

        Args:
            key: A unique string to identify the value stored.
            default: The value to return when `key` is missing instead of raising an exception.

        Raises:
            ValueError: When there's not value accessible corresponding to `key`.
        """
        async with self.lock:
            if key in self._globals:
                return self._globals[key]
            elif default is not Ellipsis:
                return default

        msg = f"Key '{key}' not found in Context"
        raise ValueError(msg)

    @property
    def data(self) -> Dict[str, Any]:
        """This property is provided for backward compatibility.

        Use `get` and `set` instead.
        """
        msg = "`data` is deprecated, please use the `get` and `set` method to store data into the Context."
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return self._globals

    @property
    def lock(self) -> asyncio.Lock:
        """Returns a mutex to lock the Context."""
        return self._lock

    @property
    def session(self) -> "Context":
        """This property is provided for backward compatibility."""
        msg = "`session` is deprecated, please use the Context instance directly."
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return self

    def _get_full_path(self, ev_type: Type[Event]) -> str:
        return f"{ev_type.__module__}.{ev_type.__name__}"

    def collect_events(
        self, ev: Event, expected: List[Type[Event]]
    ) -> Optional[List[Event]]:
        self._events_buffer[self._get_full_path(type(ev))].append(ev)

        retval: List[Event] = []
        for e_type in expected:
            e_instance_list = self._events_buffer.get(self._get_full_path(e_type))
            if e_instance_list:
                retval.append(e_instance_list.pop(0))

        if len(retval) == len(expected):
            return retval

        # put back the events if unable to collect all
        for ev in retval:
            self._events_buffer[self._get_full_path(type(ev))].append(ev)

        return None

    def send_event(self, message: Event, step: Optional[str] = None) -> None:
        """Sends an event to a specific step in the workflow.

        If step is None, the event is sent to all the receivers and we let
        them discard events they don't want.
        """
        if step is None:
            for queue in self._queues.values():
                queue.put_nowait(message)
        else:
            if step not in self._workflow._get_steps():
                raise WorkflowRuntimeError(f"Step {step} does not exist")

            step_func = self._workflow._get_steps()[step]
            step_config: Optional[StepConfig] = getattr(
                step_func, "__step_config", None
            )

            if step_config and type(message) in step_config.accepted_events:
                self._queues[step].put_nowait(message)
            else:
                raise WorkflowRuntimeError(
                    f"Step {step} does not accept event of type {type(message)}"
                )

        self._broker_log.append(message)

    async def wait_for_event(
        self,
        event_type: Type[T],
        requirements: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = 2000,
    ) -> T:
        """Asynchronously wait for a specific event type to be received.

        Args:
            event_type: The type of event to wait for
            requirements: Optional dict of requirements the event must match
            timeout: Optional timeout in seconds. Defaults to 2000s.

        Returns:
            The event type that was requested.

        Raises:
            asyncio.TimeoutError: If the timeout is reached before receiving matching event
        """
        requirements = requirements or {}

        while True:
            event = await asyncio.wait_for(
                self._queues[self._waiter_id].get(), timeout=timeout
            )
            if type(event) is event_type:
                if all(
                    event.get(k, default=None) == v for k, v in requirements.items()
                ):
                    return event
                else:
                    continue

    def write_event_to_stream(self, ev: Optional[Event]) -> None:
        self._streaming_queue.put_nowait(ev)

    def get_result(self) -> RunResultT:
        """Returns the result of the workflow."""
        return self._retval

    @property
    def streaming_queue(self) -> asyncio.Queue:
        return self._streaming_queue
