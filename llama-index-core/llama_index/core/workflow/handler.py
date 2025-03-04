import asyncio
from typing import Any, AsyncGenerator, Optional

from llama_index.core.workflow.context import Context
from llama_index.core.workflow.errors import WorkflowDone
from llama_index.core.workflow.events import Event, StopEvent

from .types import RunResultT
from .utils import BUSY_WAIT_DELAY


class WorkflowHandler(asyncio.Future[RunResultT]):
    def __init__(
        self,
        *args: Any,
        ctx: Optional[Context] = None,
        run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.run_id = run_id
        self.ctx = ctx

    def __str__(self) -> str:
        return str(self.result())

    def is_done(self) -> bool:
        return self.done()

    async def stream_events(self) -> AsyncGenerator[Event, None]:
        if not self.ctx:
            raise ValueError("Context is not set!")

        while True:
            ev = await self.ctx.streaming_queue.get()

            yield ev

            if type(ev) is StopEvent:
                break

    async def run_step(self) -> Optional[Event]:
        """Runs the next workflow step and returns the output Event.

        If return is None, then the workflow is considered done.

        Examples:
            ```python
            handler = workflow.run(stepwise=True)
            while not handler.is_done():
                ev = await handler.run_step()
                handler.ctx.send_event(ev)

            result = handler.result()
            print(result)
            ```
        """
        # since event is sent before calling this method, we need to unblock the event loop
        await asyncio.sleep(0)

        if self.ctx and not self.ctx.stepwise:
            raise ValueError("Stepwise context is required to run stepwise.")

        if self.ctx:
            try:
                # Unblock all pending steps
                for flag in self.ctx._step_flags.values():
                    flag.set()

                # Yield back control to the event loop to give an unblocked step
                # the chance to run (we won't actually sleep here).
                await asyncio.sleep(0)

                # check if we're done, or if a step raised error
                we_done = False
                exception_raised = None
                retval = None
                for t in self.ctx._tasks:
                    # Check if we're done
                    if not t.done():
                        continue

                    we_done = True
                    e = t.exception()
                    if type(e) is not WorkflowDone:
                        exception_raised = e

                if we_done:
                    # Remove any reference to the tasks
                    for t in self.ctx._tasks:
                        t.cancel()
                        await asyncio.sleep(0)

                    # the context is no longer running
                    self.ctx.is_running = False

                    if exception_raised:
                        raise exception_raised

                    if not self.done():
                        self.set_result(self.ctx.get_result())
                else:
                    # Continue with running next step. Make sure we wait for the
                    # step function to return before proceeding.
                    in_progress = len(await self.ctx.running_steps())
                    while in_progress:
                        await asyncio.sleep(BUSY_WAIT_DELAY)
                        in_progress = len(await self.ctx.running_steps())

                    # notify unblocked task that we're ready to accept next event
                    async with self.ctx._step_condition:
                        self.ctx._step_condition.notify()

                    # Wait to be notified that the new_ev has been written
                    async with self.ctx._step_event_written:
                        await self.ctx._step_event_written.wait()
                        retval = self.ctx._step_event_holding
            except Exception as e:
                if not self.is_done():  # Avoid InvalidStateError edge case
                    self.set_exception(e)
                raise
        else:
            raise ValueError("Context is not set!")

        return retval

    async def cancel_run(self) -> None:
        """Method to cancel a Workflow execution."""
        if self.ctx:
            self.ctx._cancel_flag.set()
            await asyncio.sleep(0)
