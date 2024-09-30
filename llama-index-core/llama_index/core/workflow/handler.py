import asyncio
from typing import Any, AsyncGenerator, Optional

from llama_index.core.workflow.context import Context
from llama_index.core.workflow.events import Event, StopEvent
from llama_index.core.workflow.errors import WorkflowDone


class WorkflowHandler(asyncio.Future):
    def __init__(
        self, *args: Any, ctx: Optional[Context] = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
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
        if self.ctx and not self.ctx.stepwise:
            raise ValueError("Stepwise context is required to run stepwise.")

        if self.ctx:
            # Unblock all pending steps
            for flag in self.ctx._step_flags.values():
                flag.set()

            # Yield back control to the event loop to give an unblocked step
            # the chance to run (we won't actually sleep here).
            await asyncio.sleep(0)

            # check if StopEvent is in holding
            if isinstance(self.ctx._step_event_holding, StopEvent):
                # See if we're done, or if a step raised any error
                retval = None
                we_done = False
                exception_raised = None
                for t in self.ctx._tasks:
                    # Check if we're done
                    if not t.done():
                        continue

                    we_done = True
                    e = t.exception()
                    if type(e) != WorkflowDone:
                        exception_raised = e

                if we_done:
                    # Remove any reference to the tasks
                    for t in self.ctx._tasks:
                        t.cancel()
                        await asyncio.sleep(0)
                    res = self.ctx.get_result()

                    self.set_result(res)

                if exception_raised:
                    raise exception_raised

            else:
                # notify unblocked task that we're ready to accept next event
                async with self.ctx._step_condition:
                    self.ctx._step_condition.notify()

                # Wait to be notified that the new_ev has been written
                async with self.ctx._step_event_written:
                    await self.ctx._step_event_written.wait()
                    retval = self.ctx._step_event_holding

        else:
            raise ValueError("Context is not set!")

        return retval
