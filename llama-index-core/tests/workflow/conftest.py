import pytest

from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.events import StartEvent, StopEvent, Event
from llama_index.core.workflow.session import WorkflowSession
from llama_index.core.workflow.workflow import Workflow
from llama_index.core.bridge.pydantic import Field


class OneTestEvent(Event):
    test_param: str = Field(default="test")


class AnotherTestEvent(Event):
    another_test_param: str = Field(default="another_test")


class LastEvent(Event):
    pass


class DummyWorkflow(Workflow):
    @step
    async def start_step(self, ev: StartEvent) -> OneTestEvent:
        return OneTestEvent()

    @step
    async def middle_step(self, ev: OneTestEvent) -> LastEvent:
        return LastEvent()

    @step
    async def end_step(self, ev: LastEvent) -> StopEvent:
        return StopEvent(result="Workflow completed")


@pytest.fixture()
def workflow():
    return DummyWorkflow()


@pytest.fixture()
def events():
    return [OneTestEvent, AnotherTestEvent]


@pytest.fixture()
def session():
    return WorkflowSession(workflow=Workflow())
