"""Self Reflection Agent Worker."""

import logging
import uuid
from typing import Any, List, Optional, Tuple

from llama_index.core.agent.types import (
    BaseAgentWorker,
    Task,
    TaskStep,
    TaskStepOutput,
)
from llama_index.core.bridge.pydantic import BaseModel, Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager, trace_method
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.base.llms.generic_utils import messages_to_prompt
from llama_index.core.llms.llm import LLM
from llama_index.core.program.llm_prompt_program import BaseLLMFunctionProgram

import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

REFLECTION_PROMPT_TEMPLATE = """
You are responsible for evaluating whether an agent is taking the right steps towards a solution.

You are given the current conversation history, which contains the user task, assistant responses + tool calls, \
as well as any feedback that you have already given.

Evaluate the following criteria:
- Whether the tool call arguments make sense
    - Specifically, check whether page numbers are specified when they shouldn't have. They should ONLY be specified
    if in the user query. Do NOT return done if this is the case.
- Whether the tool output completes the task.
- Whether the final message is an ASSISTANT message (not a tool message). Only if the final message
    is an assistant message does it mean the agent is done thinking.

Given the current chat history, please output a reflection response in the following format evaluating
the quality of the agent trajectory:

{chat_history}
"""


REFLECTION_RESPONSE_TEMPLATE = """
Here is a reflection on the current trajectory.

{reflection_output}

If is_done is not True, there should be feedback on what is going wrong.
Given the feedback, please try again.
"""

DEFAULT_MAX_ITERATIONS = 5


CORRECT_PROMPT_TEMPLATE = """
You are responsible for correcting an input based on a provided feedback.

Input:

{input_str}

Feedback:

{feedback}

Use the provided information to generate a corrected version of input.
"""

CORRECT_RESPONSE_FSTRING = "Here is a corrected version of the input.\n{correction}"


class Reflection(BaseModel):
    """Reflection of the current agent state."""

    is_done: bool = Field(
        ...,
        description="Whether the task is successfully completed according to evaluation criteria (do NOT output True if not).",
    )
    feedback: str = Field(
        ...,
        description="Feedback on how the output can be improved (especially if score is less than 5)",
    )


class Correction(BaseModel):
    """Data class for holding the corrected input."""

    correction: str = Field(default_factory=str, description="Corrected input")


class SelfReflectionAgentWorker(BaseModel, BaseAgentWorker):
    """Self Reflection Agent Worker.

    This agent performs a reflection without any tools on a given response
    and subsequently performs correction.
    """

    callback_manager: CallbackManager = Field(default=CallbackManager([]))
    max_iterations: int = Field(default=DEFAULT_MAX_ITERATIONS)
    _llm: LLM = PrivateAttr()
    _reflection_program: BaseLLMFunctionProgram = PrivateAttr()
    _correction_program: BaseLLMFunctionProgram = PrivateAttr()
    _verbose: bool = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        callback_manager: Optional[CallbackManager] = None,
        llm: Optional[LLM] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        self._llm = llm
        self._verbose = verbose

        # define _reflection and _correction programs
        try:
            from llama_index.program.openai import OpenAIPydanticProgram
        except ImportError:
            raise ImportError(
                "Missing OpenAIPydanticProgram. Please run `pip install llama-index-program-openai`."
            )
        self._reflection_program = OpenAIPydanticProgram.from_defaults(
            Reflection,
            prompt_template_str=REFLECTION_PROMPT_TEMPLATE,
            llm=self._llm,
        )
        self._correction_program = OpenAIPydanticProgram.from_defaults(
            Correction,
            prompt_template_str=CORRECT_PROMPT_TEMPLATE,
            llm=self._llm,
        )

        super().__init__(
            callback_manager=callback_manager or CallbackManager([]),
            max_iterations=max_iterations,
            **kwargs,
        )

    @classmethod
    def from_args(
        cls,
        llm: Optional[LLM] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "SelfReflectionAgentWorker":
        """Convenience constructor."""
        if llm is None:
            try:
                from llama_index.llms.openai import OpenAI
            except ImportError:
                raise ImportError(
                    "Missing OpenAI LLMs. Please run `pip install llama-index-llms-openai`."
                )
            llm = OpenAI(model="gpt-4-turbo-preview", temperature=0)

        return cls(
            llm=llm,
            max_iterations=max_iterations,
            callback_manager=callback_manager,
            verbose=verbose,
            **kwargs,
        )

    def initialize_step(self, task: Task, **kwargs: Any) -> TaskStep:
        """Initialize step from task."""
        # temporary memory for new messages
        new_memory = ChatMemoryBuffer.from_defaults()

        # put current history in new memory
        messages = task.memory.get()
        for message in messages:
            new_memory.put(message)

        # initialize task state
        task_state = {
            "new_memory": new_memory,
            "sources": [],
        }
        task.extra_state.update(task_state)

        return TaskStep(
            task_id=task.task_id,
            step_id=str(uuid.uuid4()),
            input=task.input,
            step_state={"count": 0},
        )

    @dispatcher.span
    def _reflect(
        self, chat_history: List[ChatMessage]
    ) -> Tuple[Reflection, ChatMessage]:
        """Reflect on the trajectory."""
        reflection = self._reflection_program(
            chat_history=messages_to_prompt(chat_history)
        )

        if self._verbose:
            print(f"> Reflection: {reflection.dict()}")

        # end state: return user message
        reflection_output_str = (
            f"Is Done: {reflection.is_done}\nCritique: {reflection.feedback}"
        )
        critique = REFLECTION_RESPONSE_TEMPLATE.format(
            reflection_output=reflection_output_str
        )

        return reflection, ChatMessage.from_str(critique, role="user")

    @dispatcher.span
    def _correct(self, input_str: str, critique: str) -> ChatMessage:
        correction = self._correction_program(input_str=input_str, feedback=critique)

        correct_response_str = CORRECT_RESPONSE_FSTRING.format(
            correction=correction.correction
        )
        if self._verbose:
            print(f"Correction: {correction.correction}", flush=True)
        return ChatMessage.from_str(correct_response_str, role="assistant")

    @dispatcher.span
    @trace_method("run_step")
    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step."""
        state = step.step_state
        state["count"] += 1

        messages = task.extra_state["new_memory"].get()
        current_response = messages[-1].content

        # reflect
        reflection, reflection_msg = self._reflect(chat_history=messages)
        is_done = reflection.is_done

        critique_msg = ChatMessage(role=MessageRole.USER, content=reflection_msg)
        task.extra_state["new_memory"].put(critique_msg)

        # correct
        if is_done:
            agent_response = AgentChatResponse(
                response=current_response, sources=task.extra_state["sources"]
            )
            new_steps = []
        else:
            input_str = current_response.replace(
                "Here is a corrected version of the input.\n", ""
            )
            correct_msg = self._correct(
                input_str=input_str, critique=reflection_msg.content
            )
            agent_response = AgentChatResponse(response=str(correct_msg))
            task.extra_state["new_memory"].put(correct_msg)

            if self.max_iterations == state["count"]:
                new_steps = []
            else:
                new_steps = [
                    step.get_next_step(
                        step_id=str(uuid.uuid4()),
                        # NOTE: input is unused
                        input=None,
                        step_state=state,
                    )
                ]

        return TaskStepOutput(
            output=agent_response,
            task_step=step,
            is_last=is_done | (self.max_iterations == state["count"]),
            next_steps=new_steps,
        )

    # Async methods
    @dispatcher.span
    async def _areflect(
        self, chat_history: List[ChatMessage]
    ) -> Tuple[Reflection, ChatMessage]:
        """Reflect on the trajectory."""
        reflection = await self._reflection_program.acall(
            chat_history=messages_to_prompt(chat_history)
        )

        if self._verbose:
            print(f"> Reflection: {reflection.dict()}")

        # end state: return user message
        reflection_output_str = (
            f"Is Done: {reflection.is_done}\nCritique: {reflection.feedback}"
        )
        critique = REFLECTION_RESPONSE_TEMPLATE.format(
            reflection_output=reflection_output_str
        )

        return reflection, ChatMessage.from_str(critique, role="user")

    @dispatcher.span
    async def _acorrect(self, input_str: str, critique: str) -> ChatMessage:
        correction = await self._correction_program.acall(
            input_str=input_str, feedback=critique
        )

        correct_response_str = CORRECT_RESPONSE_FSTRING.format(
            correction=correction.correction
        )
        if self._verbose:
            print(f"Correction: {correction.correction}", flush=True)
        return ChatMessage.from_str(correct_response_str, role="assistant")

    @dispatcher.span
    @trace_method("run_step")
    async def arun_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async)."""
        state = step.step_state
        state["count"] += 1

        messages = task.extra_state["new_memory"].get()
        current_response = messages[-1].content

        # reflect
        reflection, reflection_msg = await self._areflect(chat_history=messages)
        is_done = reflection.is_done

        critique_msg = ChatMessage(role=MessageRole.USER, content=reflection_msg)
        task.extra_state["new_memory"].put(critique_msg)

        # correct
        if is_done:
            agent_response = AgentChatResponse(
                response=current_response, sources=task.extra_state["sources"]
            )
            new_steps = []
        else:
            input_str = current_response.replace(
                "Here is a corrected version of the input.\n", ""
            )
            correct_msg = await self._acorrect(
                input_str=input_str, critique=reflection_msg.content
            )
            agent_response = AgentChatResponse(response=str(correct_msg))
            task.extra_state["new_memory"].put(correct_msg)

            if self.max_iterations == state["count"]:
                new_steps = []
            else:
                new_steps = [
                    step.get_next_step(
                        step_id=str(uuid.uuid4()),
                        # NOTE: input is unused
                        input=None,
                        step_state=state,
                    )
                ]

        return TaskStepOutput(
            output=agent_response,
            task_step=step,
            is_last=is_done | (self.max_iterations == state["count"]),
            next_steps=new_steps,
        )

    # Stream methods
    @dispatcher.span
    @trace_method("run_step")
    def stream_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step (stream)."""
        raise NotImplementedError("Stream not supported for self reflection agent")

    @dispatcher.span
    @trace_method("run_step")
    async def astream_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async stream)."""
        raise NotImplementedError("Stream not supported for self reflection agent")

    def get_all_messages(self, task: Task) -> List[ChatMessage]:
        return (
            self.prefix_messages
            + task.memory.get()
            + task.extra_state["new_memory"].get_all()
        )

    def finalize_task(self, task: Task, **kwargs: Any) -> None:
        """Finalize task, after all the steps are completed."""
        # add new messages to memory
        task.memory.set(task.extra_state["new_memory"].get_all())
        # reset new memory
        task.extra_state["new_memory"].reset()
