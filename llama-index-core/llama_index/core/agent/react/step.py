"""ReAct agent worker."""

import asyncio
import json
import uuid
from functools import partial
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    cast,
    Callable,
)

from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.agent.react.output_parser import ReActOutputParser
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    BaseReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
from llama_index.core.agent.types import (
    BaseAgentWorker,
    Task,
    TaskStep,
    TaskStepOutput,
)
from llama_index.core.base.llms.types import MessageRole
from llama_index.core.callbacks import (
    CallbackManager,
    CBEventType,
    EventPayload,
    trace_method,
)
from llama_index.core.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    AgentChatResponse,
    StreamingAgentChatResponse,
)
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.agent import AgentToolCallEvent
from llama_index.core.llms.llm import LLM
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.memory.types import BaseMemory
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.settings import Settings
from llama_index.core.tools import BaseTool, ToolOutput, adapt_to_async_tool
from llama_index.core.tools.types import AsyncBaseTool
from llama_index.core.types import Thread
from llama_index.core.utils import print_text

dispatcher = get_dispatcher(__name__)


def add_user_step_to_reasoning(
    step: TaskStep,
    memory: BaseMemory,
    current_reasoning: List[BaseReasoningStep],
    verbose: bool = False,
) -> None:
    """Add user step to memory."""
    if "is_first" in step.step_state and step.step_state["is_first"]:
        # add to new memory
        memory.put(ChatMessage(content=step.input, role=MessageRole.USER))
        step.step_state["is_first"] = False
    else:
        reasoning_step = ObservationReasoningStep(observation=step.input)
        current_reasoning.append(reasoning_step)
        if verbose:
            print(f"Added user message to memory: {step.input}")


def tell_llm_about_failure_in_extract_reasoning_step(
    callback_manager: CallbackManager, _: ValueError
) -> ToolOutput:
    """
    If the developer has instructed to tell the Agent a complaint about its non-cooperation,
    we will emit a Tool Output that we prepared (at initialization time) to the LLM, so that
    the LLM can be more cooperative in its next generation.
    """
    message = "Error: Could not parse output. Please follow the thought-action-input format. Try again."
    dummy_tool_output = ToolOutput(
        content=message,
        tool_name="unknown",
        raw_input={},
        raw_output=message,
    )
    with callback_manager.event(
        CBEventType.FUNCTION_CALL,
        payload={
            EventPayload.FUNCTION_CALL: "unknown",
        },
    ) as event:
        event.on_end(payload={EventPayload.FUNCTION_OUTPUT: str(dummy_tool_output)})

    return dummy_tool_output


class ReActAgentWorker(BaseAgentWorker):
    """OpenAI Agent worker."""

    def __init__(
        self,
        tools: Sequence[BaseTool],
        llm: LLM,
        max_iterations: int = 10,
        react_chat_formatter: Optional[ReActChatFormatter] = None,
        output_parser: Optional[ReActOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        handle_reasoning_failure_fn: Optional[
            Callable[[CallbackManager, Exception], ToolOutput]
        ] = None,
    ) -> None:
        self._llm = llm
        self.callback_manager = callback_manager or llm.callback_manager
        self._max_iterations = max_iterations
        self._react_chat_formatter = react_chat_formatter or ReActChatFormatter()
        self._output_parser = output_parser or ReActOutputParser()
        self._verbose = verbose
        self._handle_reasoning_failure_fn = (
            handle_reasoning_failure_fn
            or tell_llm_about_failure_in_extract_reasoning_step
        )

        if len(tools) > 0 and tool_retriever is not None:
            raise ValueError("Cannot specify both tools and tool_retriever")
        elif len(tools) > 0:
            self._get_tools = lambda _: tools
        elif tool_retriever is not None:
            tool_retriever_c = cast(ObjectRetriever[BaseTool], tool_retriever)
            self._get_tools = lambda message: tool_retriever_c.retrieve(message)
        else:
            self._get_tools = lambda _: []

    @classmethod
    def from_tools(
        cls,
        tools: Optional[Sequence[BaseTool]] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        llm: Optional[LLM] = None,
        max_iterations: int = 10,
        react_chat_formatter: Optional[ReActChatFormatter] = None,
        output_parser: Optional[ReActOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        handle_reasoning_failure_fn: Optional[
            Callable[[CallbackManager, Exception], ToolOutput]
        ] = None,
        **kwargs: Any,
    ) -> "ReActAgentWorker":
        """Convenience constructor method from set of BaseTools (Optional).

        NOTE: kwargs should have been exhausted by this point. In other words
        the various upstream components such as BaseSynthesizer (response synthesizer)
        or BaseRetriever should have picked up off their respective kwargs in their
        constructions.

        Returns:
            ReActAgentWorker
        """
        llm = llm or Settings.llm
        if callback_manager is not None:
            llm.callback_manager = callback_manager
        return cls(
            tools=tools or [],
            tool_retriever=tool_retriever,
            llm=llm,
            max_iterations=max_iterations,
            react_chat_formatter=react_chat_formatter,
            output_parser=output_parser,
            callback_manager=callback_manager,
            verbose=verbose,
            handle_reasoning_failure_fn=handle_reasoning_failure_fn,
        )

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        # TODO: the ReAct formatter does not explicitly specify PromptTemplate
        # objects, but wrap it in this to obey the interface
        sys_header = self._react_chat_formatter.system_header
        return {"system_prompt": PromptTemplate(sys_header)}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "system_prompt" in prompts:
            sys_prompt = cast(PromptTemplate, prompts["system_prompt"])
            self._react_chat_formatter.system_header = sys_prompt.template

    def initialize_step(self, task: Task, **kwargs: Any) -> TaskStep:
        """Initialize step from task."""
        sources: List[ToolOutput] = []
        current_reasoning: List[BaseReasoningStep] = []
        # temporary memory for new messages
        new_memory = ChatMemoryBuffer.from_defaults()

        # initialize task state
        task_state = {
            "sources": sources,
            "current_reasoning": current_reasoning,
            "new_memory": new_memory,
        }
        task.extra_state.update(task_state)

        return TaskStep(
            task_id=task.task_id,
            step_id=str(uuid.uuid4()),
            input=task.input,
            step_state={"is_first": True},
        )

    def get_tools(self, input: str) -> List[AsyncBaseTool]:
        """Get tools."""
        return [adapt_to_async_tool(t) for t in self._get_tools(input)]

    def _extract_reasoning_step(
        self, output: ChatResponse, is_streaming: bool = False
    ) -> Tuple[str, List[BaseReasoningStep], bool]:
        """
        Extracts the reasoning step from the given output.

        This method parses the message content from the output,
        extracts the reasoning step, and determines whether the processing is
        complete. It also performs validation checks on the output and
        handles possible errors.
        """
        if output.message.content is None:
            raise ValueError("Got empty message.")
        message_content = output.message.content

        current_reasoning = []
        try:
            reasoning_step = self._output_parser.parse(message_content, is_streaming)
        except BaseException as exc:
            raise ValueError(f"Could not parse output: {message_content}") from exc
        if self._verbose:
            print_text(f"{reasoning_step.get_content()}\n", color="pink")
        current_reasoning.append(reasoning_step)

        if reasoning_step.is_done:
            return message_content, current_reasoning, True

        reasoning_step = cast(ActionReasoningStep, reasoning_step)
        if not isinstance(reasoning_step, ActionReasoningStep):
            raise ValueError(f"Expected ActionReasoningStep, got {reasoning_step}")

        return message_content, current_reasoning, False

    def _process_actions(
        self,
        task: Task,
        tools: Sequence[AsyncBaseTool],
        output: ChatResponse,
        is_streaming: bool = False,
    ) -> Tuple[List[BaseReasoningStep], bool]:
        tools_dict: Dict[str, AsyncBaseTool] = {
            tool.metadata.get_name(): tool for tool in tools
        }
        tool = None

        try:
            _, current_reasoning, is_done = self._extract_reasoning_step(
                output, is_streaming
            )
        except ValueError as exp:
            current_reasoning = []
            tool_output = self._handle_reasoning_failure_fn(self.callback_manager, exp)
        else:
            if is_done:
                return current_reasoning, True

            # call tool with input
            reasoning_step = cast(ActionReasoningStep, current_reasoning[-1])
            if reasoning_step.action in tools_dict:
                tool = tools_dict[reasoning_step.action]
                with self.callback_manager.event(
                    CBEventType.FUNCTION_CALL,
                    payload={
                        EventPayload.FUNCTION_CALL: reasoning_step.action_input,
                        EventPayload.TOOL: tool.metadata,
                    },
                ) as event:
                    try:
                        dispatcher.event(
                            AgentToolCallEvent(
                                arguments=json.dumps({**reasoning_step.action_input}),
                                tool=tool.metadata,
                            )
                        )
                        tool_output = tool.call(**reasoning_step.action_input)
                    except Exception as e:
                        tool_output = ToolOutput(
                            content=f"Error: {e!s}",
                            tool_name=tool.metadata.name,
                            raw_input={"kwargs": reasoning_step.action_input},
                            raw_output=e,
                            is_error=True,
                        )
                    event.on_end(
                        payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)}
                    )
            else:
                tool_output = self._handle_nonexistent_tool_name(reasoning_step)

        task.extra_state["sources"].append(tool_output)

        observation_step = ObservationReasoningStep(
            observation=str(tool_output),
            return_direct=(
                tool.metadata.return_direct and not tool_output.is_error
                if tool
                else False
            ),
        )
        current_reasoning.append(observation_step)
        if self._verbose:
            print_text(f"{observation_step.get_content()}\n", color="blue")
        return (
            current_reasoning,
            tool.metadata.return_direct and not tool_output.is_error if tool else False,
        )

    async def _aprocess_actions(
        self,
        task: Task,
        tools: Sequence[AsyncBaseTool],
        output: ChatResponse,
        is_streaming: bool = False,
    ) -> Tuple[List[BaseReasoningStep], bool]:
        tools_dict = {tool.metadata.name: tool for tool in tools}
        tool = None

        try:
            _, current_reasoning, is_done = self._extract_reasoning_step(
                output, is_streaming
            )
        except ValueError as exp:
            current_reasoning = []
            tool_output = self._handle_reasoning_failure_fn(self.callback_manager, exp)
        else:
            if is_done:
                return current_reasoning, True

            # call tool with input
            reasoning_step = cast(ActionReasoningStep, current_reasoning[-1])
            if reasoning_step.action in tools_dict:
                tool = tools_dict[reasoning_step.action]
                with self.callback_manager.event(
                    CBEventType.FUNCTION_CALL,
                    payload={
                        EventPayload.FUNCTION_CALL: reasoning_step.action_input,
                        EventPayload.TOOL: tool.metadata,
                    },
                ) as event:
                    try:
                        dispatcher.event(
                            AgentToolCallEvent(
                                arguments=json.dumps({**reasoning_step.action_input}),
                                tool=tool.metadata,
                            )
                        )
                        tool_output = await tool.acall(**reasoning_step.action_input)
                    except Exception as e:
                        tool_output = ToolOutput(
                            content=f"Error: {e!s}",
                            tool_name=tool.metadata.name,
                            raw_input={"kwargs": reasoning_step.action_input},
                            raw_output=e,
                            is_error=True,
                        )
                    event.on_end(
                        payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)}
                    )
            else:
                tool_output = self._handle_nonexistent_tool_name(reasoning_step)

        task.extra_state["sources"].append(tool_output)

        observation_step = ObservationReasoningStep(
            observation=str(tool_output),
            return_direct=(
                tool.metadata.return_direct and not tool_output.is_error
                if tool
                else False
            ),
        )
        current_reasoning.append(observation_step)
        if self._verbose:
            print_text(f"{observation_step.get_content()}\n", color="blue")
        return (
            current_reasoning,
            tool.metadata.return_direct and not tool_output.is_error if tool else False,
        )

    def _handle_nonexistent_tool_name(
        self, reasoning_step: ActionReasoningStep
    ) -> ToolOutput:
        # We still emit a `tool_output` object to the task, so that the LLM can know
        # it has hallucinated in the next reasoning step.
        with self.callback_manager.event(
            CBEventType.FUNCTION_CALL,
            payload={
                EventPayload.FUNCTION_CALL: reasoning_step.action_input,
            },
        ) as event:
            # TODO(L10N): This should be localized.
            content = f"Error: No such tool named `{reasoning_step.action}`."
            tool_output = ToolOutput(
                content=content,
                tool_name=reasoning_step.action,
                raw_input={"kwargs": reasoning_step.action_input},
                raw_output=content,
                is_error=True,
            )
            event.on_end(payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)})
        return tool_output

    def _get_response(
        self,
        current_reasoning: List[BaseReasoningStep],
        sources: List[ToolOutput],
    ) -> AgentChatResponse:
        """Get response from reasoning steps."""
        if len(current_reasoning) == 0:
            raise ValueError("No reasoning steps were taken.")
        elif len(current_reasoning) == self._max_iterations:
            raise ValueError("Reached max iterations.")

        if isinstance(current_reasoning[-1], ResponseReasoningStep):
            response_step = cast(ResponseReasoningStep, current_reasoning[-1])
            response_str = response_step.response
        elif (
            isinstance(current_reasoning[-1], ObservationReasoningStep)
            and current_reasoning[-1].return_direct
        ):
            response_str = current_reasoning[-1].observation
        else:
            response_str = current_reasoning[-1].get_content()

        # TODO: add sources from reasoning steps
        return AgentChatResponse(response=response_str, sources=sources)

    def _get_task_step_response(
        self, agent_response: AGENT_CHAT_RESPONSE_TYPE, step: TaskStep, is_done: bool
    ) -> TaskStepOutput:
        """Get task step response."""
        if is_done:
            new_steps = []
        else:
            new_steps = [
                step.get_next_step(
                    step_id=str(uuid.uuid4()),
                    # NOTE: input is unused
                    input=None,
                )
            ]

        return TaskStepOutput(
            output=agent_response,
            task_step=step,
            is_last=is_done,
            next_steps=new_steps,
        )

    def _infer_stream_chunk_is_final(
        self, chunk: ChatResponse, missed_chunks_storage: list
    ) -> bool:
        """Infers if a chunk from a live stream is the start of the final
        reasoning step. (i.e., and should eventually become
        ResponseReasoningStep — not part of this function's logic tho.).

        Args:
            chunk (ChatResponse): the current chunk stream to check
            missed_chunks_storage (list): list to store missed chunks

        Returns:
            bool: Boolean on whether the chunk is the start of the final response
        """
        latest_content = (
            None if chunk.message.content is None else chunk.message.content.strip()
        )
        if latest_content:
            # doesn't follow thought-action format
            # keep first chunks
            if len(latest_content) < len("Thought"):
                missed_chunks_storage.append(chunk)
            elif not latest_content.startswith("Thought"):
                return True
            elif "Answer:" in latest_content:
                missed_chunks_storage.clear()
                return True
        return False

    def _add_back_chunk_to_stream(
        self,
        chunks: List[ChatResponse],
        chat_stream: Generator[ChatResponse, None, None],
    ) -> Generator[ChatResponse, None, None]:
        """Helper method for adding back initial chunk stream of final response
        back to the rest of the chat_stream.

        Args:
            chunks List[ChatResponse]: the chunks to add back to the beginning of the
                                    chat_stream.

        Return:
            Generator[ChatResponse, None, None]: the updated chat_stream
        """

        def gen() -> Generator[ChatResponse, None, None]:
            yield from chunks
            yield from chat_stream

        return gen()

    async def _async_add_back_chunk_to_stream(
        self,
        chunks: List[ChatResponse],
        chat_stream: AsyncGenerator[ChatResponse, None],
    ) -> AsyncGenerator[ChatResponse, None]:
        """Helper method for adding back initial chunk stream of final response
        back to the rest of the chat_stream.

        NOTE: this itself is not an async function.

        Args:
            chunks List[ChatResponse]: the chunks to add back to the beginning of the
                                    chat_stream.

        Return:
            AsyncGenerator[ChatResponse, None]: the updated async chat_stream
        """
        for chunk in chunks:
            yield chunk

        async for item in chat_stream:
            yield item

    def _run_step(
        self,
        step: TaskStep,
        task: Task,
    ) -> TaskStepOutput:
        """Run step."""
        if step.input is not None:
            add_user_step_to_reasoning(
                step,
                task.extra_state["new_memory"],
                task.extra_state["current_reasoning"],
                verbose=self._verbose,
            )
        # TODO: see if we want to do step-based inputs
        tools = self.get_tools(task.input)
        input_chat = self._react_chat_formatter.format(
            tools,
            chat_history=task.memory.get(input=task.input)
            + task.extra_state["new_memory"].get_all(),
            # Let LLM know that it can only take at most this many steps to complete the task.
            # - `//2`: An observation step also takes as an iteration.
            # - `-1`: The final output reasoning step needs to take a spot.
            allowance=(self._max_iterations - 1) // 2,
            current_reasoning=task.extra_state["current_reasoning"],
        )

        # send prompt
        chat_response = self._llm.chat(input_chat)
        # given react prompt outputs, call tools or return response
        reasoning_steps, is_done = self._process_actions(
            task, tools, output=chat_response
        )
        task.extra_state["current_reasoning"].extend(reasoning_steps)
        agent_response = self._get_response(
            task.extra_state["current_reasoning"], task.extra_state["sources"]
        )
        if is_done:
            task.extra_state["new_memory"].put(
                ChatMessage(content=agent_response.response, role=MessageRole.ASSISTANT)
            )

        return self._get_task_step_response(agent_response, step, is_done)

    async def _arun_step(
        self,
        step: TaskStep,
        task: Task,
    ) -> TaskStepOutput:
        """Run step."""
        if step.input is not None:
            add_user_step_to_reasoning(
                step,
                task.extra_state["new_memory"],
                task.extra_state["current_reasoning"],
                verbose=self._verbose,
            )
        # TODO: see if we want to do step-based inputs
        tools = self.get_tools(task.input)

        input_chat = self._react_chat_formatter.format(
            tools,
            chat_history=task.memory.get(input=task.input)
            + task.extra_state["new_memory"].get_all(),
            # Let LLM know that it can only take at most this many steps to complete the task.
            # - `//2`: An observation step also takes as an iteration.
            # - `-1`: The final output reasoning step needs to take a spot.
            allowance=(self._max_iterations - 1) // 2,
            current_reasoning=task.extra_state["current_reasoning"],
        )
        # send prompt
        chat_response = await self._llm.achat(input_chat)
        # given react prompt outputs, call tools or return response
        reasoning_steps, is_done = await self._aprocess_actions(
            task, tools, output=chat_response
        )
        task.extra_state["current_reasoning"].extend(reasoning_steps)
        agent_response = self._get_response(
            task.extra_state["current_reasoning"], task.extra_state["sources"]
        )
        if is_done:
            task.extra_state["new_memory"].put(
                ChatMessage(content=agent_response.response, role=MessageRole.ASSISTANT)
            )

        return self._get_task_step_response(agent_response, step, is_done)

    def _run_step_stream(
        self,
        step: TaskStep,
        task: Task,
    ) -> TaskStepOutput:
        """Run step."""
        if step.input is not None:
            add_user_step_to_reasoning(
                step,
                task.extra_state["new_memory"],
                task.extra_state["current_reasoning"],
                verbose=self._verbose,
            )
        # TODO: see if we want to do step-based inputs
        tools = self.get_tools(task.input)

        input_chat = self._react_chat_formatter.format(
            tools,
            chat_history=task.memory.get(input=task.input)
            + task.extra_state["new_memory"].get_all(),
            # Let LLM know that it can only take at most this many steps to complete the task.
            # - `//2`: An observation step also takes as an iteration.
            # - `-1`: The final output reasoning step needs to take a spot.
            allowance=(self._max_iterations - 1) // 2,
            current_reasoning=task.extra_state["current_reasoning"],
        )

        chat_stream = self._llm.stream_chat(input_chat)

        # iterate over stream, break out if is final answer after the "Answer: "
        full_response = ChatResponse(
            message=ChatMessage(content=None, role="assistant")
        )
        missed_chunks_storage: List[ChatResponse] = []
        is_done = False
        for latest_chunk in chat_stream:
            full_response = latest_chunk
            is_done = self._infer_stream_chunk_is_final(
                latest_chunk, missed_chunks_storage
            )
            if is_done:
                break

        non_streaming_agent_response = None
        agent_response_stream = None
        if not is_done:
            # given react prompt outputs, call tools or return response
            reasoning_steps, is_done = self._process_actions(
                task, tools=tools, output=full_response, is_streaming=True
            )
            task.extra_state["current_reasoning"].extend(reasoning_steps)
            # use _get_response to return intermediate response
            non_streaming_agent_response = self._get_response(
                task.extra_state["current_reasoning"], task.extra_state["sources"]
            )
            if is_done:
                non_streaming_agent_response.is_dummy_stream = True
                task.extra_state["new_memory"].put(
                    ChatMessage(
                        content=non_streaming_agent_response.response,
                        role=MessageRole.ASSISTANT,
                    )
                )
        else:
            # remove "Answer: " from the response, and anything before it
            start_idx = (latest_chunk.message.content or "").find("Answer:")
            if start_idx != -1 and latest_chunk.message.content:
                latest_chunk.message.content = latest_chunk.message.content[
                    start_idx + len("Answer:") :
                ].strip()

                # set delta to the content, minus the "Answer: "
                latest_chunk.delta = latest_chunk.message.content

            # add back the chunks that were missed
            response_stream = self._add_back_chunk_to_stream(
                chunks=[*missed_chunks_storage, latest_chunk], chat_stream=chat_stream
            )

            # Get the response in a separate thread so we can yield the response
            agent_response_stream = StreamingAgentChatResponse(
                chat_stream=response_stream,
                sources=task.extra_state["sources"],
            )
            thread = Thread(
                target=agent_response_stream.write_response_to_history,
                args=(task.extra_state["new_memory"],),
                kwargs={"on_stream_end_fn": partial(self.finalize_task, task)},
            )
            thread.start()

        response = agent_response_stream or non_streaming_agent_response
        assert response is not None

        return self._get_task_step_response(response, step, is_done)

    async def _arun_step_stream(
        self,
        step: TaskStep,
        task: Task,
    ) -> TaskStepOutput:
        """Run step."""
        if step.input is not None:
            add_user_step_to_reasoning(
                step,
                task.extra_state["new_memory"],
                task.extra_state["current_reasoning"],
                verbose=self._verbose,
            )
        # TODO: see if we want to do step-based inputs
        tools = self.get_tools(task.input)

        input_chat = self._react_chat_formatter.format(
            tools,
            chat_history=task.memory.get(input=task.input)
            + task.extra_state["new_memory"].get_all(),
            # Let LLM know that it can only take at most this many steps to complete the task.
            # - `//2`: An observation step also takes as an iteration.
            # - `-1`: The final output reasoning step needs to take a spot.
            allowance=(self._max_iterations - 1) // 2,
            current_reasoning=task.extra_state["current_reasoning"],
        )

        chat_stream = await self._llm.astream_chat(input_chat)

        # iterate over stream, break out if is final answer after the "Answer: "
        full_response = ChatResponse(
            message=ChatMessage(content=None, role="assistant")
        )
        missed_chunks_storage: List[ChatResponse] = []
        is_done = False
        async for latest_chunk in chat_stream:
            full_response = latest_chunk
            is_done = self._infer_stream_chunk_is_final(
                latest_chunk, missed_chunks_storage
            )
            if is_done:
                break

        non_streaming_agent_response = None
        agent_response_stream = None
        if not is_done:
            # given react prompt outputs, call tools or return response
            reasoning_steps, is_done = await self._aprocess_actions(
                task, tools=tools, output=full_response, is_streaming=True
            )
            task.extra_state["current_reasoning"].extend(reasoning_steps)
            # use _get_response to return intermediate response
            non_streaming_agent_response = self._get_response(
                task.extra_state["current_reasoning"], task.extra_state["sources"]
            )

            if is_done:
                non_streaming_agent_response.is_dummy_stream = True
                task.extra_state["new_memory"].put(
                    ChatMessage(
                        content=non_streaming_agent_response.response,
                        role=MessageRole.ASSISTANT,
                    )
                )
        else:
            # remove "Answer: " from the response, and anything before it
            start_idx = (latest_chunk.message.content or "").find("Answer:")
            if start_idx != -1 and latest_chunk.message.content:
                latest_chunk.message.content = latest_chunk.message.content[
                    start_idx + len("Answer:") :
                ].strip()

            # set delta to the content, minus the "Answer: "
            latest_chunk.delta = latest_chunk.message.content

            # add back the chunks that were missed
            response_stream = self._async_add_back_chunk_to_stream(
                chunks=[*missed_chunks_storage, latest_chunk], chat_stream=chat_stream
            )

            agent_response_stream = StreamingAgentChatResponse(
                achat_stream=response_stream,
                sources=task.extra_state["sources"],
            )
            # create task to write chat response to history
            agent_response_stream.awrite_response_to_history_task = asyncio.create_task(
                agent_response_stream.awrite_response_to_history(
                    task.extra_state["new_memory"],
                    on_stream_end_fn=partial(self.finalize_task, task),
                )
            )
            # wait until response writing is done
            agent_response_stream._ensure_async_setup()

            assert agent_response_stream.is_function_false_event is not None
            await agent_response_stream.is_function_false_event.wait()

        response = agent_response_stream or non_streaming_agent_response
        assert response is not None

        return self._get_task_step_response(response, step, is_done)

    @trace_method("run_step")
    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step."""
        return self._run_step(step, task)

    @trace_method("run_step")
    async def arun_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async)."""
        return await self._arun_step(step, task)

    @trace_method("run_step")
    def stream_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step (stream)."""
        # TODO: figure out if we need a different type for TaskStepOutput
        return self._run_step_stream(step, task)

    @trace_method("run_step")
    async def astream_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async stream)."""
        return await self._arun_step_stream(step, task)

    def finalize_task(self, task: Task, **kwargs: Any) -> None:
        """Finalize task, after all the steps are completed."""
        # add new messages to memory
        task.memory.set(
            task.memory.get_all() + task.extra_state["new_memory"].get_all()
        )
        # reset new memory
        task.extra_state["new_memory"].reset()

    def set_callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
        # TODO: make this abstractmethod (right now will break some agent impls)
        self.callback_manager = callback_manager
