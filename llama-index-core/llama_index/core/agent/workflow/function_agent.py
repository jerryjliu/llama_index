from typing import List

from llama_index.core.agent.workflow.base_agent import BaseWorkflowAgent
from llama_index.core.agent.workflow.workflow_events import (
    AgentInput,
    AgentOutput,
    AgentStream,
    ToolCallResult,
)
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import BaseMemory
from llama_index.core.tools import AsyncBaseTool
from llama_index.core.workflow import Context


class FunctionAgent(BaseWorkflowAgent):
    """Function calling agent implementation."""

    scratchpad_key: str = "scratchpad"

    async def take_step(
        self,
        ctx: Context,
        llm_input: List[ChatMessage],
        tools: List[AsyncBaseTool],
        memory: BaseMemory,
    ) -> AgentOutput:
        """Take a single step with the function calling agent."""
        if not self.llm.metadata.is_function_calling_model:
            raise ValueError("LLM must be a FunctionCallingLLM")

        scratchpad: List[ChatMessage] = await ctx.get(self.scratchpad_key, default=[])
        current_llm_input = [*llm_input, *scratchpad]

        ctx.write_event_to_stream(
            AgentInput(input=current_llm_input, current_agent_name=self.name)
        )

        response = await self.llm.astream_chat_with_tools(  # type: ignore
            tools, chat_history=current_llm_input, allow_parallel_tool_calls=True
        )
        async for r in response:
            tool_calls = self.llm.get_tool_calls_from_response(  # type: ignore
                r, error_on_no_tool_call=False
            )
            ctx.write_event_to_stream(
                AgentStream(
                    delta=r.delta or "",
                    response=r.message.content or "",
                    tool_calls=tool_calls or [],
                    raw=r.raw,
                    current_agent_name=self.name,
                )
            )

        tool_calls = self.llm.get_tool_calls_from_response(  # type: ignore
            r, error_on_no_tool_call=False
        )

        # only add to scratchpad if we didn't select the handoff tool
        if not any(tool_call.tool_name == "handoff" for tool_call in tool_calls):
            scratchpad.append(r.message)
            await ctx.set(self.scratchpad_key, scratchpad)

        return AgentOutput(
            response=r.message.content or "",
            tool_calls=tool_calls or [],
            raw=r.raw,
            current_agent_name=self.name,
        )

    async def handle_tool_call_results(
        self, ctx: Context, results: List[ToolCallResult], memory: BaseMemory
    ) -> None:
        """Handle tool call results for function calling agent."""
        scratchpad: List[ChatMessage] = await ctx.get(self.scratchpad_key, default=[])

        for tool_call_result in results:
            # don't add handoff tool calls to memory
            if tool_call_result.tool_name == "handoff":
                continue

            scratchpad.append(
                ChatMessage(
                    role="tool",
                    content=str(tool_call_result.tool_output.content),
                    additional_kwargs={"tool_call_id": tool_call_result.tool_id},
                )
            )

            if tool_call_result.return_direct:
                scratchpad.append(
                    ChatMessage(
                        role="assistant",
                        content=str(tool_call_result.tool_output.content),
                        additional_kwargs={"tool_call_id": tool_call_result.tool_id},
                    )
                )
                break

        await ctx.set(self.scratchpad_key, scratchpad)

    async def finalize(
        self, ctx: Context, output: AgentOutput, memory: BaseMemory
    ) -> AgentOutput:
        """Finalize the function calling agent.

        Adds all in-progress messages to memory.
        """
        scratchpad: List[ChatMessage] = await ctx.get(self.scratchpad_key, default=[])
        for msg in scratchpad:
            await memory.aput(msg)

        return output
