"""ErsatzO1 Query Engine.

This module implements the ErsatzO1 approach, combining Chain of Thought (CoT) with
re-reading technique and SELF-CONSISTENCY prompting for generic text input.

The approach is adapted from the concepts presented in various papers on prompting techniques.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core import PromptTemplate
from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.core.llms import LLM
from llama_index.core.llms.utils import LLMType, resolve_llm
from llama_index.core.output_parsers import ChainableOutputParser
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.query_pipeline import QueryPipeline
from pydantic import Field

text_prompt_str = """\
You are an advanced AI implementing the ErsatzO1 approach, capable of analyzing and understanding information within text. Read the following text:

{context}

Based on the given text, answer the following question:

{question}

Read the question again: {question}

Let's think step by step:
1) [Your first step of reasoning]
2) [Your second step of reasoning]
3) [Continue with more steps as needed]

Final Answer: Provide your answer here. Ensure it is as concise as possible, without any explanation.

Confidence: Rate your confidence in the answer from 1 (low) to 5 (high).

Your response should be in the exact format:

<answer>
<confidence_value>

Replace <answer> with your actual answer and <confidence_value> with your actual confidence rating.
"""
text_prompt = PromptTemplate(template=text_prompt_str)


class FinalAnswerOutputParser(ChainableOutputParser):
    """Output parser for the ErsatzO1 approach."""

    def parse(self, output: str) -> Optional[str]:
        lines = output.strip().split("\n")
        if len(lines) >= 2:
            return output.strip()  # Return the entire formatted output
        return None

    def format(self, query: Any) -> str:
        return query


def parse_response(response: str) -> Tuple[str, int]:
    """Parse the response from the LLM in the ErsatzO1 approach."""
    lines = response.strip().split("\n")
    if len(lines) != 2:
        raise ValueError("Response format is incorrect")

    answer = lines[0].strip("'")  # Remove surrounding quotes if present
    confidence = int(lines[1].strip())

    return answer, confidence


async def async_textual_reasoning(
    context: str,
    query_str: str,
    llm: LLM,
    num_paths: int = 5,
    verbose: bool = False,
    temperature: float = 1,
) -> List[Tuple[str, int]]:
    """
    Perform asynchronous textual reasoning using the ErsatzO1 approach.

    This function generates multiple reasoning paths and returns a list of answers with confidence scores.
    """
    llm.temperature = temperature
    output_parser = FinalAnswerOutputParser()

    results = []
    for _ in range(num_paths):
        chain = QueryPipeline(chain=[text_prompt, llm, output_parser], verbose=verbose)
        response = await chain.arun(
            question=query_str,
            context=context,
        )
        try:
            answer, confidence = parse_response(str(response))
            results.append((answer, confidence))
        except IndexError:
            print(f"Error parsing response: {response}")

    return results


def aggregate_textual_results(results: List[Tuple[str, int]]) -> str:
    """
    Aggregate results from multiple reasoning paths in the ErsatzO1 approach.

    This function selects the most common answer, weighted by confidence scores.
    """
    answer_counts = {}
    for answer, confidence in results:
        if answer in answer_counts:
            answer_counts[answer] += confidence
        else:
            answer_counts[answer] = confidence

    return max(answer_counts, key=answer_counts.get)


class ErsatzO1QueryEngine(CustomQueryEngine):
    context: str = Field(..., description="Input text context.")
    llm: LLM = Field(..., description="LLM to use.")
    verbose: bool = Field(
        default=False, description="Whether to print debug information."
    )
    reasoning_paths: int = Field(default=5, description="Number of reasoning paths.")

    def __init__(
        self,
        context: str,
        llm: Optional[LLMType] = None,
        verbose: bool = False,
        reasoning_paths: int = 5,
        **kwargs: Any,
    ):
        llm = resolve_llm(llm)

        super().__init__(
            context=context,
            llm=llm,
            verbose=verbose,
            reasoning_paths=reasoning_paths,
            **kwargs,
        )

    def custom_query(self, query_str: str) -> str:
        results = asyncio.run(
            async_textual_reasoning(
                self.context,
                query_str,
                self.llm,
                self.reasoning_paths,
                self.verbose,
            )
        )
        return aggregate_textual_results(results)

    async def acustom_query(self, query_str: str) -> str:
        results = await async_textual_reasoning(
            self.context,
            query_str,
            self.llm,
            self.reasoning_paths,
            self.verbose,
        )
        return aggregate_textual_results(results)


class ErsatzO1Pack(BaseLlamaPack):
    """ErsatzO1 Pack for generic text input."""

    def __init__(
        self,
        context: str,
        llm: Optional[LLMType] = None,
        verbose: bool = False,
        reasoning_paths: int = 5,
        **kwargs: Any,
    ) -> None:
        self.query_engine = ErsatzO1QueryEngine(
            context=context,
            llm=llm,
            verbose=verbose,
            reasoning_paths=reasoning_paths,
            **kwargs,
        )

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "query_engine": self.query_engine,
            "llm": self.query_engine.llm,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return self.query_engine.query(*args, **kwargs)
