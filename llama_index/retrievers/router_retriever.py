"""Router retriever."""

import logging
from typing import List, Optional, Sequence
import asyncio

from llama_index.schema import NodeWithScore
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.service_context import ServiceContext
from llama_index.selectors.llm_selectors import LLMMultiSelector, LLMSingleSelector
from llama_index.selectors.pydantic_selectors import (
    PydanticMultiSelector,
    PydanticSingleSelector,
)
from llama_index.selectors.types import BaseSelector
from llama_index.tools.retriever_tool import RetrieverTool

logger = logging.getLogger(__name__)


class RouterRetriever(BaseRetriever):
    """Router retriever.

    Selects one (or multiple) out of several candidate retrievers to execute a query.

    Args:
        selector (BaseSelector): A selector that chooses one out of many options based
            on each candidate's metadata and query.
        retriever_tools (Sequence[RetrieverTool]): A sequence of candidate
            retrievers. They must be wrapped as tools to expose metadata to
            the selector.
        service_context (Optional[ServiceContext]): A service context.

    """

    def __init__(
        self,
        selector: BaseSelector,
        retriever_tools: Sequence[RetrieverTool],
        service_context: Optional[ServiceContext] = None,
    ) -> None:
        self.service_context = service_context or ServiceContext.from_defaults()
        self._selector = selector
        self._retrievers: List[BaseRetriever] = [x.retriever for x in retriever_tools]
        self._metadatas = [x.metadata for x in retriever_tools]
        self.callback_manager = self.service_context.callback_manager

    @classmethod
    def from_defaults(
        cls,
        retriever_tools: Sequence[RetrieverTool],
        service_context: Optional[ServiceContext] = None,
        selector: Optional[BaseSelector] = None,
        select_multi: bool = False,
    ) -> "RouterRetriever":
        if selector is None and select_multi:
            try:
                llm = service_context.llm_predictor.llm if service_context else None
                selector = PydanticMultiSelector.from_defaults(llm=llm)  # type: ignore
            except ValueError:
                selector = LLMMultiSelector.from_defaults(
                    service_context=service_context
                )
        elif selector is None and not select_multi:
            try:
                llm = service_context.llm_predictor.llm if service_context else None
                selector = PydanticSingleSelector.from_defaults(llm=llm)  # type: ignore
            except ValueError:
                selector = LLMSingleSelector.from_defaults(
                    service_context=service_context
                )

        assert selector is not None

        return cls(
            selector,
            retriever_tools,
            service_context=service_context,
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        with self.callback_manager.event(
            CBEventType.RETRIEVE,
            payload={EventPayload.QUERY_STR: query_bundle.query_str},
        ) as query_event:
            result = self._selector.select(self._metadatas, query_bundle)

            if len(result.inds) > 1:
                retrieved_results = {}
                for i, engine_ind in enumerate(result.inds):
                    logger.info(
                        f"Selecting retriever {engine_ind}: " f"{result.reasons[i]}."
                    )
                    selected_retriever = self._retrievers[engine_ind]
                    cur_results = selected_retriever.retrieve(query_bundle)
                    retrieved_results.update({n.node.node_id: n for n in cur_results})
            else:
                try:
                    selected_retriever = self._retrievers[result.ind]
                    logger.info(f"Selecting retriever {result.ind}: {result.reason}.")
                except ValueError as e:
                    raise ValueError("Failed to select retriever") from e

                cur_results = selected_retriever.retrieve(query_bundle)
                retrieved_results = {n.node.node_id: n for n in cur_results}

            query_event.on_end(payload={EventPayload.NODES: retrieved_results})

        return list(retrieved_results.values())

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        with self.callback_manager.event(
            CBEventType.RETRIEVE,
            payload={EventPayload.QUERY_STR: query_bundle.query_str},
        ) as query_event:
            result = await self._selector.aselect(self._metadatas, query_bundle)

            if len(result.inds) > 1:
                retrieved_results = {}
                tasks = []
                for i, engine_ind in enumerate(result.inds):
                    logger.info(
                        f"Selecting retriever {engine_ind}: " f"{result.reasons[i]}."
                    )
                    selected_retriever = self._retrievers[engine_ind]
                    tasks.append(selected_retriever.aretrieve(query_bundle))

                results_of_results = await asyncio.gather(*tasks)
                cur_results = [
                    item for sublist in results_of_results for item in sublist
                ]
                retrieved_results.update({n.node.node_id: n for n in cur_results})
            else:
                try:
                    selected_retriever = self._retrievers[result.ind]
                    logger.info(f"Selecting retriever {result.ind}: {result.reason}.")
                except ValueError as e:
                    raise ValueError("Failed to select retriever") from e

                cur_results = await selected_retriever.aretrieve(query_bundle)
                retrieved_results = {n.node.node_id: n for n in cur_results}

            query_event.on_end(payload={EventPayload.NODES: retrieved_results})

        return list(retrieved_results.values())
