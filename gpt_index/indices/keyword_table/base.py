"""Keyword-table based index.

Similar to a "hash table" in concept. LlamaIndex first tries
to extract keywords from the source text, and stores the
keywords as keys per item. It similarly extracts keywords
from the query text. Then, it tries to match those keywords to
existing keywords in the table.

"""

from abc import abstractmethod
from typing import Any, Dict, Optional, Sequence, Set, Type

from gpt_index.async_utils import run_async_tasks
from gpt_index.data_structs.data_structs import Node
from gpt_index.data_structs.data_structs_v2 import KeywordTable
from gpt_index.indices.base import BaseGPTIndex
from gpt_index.indices.keyword_table.utils import extract_keywords_given_response
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.keyword_table.query import (
    GPTKeywordTableGPTQuery,
    GPTKeywordTableRAKEQuery,
    GPTKeywordTableSimpleQuery,
)
from gpt_index.indices.query.schema import QueryMode
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.node_parser.interface import NodeParser
from gpt_index.prompts.default_prompts import (
    DEFAULT_KEYWORD_EXTRACT_TEMPLATE,
    DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE,
)
from gpt_index.prompts.prompts import KeywordExtractPrompt

DQKET = DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE


class BaseGPTKeywordTableIndex(BaseGPTIndex[KeywordTable]):
    """GPT Keyword Table Index.

    This index extracts keywords from the text, and maps each
    keyword to the node(s) that it corresponds to. In this sense it mimicks a
    "hash table". During index construction, the keyword table is constructed
    by extracting keywords from each node and creating an internal mapping.

    During query time, the keywords are extracted from the query text, and these
    keywords are used to index into the keyword table. The retrieved nodes
    are then used to answer the query.

    Args:
        keyword_extract_template (Optional[KeywordExtractPrompt]): A Keyword
            Extraction Prompt
            (see :ref:`Prompt-Templates`).
        use_async (bool): Whether to use asynchronous calls. Defaults to False.

    """

    index_struct_cls = KeywordTable

    def __init__(
        self,
        nodes: Optional[Sequence[Node]] = None,
        index_struct: Optional[KeywordTable] = None,
        keyword_extract_template: Optional[KeywordExtractPrompt] = None,
        max_keywords_per_chunk: int = 10,
        llm_predictor: Optional[LLMPredictor] = None,
        node_parser: Optional[NodeParser] = None,
        use_async: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        # need to set parameters before building index in base class.
        self.max_keywords_per_chunk = max_keywords_per_chunk
        self.keyword_extract_template = (
            keyword_extract_template or DEFAULT_KEYWORD_EXTRACT_TEMPLATE
        )
        # NOTE: Partially format keyword extract template here.
        self.keyword_extract_template = self.keyword_extract_template.partial_format(
            max_keywords=self.max_keywords_per_chunk
        )
        self._use_async = use_async
        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            llm_predictor=llm_predictor,
            node_parser=node_parser,
            **kwargs,
        )

    @classmethod
    def get_query_map(self) -> Dict[str, Type[BaseGPTIndexQuery]]:
        """Get query map."""
        return {
            QueryMode.DEFAULT: GPTKeywordTableGPTQuery,
            QueryMode.SIMPLE: GPTKeywordTableSimpleQuery,
            QueryMode.RAKE: GPTKeywordTableRAKEQuery,
        }

    @abstractmethod
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""

    async def _async_extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        # by default just call sync version
        return self._extract_keywords(text)

    def _add_nodes_to_index(
        self, index_struct: KeywordTable, nodes: Sequence[Node]
    ) -> None:
        """Add document to index."""
        for n in nodes:
            keywords = self._extract_keywords(n.get_text())
            index_struct.add_node(list(keywords), n)
            self._docstore.add_documents([n])

    async def _async_add_nodes_to_index(
        self, index_struct: KeywordTable, nodes: Sequence[Node]
    ) -> None:
        """Add document to index."""
        for n in nodes:
            keywords = await self._async_extract_keywords(n.get_text())
            index_struct.add_node(list(keywords), n)
            self._docstore.add_documents([n])

    def _build_index_from_nodes(
        self, nodes: Sequence[Node]
    ) -> KeywordTable:
        """Build the index from documents."""
        # do simple concatenation
        index_struct = KeywordTable(table={})
        if self._use_async:
            tasks = [
                self._async_add_nodes_to_index(index_struct, nodes)
            ]
            run_async_tasks(tasks)
        else:
            self._add_nodes_to_index(index_struct, nodes)

        return index_struct

    def _insert(self, nodes: Sequence[Node], **insert_kwargs: Any) -> None:
        """Insert nodes."""
        for n in nodes:
            keywords = self._extract_keywords(n.get_text())
            self._index_struct.add_node(list(keywords), n)
            self._docstore.add_documents([n])

    def _delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document."""
        # get set of ids that correspond to node
        node_idxs_to_delete = set()
        node_id_list = list(self._index_struct.node_ids)
        nodes = self._docstore.get_nodes(node_id_list)
        for node_idx, node in zip(node_id_list, nodes):
            if node.ref_doc_id != doc_id:
                continue
            node_idxs_to_delete.add(node_idx)
        for node_idx in node_idxs_to_delete:
            self._docstore.delete_document(node_idx)

        # delete node_idxs from keyword to node idxs mapping
        keywords_to_delete = set()
        for keyword, node_idxs in self._index_struct.table.items():
            if node_idxs_to_delete.intersection(node_idxs):
                self._index_struct.table[keyword] = node_idxs.difference(
                    node_idxs_to_delete
                )
                if not self._index_struct.table[keyword]:
                    keywords_to_delete.add(keyword)

        for keyword in keywords_to_delete:
            del self._index_struct.table[keyword]


class GPTKeywordTableIndex(BaseGPTKeywordTableIndex):
    """GPT Keyword Table Index.

    This index uses a GPT model to extract keywords from the text.

    """

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        response, _ = self._llm_predictor.predict(
            self.keyword_extract_template,
            text=text,
        )
        keywords = extract_keywords_given_response(response, start_token="KEYWORDS:")
        return keywords

    async def _async_extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        response, _ = await self._llm_predictor.apredict(
            self.keyword_extract_template,
            text=text,
        )
        keywords = extract_keywords_given_response(response, start_token="KEYWORDS:")
        return keywords
