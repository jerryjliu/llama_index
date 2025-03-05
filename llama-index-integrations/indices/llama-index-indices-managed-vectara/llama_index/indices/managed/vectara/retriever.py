"""
Vectara index.
An index that is built on top of Vectara.
"""

import json
import logging
from typing import Any, List, Optional, Tuple, Dict, Callable, Union
from enum import Enum

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.vector_store.retrievers.auto_retriever.auto_retriever import (
    VectorIndexAutoRetriever,
)
from llama_index.core.schema import NodeWithScore, QueryBundle, Node, MediaResource
from llama_index.core.types import TokenGen
from llama_index.core.base.response.schema import StreamingResponse

from llama_index.core.vector_stores.types import (
    FilterCondition,
    MetadataFilters,
    VectorStoreInfo,
    VectorStoreQuerySpec,
)
from llama_index.indices.managed.vectara.base import VectaraIndex
from llama_index.indices.managed.vectara.prompts import (
    DEFAULT_VECTARA_QUERY_PROMPT_TMPL,
)


_logger = logging.getLogger(__name__)


class VectaraReranker(str, Enum):
    NONE = "none"
    MMR = "mmr"
    SLINGSHOT = "multilingual_reranker_v1"
    SLINGSHOT_ALT_NAME = "slingshot"
    UDF = "userfn"
    CHAIN = "chain"


class VectaraRetriever(BaseRetriever):
    """
    Vectara Retriever.

    Args:
        index (VectaraIndex): the Vectara Index
        similarity_top_k (int): number of top k results to return, defaults to 5.
        offset (int): number of results to skip, defaults to 0.
        lambda_val (Union[List[float], float]): for hybrid search.
            0 = neural search only.
            1 = keyword match only.
            In between values are a linear interpolation.
            Provide single value for one corpus or a list of values for each corpus.
        semantics (Union[List[str], str]): Indicates whether the query is intended as a query or response.
            Provide single value for one corpus or a list of values for each corpus.
        custom_dimensions (Dict): Custom dimensions for the query.
            See (https://docs.vectara.com/docs/learn/semantic-search/add-custom-dimensions)
            for more details about usage.
            Provide single dict for one corpus or a list of dicts for each corpus.
        n_sentences_before (int):
            number of sentences before the matched sentence to return in the node
        n_sentences_after (int):
            number of sentences after the matched sentence to return in the node
        filter (Union[List[str], str]): metadata filter (if specified). Provide single string for one corpus
            or a list of strings to specify the filter for each corpus (if multiple corpora).
        reranker (str): reranker to use: none, mmr, slingshot/multilingual_reranker_v1, userfn, or chain.
        rerank_k (int): number of results to fetch for Reranking, defaults to 50.
        rerank_limit (int): maximum number of results to return after reranking, defaults to 50.
            Don't specify this for chain reranking. Instead, put the "limit" parameter in the dict for each individual reranker.
        rerank_cutoff (float): minimum score threshold for results to include after reranking, defaults to 0.
            Don't specify this for chain reranking. Instead, put the "chain" parameter in the dict for each individual reranker.
        mmr_diversity_bias (float): number between 0 and 1 that determines the degree
            of diversity among the results with 0 corresponding
            to minimum diversity and 1 to maximum diversity.
            Defaults to 0.3.
        udf_expression (str): the user defined expression for reranking results.
            See (https://docs.vectara.com/docs/learn/user-defined-function-reranker)
            for more details about syntax for udf reranker expressions.
        rerank_chain (List[Dict]): a list of rerankers to be applied in a sequence and their associated parameters
            for the chain reranker. Each element should specify the "type" of reranker (mmr, slingshot, userfn)
            and any other parameters (e.g. "limit" or "cutoff" for any type,  "diversity_bias" for mmr, and "user_function" for userfn).
            If using slingshot/multilingual_reranker_v1, it must be first in the list.
        summary_enabled (bool): whether to generate summaries or not. Defaults to False.
        summary_response_lang (str): language to use for summary generation.
        summary_num_results (int): number of results to use for summary generation.
        summary_prompt_name (str): name of the prompt to use for summary generation.
            To use Vectara's Mockingbird LLM designed specifically for RAG, set to "mockingbird-1.0-2024-07-16".
            If you are indexing documents with tables, we recommend "vectara-summary-table-query-ext-dec-2024-gpt-4o".
            See (https://docs.vectara.com/docs/learn/grounded-generation/select-a-summarizer) for all available prompts.
        prompt_text (str): the custom prompt, using appropriate prompt variables and functions.
            See (https://docs.vectara.com/docs/1.0/prompts/custom-prompts-with-metadata)
            for more details.
        max_response_chars (int): the desired maximum number of characters for the generated summary.
        max_tokens (int): the maximum number of tokens to be returned by the LLM.
        temperature (float): The sampling temperature; higher values lead to more randomness.
        frequency_penalty (float): How much to penalize repeating tokens in the response, reducing likelihood of repeating the same line.
        presence_penalty (float): How much to penalize repeating tokens in the response, increasing the diversity of topics.
        citations_style (str): The style of the citations in the summary generation,
            either "numeric", "html", "markdown", or "none". Defaults to None.
        citations_url_pattern (str): URL pattern for html and markdown citations.
            If non-empty, specifies the URL pattern to use for citations; e.g. "{doc.url}".
            See (https://docs.vectara.com/docs/api-reference/search-apis/search
                 #citation-format-in-summary) for more details. Defaults to None.
        citations_text_pattern (str): The displayed text for citations.
            If not specified, numeric citations are displayed for text.
        save_history (bool): Whether to save the query in history. Defaults to False.
    """

    def __init__(
        self,
        index: VectaraIndex,
        similarity_top_k: int = 10,
        offset: int = 0,
        lambda_val: Union[List[float], float] = 0.005,
        semantics: Union[List[str], str] = "default",
        custom_dimensions: Union[List[Dict], Dict] = {},
        n_sentences_before: int = 2,
        n_sentences_after: int = 2,
        filter: Union[List[str], str] = "",
        reranker: VectaraReranker = VectaraReranker.NONE,
        rerank_k: int = 50,
        rerank_limit: Optional[int] = None,
        rerank_cutoff: Optional[float] = None,
        mmr_diversity_bias: float = 0.3,
        udf_expression: str = None,
        rerank_chain: List[Dict] = None,
        summary_enabled: bool = False,
        summary_response_lang: str = "eng",
        summary_num_results: int = 7,
        summary_prompt_name: str = "vectara-summary-ext-24-05-med-omni",
        prompt_text: Optional[str] = None,
        max_response_chars: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        citations_style: Optional[str] = None,
        citations_url_pattern: Optional[str] = None,
        citations_text_pattern: Optional[str] = None,
        save_history: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        x_source_str: str = "llama_index",
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._index = index
        self._similarity_top_k = similarity_top_k
        self._offset = offset
        self._lambda_val = lambda_val
        self._semantics = semantics
        self._custom_dimensions = custom_dimensions
        self._n_sentences_before = n_sentences_before
        self._n_sentences_after = n_sentences_after
        self._filter = filter
        self._citations_style = citations_style
        self._citations_url_pattern = citations_url_pattern
        self._citations_text_pattern = citations_text_pattern
        self._save_history = save_history

        self._conv_id = None
        self._x_source_str = x_source_str

        if reranker in [
            VectaraReranker.MMR,
            VectaraReranker.SLINGSHOT,
            VectaraReranker.SLINGSHOT_ALT_NAME,
            VectaraReranker.UDF,
            VectaraReranker.CHAIN,
            VectaraReranker.NONE,
        ]:
            self._rerank = True
            self._reranker = reranker
            self._rerank_k = rerank_k
            self._rerank_limit = rerank_limit
            self._rerank_cutoff = rerank_cutoff

            if self._reranker == VectaraReranker.MMR:
                self._mmr_diversity_bias = mmr_diversity_bias

            elif self._reranker == VectaraReranker.UDF:
                self._udf_expression = udf_expression

            elif self._reranker == VectaraReranker.CHAIN:
                self._rerank_chain = rerank_chain
                for sub_reranker in self._rerank_chain:
                    if sub_reranker["type"] in [
                        VectaraReranker.SLINGSHOT,
                        VectaraReranker.SLINGSHOT_ALT_NAME,
                    ]:
                        sub_reranker["type"] = "customer_reranker"
                        sub_reranker["reranker_name"] = "Rerank_Multilingual_v1"

        else:
            self._rerank = False

        if summary_enabled:
            self._summary_enabled = True
            self._summary_response_lang = summary_response_lang
            self._summary_num_results = summary_num_results
            self._summary_prompt_name = summary_prompt_name
            self._prompt_text = prompt_text
            self._max_response_chars = max_response_chars
            self._max_tokens = max_tokens
            self._temperature = temperature
            self._frequency_penalty = frequency_penalty
            self._presence_penalty = presence_penalty

        else:
            self._summary_enabled = False
        super().__init__(callback_manager)

    def _get_post_headers(self) -> dict:
        """Returns headers that should be attached to each post request."""
        return {
            "x-api-key": self._index._vectara_api_key,
            "Content-Type": "application/json",
            "X-Source": self._x_source_str,
        }

    @property
    def similarity_top_k(self) -> int:
        """Return similarity top k."""
        return self._similarity_top_k

    @similarity_top_k.setter
    def similarity_top_k(self, similarity_top_k: int) -> None:
        """Set similarity top k."""
        self._similarity_top_k = similarity_top_k

    def _retrieve(
        self,
        query_bundle: QueryBundle,
        **kwargs: Any,
    ) -> List[NodeWithScore]:
        """
        Retrieve top k most similar nodes.

        Args:
            query_bundle: Query Bundle
        """
        return self._vectara_query(query_bundle, **kwargs)[0]  # return top_nodes only

    def _build_vectara_query_body(
        self,
        query_str: str,
        **kwargs: Any,
    ) -> Dict:
        data = {
            "query": query_str,
            "search": {
                "offset": self._offset,
                "limit": self._rerank_k if self._rerank else self._similarity_top_k,
                "context_configuration": {
                    "sentences_before": self._n_sentences_before,
                    "sentences_after": self._n_sentences_after,
                },
            },
        }

        corpora_config = [
            {"corpus_key": corpus_key}
            for corpus_key in self._index._vectara_corpus_key.split(",")
        ]

        for i in range(len(corpora_config)):
            corpora_config[i]["custom_dimensions"] = (
                self._custom_dimensions[i]
                if isinstance(self._custom_dimensions, list)
                else self._custom_dimensions
            )
            corpora_config[i]["metadata_filter"] = (
                self._filter[i] if isinstance(self._filter, list) else self._filter
            )
            corpora_config[i]["lexical_interpolation"] = (
                self._lambda_val[i]
                if isinstance(self._lambda_val, list)
                else self._lambda_val
            )
            corpora_config[i]["semantics"] = (
                self._semantics[i]
                if isinstance(self._semantics, list)
                else self._semantics
            )

        data["search"]["corpora"] = corpora_config

        if self._rerank:
            rerank_config = {}

            if self._reranker in [
                VectaraReranker.SLINGSHOT,
                VectaraReranker.SLINGSHOT_ALT_NAME,
            ]:
                rerank_config["type"] = "customer_reranker"
                rerank_config["reranker_name"] = "Rerank_Multilingual_v1"
            else:
                rerank_config["type"] = self._reranker

            if self._reranker == VectaraReranker.MMR:
                rerank_config["diversity_bias"] = self._mmr_diversity_bias

            elif self._reranker == VectaraReranker.UDF:
                rerank_config["user_function"] = self._udf_expression

            elif self._reranker == VectaraReranker.CHAIN:
                rerank_config["rerankers"] = self._rerank_chain

            if self._rerank_limit:
                rerank_config["limit"] = self._rerank_limit
            if self._rerank_cutoff:
                rerank_config["cutoff"] = self._rerank_cutoff

            data["search"]["reranker"] = rerank_config

        if self._summary_enabled:
            summary_config = {
                "response_language": self._summary_response_lang,
                "max_used_search_results": self._summary_num_results,
                "generation_preset_name": self._summary_prompt_name,
                "enable_factual_consistency_score": True,
            }
            if self._prompt_text:
                summary_config["prompt_template"] = self._prompt_text
            if self._max_response_chars:
                summary_config["max_response_characters"] = self._max_response_chars

            model_parameters = {}
            if self._max_tokens:
                model_parameters["max_tokens"] = self._max_tokens
            if self._temperature:
                model_parameters["temperature"] = self._temperature
            if self._frequency_penalty:
                model_parameters["frequency_penalty"] = self._frequency_penalty
            if self._presence_penalty:
                model_parameters["presence_penalty"] = self._presence_penalty

            if len(model_parameters) > 0:
                summary_config["model_parameters"] = model_parameters

            citations_config = {}
            if self._citations_style:
                if self._citations_style in ["numeric", "none"]:
                    citations_config["style"] = self._citations_style
                elif (
                    self._citations_style in ["html", "markdown"]
                    and self._citations_url_pattern
                ):
                    citations_config["style"] = self._citations_style
                    citations_config["url_pattern"] = self._citations_url_pattern
                    citations_config["text_pattern"] = self._citations_text_pattern
                else:
                    _logger.warning(
                        f"Invalid citations style {self._citations_style}. Must be one of 'numeric', 'html', 'markdown', or 'none'."
                    )

            if len(citations_config) > 0:
                summary_config["citations"] = citations_config

            data["generation"] = summary_config
            data["save_history"] = self._save_history

        return data

    def _vectara_stream(
        self,
        query_bundle: QueryBundle,
        chat: bool = False,
        conv_id: Optional[str] = None,
        verbose: bool = False,
        callback_func: Callable[[List, Dict], None] = None,
        **kwargs: Any,
    ) -> StreamingResponse:
        """
        Query Vectara index to get for top k most similar nodes.

        Args:
            query_bundle: Query Bundle
            chat: whether to use chat API in Vectara
            conv_id: conversation ID, if adding to existing chat
        """
        body = self._build_vectara_query_body(query_bundle.query_str)
        body["stream_response"] = True
        if verbose:
            print(f"Vectara streaming query request body: {body}")

        if chat:
            body["chat"] = {"store": True}
            if conv_id or self._conv_id:
                conv_id = conv_id or self._conv_id
                response = self._index._session.post(
                    headers=self._get_post_headers(),
                    url=f"{self._index._base_url}/v2/chats/{conv_id}/turns",
                    data=json.dumps(body),
                    timeout=self._index.vectara_api_timeout,
                    stream=True,
                )
            else:
                response = self._index._session.post(
                    headers=self._get_post_headers(),
                    url=f"{self._index._base_url}/v2/chats",
                    data=json.dumps(body),
                    timeout=self._index.vectara_api_timeout,
                    stream=True,
                )

        else:
            response = self._index._session.post(
                headers=self._get_post_headers(),
                url=f"{self._index._base_url}/v2/query",
                data=json.dumps(body),
                timeout=self._index.vectara_api_timeout,
                stream=True,
            )

        if response.status_code != 200:
            result = response.json()
            if response.status_code == 400:
                _logger.error(
                    f"Query failed (code {response.status_code}), reason {result['field_errors']}"
                )
            else:
                _logger.error(
                    f"Query failed (code {response.status_code}), reason {result['messages'][0]}"
                )
            return None

        def process_chunks(response):
            source_nodes = []
            response_metadata = {}

            def text_generator() -> TokenGen:
                for line in response.iter_lines():
                    line = line.decode("utf-8")
                    if line:
                        key, value = line.split(":", 1)
                        if key == "data":
                            line = json.loads(value)
                            if line["type"] == "generation_chunk":
                                yield line["generation_chunk"]

                            elif line["type"] == "factual_consistency_score":
                                response_metadata["fcs"] = line[
                                    "factual_consistency_score"
                                ]

                            elif line["type"] == "search_results":
                                search_results = line["search_results"]
                                source_nodes.extend(
                                    [
                                        NodeWithScore(
                                            node=Node(
                                                text_resource=MediaResource(
                                                    text=search_result["text"]
                                                ),
                                                id_=search_result["document_id"],
                                                metadata={
                                                    # Metadata from the matched part
                                                    **search_result.get(
                                                        "part_metadata", {}
                                                    ),
                                                    # Document-level metadata
                                                    "document": search_result.get(
                                                        "document_metadata", {}
                                                    ),
                                                },
                                            ),
                                            score=search_result["score"],
                                        )
                                        for search_result in search_results[
                                            : self._similarity_top_k
                                        ]
                                    ]
                                )

                            elif line["type"] == "chat_info":
                                self._conv_id = line["chat_id"]
                                response_metadata["chat_id"] = line["chat_id"]

                if callback_func:
                    callback_func(source_nodes, response_metadata)

            return text_generator(), source_nodes, response_metadata

        response_chunks, response_nodes, response_metadata = process_chunks(response)

        return StreamingResponse(
            response_gen=response_chunks,
            source_nodes=response_nodes,
            metadata=response_metadata,
        )

    def _vectara_query(
        self,
        query_bundle: QueryBundle,
        chat: bool = False,
        conv_id: Optional[str] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Tuple[List[NodeWithScore], Dict, str]:
        """
        Query Vectara index to get for top k most similar nodes.

        Args:
            query: Query Bundle
            chat: whether to use chat API in Vectara
            conv_id: conversation ID, if adding to existing chat
            verbose: whether to print verbose output (e.g. for debugging)
            Additional keyword arguments

        Returns:
            List[NodeWithScore]: list of nodes with scores
            Dict: summary
            str: conversation ID, if applicable
        """
        data = self._build_vectara_query_body(query_bundle.query_str)

        if verbose:
            print(f"Vectara query request body: {data}")

        if chat:
            data["chat"] = {"store": True}
            if conv_id:
                response = self._index._session.post(
                    headers=self._get_post_headers(),
                    url=f"{self._index._base_url}/v2/chats/{conv_id}/turns",
                    data=json.dumps(data),
                    timeout=self._index.vectara_api_timeout,
                )
            else:
                response = self._index._session.post(
                    headers=self._get_post_headers(),
                    url=f"{self._index._base_url}/v2/chats",
                    data=json.dumps(data),
                    timeout=self._index.vectara_api_timeout,
                )

        else:
            response = self._index._session.post(
                headers=self._get_post_headers(),
                url=f"{self._index._base_url}/v2/query",
                data=json.dumps(data),
                timeout=self._index.vectara_api_timeout,
            )

        result = response.json()
        if response.status_code != 200:
            if response.status_code == 400:
                _logger.error(
                    f"Query failed (code {response.status_code}), reason {result['field_errors']}"
                )
            else:
                _logger.error(
                    f"Query failed (code {response.status_code}), reason {result['messages'][0]}"
                )
            return [], {"text": ""}, ""

        if "warnings" in result:
            _logger.warning(f"Query warning(s) {(', ').join(result['warnings'])}")

        if verbose:
            print(f"Vectara query response: {result}")

        if self._summary_enabled:
            summary = {
                "text": result["answer"] if chat else result["summary"],
                "fcs": result.get("factual_consistency_score"),
            }
        else:
            summary = None

        search_results = result["search_results"]
        top_nodes = [
            NodeWithScore(
                node=Node(
                    text_resource=MediaResource(text=search_result["text"]),
                    id_=search_result["document_id"],
                    metadata={
                        # Metadata from the matched part
                        **search_result.get("part_metadata", {}),
                        # Document-level metadata
                        "document": search_result.get("document_metadata", {}),
                    },
                ),
                score=search_result["score"],
            )
            for search_result in search_results[: self._similarity_top_k]
        ]

        conv_id = result["chat_id"] if chat else None

        return top_nodes, summary, conv_id

    async def _avectara_query(
        self,
        query_bundle: QueryBundle,
        chat: bool = False,
        conv_id: Optional[str] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Tuple[List[NodeWithScore], Dict]:
        """
        Asynchronously query Vectara index to get for top k most similar nodes.

        Args:
            query: Query Bundle
            chat: whether to use chat API in Vectara
            conv_id: conversation ID, if adding to existing chat
            verbose: whether to print verbose output (e.g. for debugging)
            Additional keyword arguments

        Returns:
            List[NodeWithScore]: list of nodes with scores
            Dict: summary
        """
        return await self._vectara_query(query_bundle, chat, conv_id, verbose, **kwargs)


class VectaraAutoRetriever(VectorIndexAutoRetriever):
    """
    Managed Index auto retriever.

    A retriever for a Vectara index that uses an LLM to automatically set
    filtering query parameters.
    Based on VectorStoreAutoRetriever, and uses some of the vector_store
    types that are associated with auto retrieval.

    Args:
        index (VectaraIndex): Vectara Index instance
        vector_store_info (VectorStoreInfo): additional information about
            vector store content and supported metadata filters. The natural language
            description is used by an LLM to automatically set vector store query
            parameters.
        Other variables are the same as VectorStoreAutoRetriever or VectaraRetriever
    """

    def __init__(
        self,
        index: VectaraIndex,
        vector_store_info: VectorStoreInfo,
        **kwargs: Any,
    ) -> None:
        super().__init__(index, vector_store_info, prompt_template_str=DEFAULT_VECTARA_QUERY_PROMPT_TMPL, **kwargs)  # type: ignore
        self._index = index  # type: ignore
        self._kwargs = kwargs
        self._verbose = self._kwargs.get("verbose", False)
        self._explicit_filter = self._kwargs.pop("filter", "")

    def _build_retriever_from_spec(
        self, spec: VectorStoreQuerySpec
    ) -> Tuple[VectaraRetriever, QueryBundle]:
        query_bundle = self._get_query_bundle(spec.query)

        filter_list = [
            (filter.key, filter.operator.value, filter.value) for filter in spec.filters
        ]
        if self._verbose:
            print(f"Using query str: {spec.query}")
            print(f"Using implicit filters: {filter_list}")

        # create filter string from implicit filters
        if len(spec.filters) == 0:
            filter_str = ""
        else:
            filters = MetadataFilters(
                filters=[*spec.filters, *self._extra_filters.filters]
            )
            condition = " and " if filters.condition == FilterCondition.AND else " or "
            filter_str = condition.join(
                [
                    f"(doc.{f.key} {f.operator.value} '{f.value}')"
                    for f in filters.filters
                ]
            )

        # add explicit filter if specified
        if self._explicit_filter:
            if len(filter_str) > 0:
                filter_str = f"({filter_str}) and ({self._explicit_filter})"
            else:
                filter_str = self._explicit_filter

        if self._verbose:
            print(f"final filter string: {filter_str}")

        return (
            VectaraRetriever(
                index=self._index,  # type: ignore
                filter=filter_str,
                **self._kwargs,
            ),
            query_bundle,
        )

    def _vectara_query(
        self,
        query_bundle: QueryBundle,
        **kwargs: Any,
    ) -> Tuple[List[NodeWithScore], str]:
        spec = self.generate_retrieval_spec(query_bundle)
        vectara_retriever, new_query = self._build_retriever_from_spec(
            VectorStoreQuerySpec(
                query=spec.query, filters=spec.filters, top_k=self._similarity_top_k
            )
        )
        return vectara_retriever._vectara_query(new_query, **kwargs)
