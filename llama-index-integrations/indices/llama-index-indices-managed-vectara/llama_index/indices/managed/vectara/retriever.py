"""
Vectara index.
An index that is built on top of Vectara.
"""

import json
import logging
from typing import Any, List, Optional, Tuple, Dict
from enum import Enum
import urllib.parse

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.vector_store.retrievers.auto_retriever.auto_retriever import (
    VectorIndexAutoRetriever,
)
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.types import TokenGen
from llama_index.core.llms import (
    CompletionResponse,
)
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


# CHAIN_RERANKER_NAMES = {
#     VectaraReranker.MMR: "Maximum Marginal Relevance Reranker",
#     VectaraReranker.SLINGSHOT: "Rerank_Multilingual_v1",
#     VectaraReranker.SLINGSHOT_ALT_NAME: "Rerank_Multilingual_v1",
#     VectaraReranker.UDF: "User_Defined_Function_Reranker",
# }


class VectaraRetriever(BaseRetriever):
    """
    Vectara Retriever.

    Args:
        index (VectaraIndex): the Vectara Index
        similarity_top_k (int): number of top k results to return, defaults to 5.
        offset (int): number of results to skip, defaults to 0.
        lambda_val (float): for hybrid search.
            0 = neural search only.
            1 = keyword match only.
            In between values are a linear interpolation
        semantics (str): Indicates whether the query is intended as a query or response.
        custom_dimensions (Dict): Custom dimensions for the query.
            See (https://docs.vectara.com/docs/learn/semantic-search/add-custom-dimensions)
            for more details about usage.
        n_sentences_before (int):
            number of sentences before the matched sentence to return in the node
        n_sentences_after (int):
            number of sentences after the matched sentence to return in the node
        filter (str): metadata filter (if specified)
        reranker (str): reranker to use: none, mmr, slingshot/multilingual_reranker_v1, userfn, or chain.
            Note that "multilingual_reranker_v1" is a Vectara Scale feature only.
        rerank_k (int): number of results to fetch for Reranking, defaults to 50.
        rerank_limit (int): maximum number of results to return after reranking, defaults to 50.
        rerank_cutoff (float): minimum score threshold for results to include after reranking, defaults to 0.
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
        max_response_chars (int): the desired maximum number of characters for the generated summary.
        max_tokens (int): the maximum number of tokens to be returned by the LLM.
        temperature (float): The sampling temperature; higher values lead to more randomness.
        frequency_penalty (float): How much to penalize repeating tokens in the response, reducing likelihood of repeating the same line.
        presence_penalty (float): How much to penalize repeating tokens in the response, increasing the diversity of topics.
        prompt_text (str): the custom prompt, using appropriate prompt variables and functions.
            See (https://docs.vectara.com/docs/1.0/prompts/custom-prompts-with-metadata)
            for more details.
        citations_style (str): The style of the citations in the summary generation,
            either "numeric", "html", "markdown", or "none".
            This is a Vectara Scale only feature. Defaults to None.
        citations_url_pattern (str): URL pattern for html and markdown citations.
            If non-empty, specifies the URL pattern to use for citations; e.g. "{doc.url}".
            See (https://docs.vectara.com/docs/api-reference/search-apis/search
                 #citation-format-in-summary) for more details.
            This is a Vectara Scale only feature. Defaults to None.
        citations_text_pattern (str): The displayed text for citations.
            If not specified, numeric citations are displayed for text.
    """

    def __init__(
        self,
        index: VectaraIndex,
        similarity_top_k: int = 10,
        offset: int = 0,
        lambda_val: float = 0.005,
        semantics: str = "default",
        custom_dimensions: Dict = {},
        n_sentences_before: int = 2,
        n_sentences_after: int = 2,
        filter: str = "",
        reranker: VectaraReranker = VectaraReranker.NONE,
        rerank_k: int = 50,
        rerank_limit: int = 50,
        rerank_cutoff: float = 0,
        mmr_diversity_bias: float = 0.3,
        udf_expression: str = None,
        rerank_chain: List[Dict] = None,
        summary_enabled: bool = False,
        summary_response_lang: str = "eng",
        summary_num_results: int = 7,
        summary_prompt_name: str = "vectara-summary-ext-24-05-sml",
        prompt_text: Optional[str] = None,
        max_response_chars: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        citations_style: Optional[str] = None,
        citations_url_pattern: Optional[str] = None,
        citations_text_pattern: Optional[str] = None,
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
        self._prompt_text = prompt_text
        self._citations_style = citations_style.upper() if citations_style else None
        self._citations_url_pattern = citations_url_pattern
        self._citations_text_pattern = citations_text_pattern
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

            if (
                self._reranker != VectaraReranker.NONE
                and self._reranker != VectaraReranker.CHAIN
            ):
                self._rerank_cutoff = rerank_cutoff

        else:
            self._rerank = False

        if summary_enabled:
            self._summary_enabled = True
            self._summary_response_lang = summary_response_lang
            self._summary_num_results = summary_num_results
            self._summary_prompt_name = summary_prompt_name
            self._max_response_chars = max_response_chars
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
            query: Query Bundle
        """
        return self._vectara_query(query_bundle, **kwargs)[0]  # return top_nodes only

    # For now we are just doing a single corpus query, but we may want to be able to support more.
    # This becomes a little more complicated because for each corpus, we would need to have the user pass in parameters for each corpus,
    # such as metadata_filter, lexical_interpolation, semantics, etc.
    def _build_vectara_query_body(
        self,
        query_str: str,
        chat: bool = False,
        chat_conv_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict:
        data = {
            "query": query_str,
            "search": {
                "custom_dimensions": self._custom_dimensions,
                "metadata_filter": self._filter,
                "lexical_interpolation": self._lambda_val,
                "semantics": self._semantics,
                "offset": self._offset,
                "limit": self._similarity_top_k,
                "context_configuration": {
                    "sentences_before": self._n_sentences_before,
                    "sentences_after": self._n_sentences_after,
                },
            },
        }

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

            if (
                self._reranker != VectaraReranker.NONE
                and self._reranker != VectaraReranker.CHAIN
            ):
                rerank_config["cutoff"] = self._rerank_cutoff

            rerank_config["limit"] = self._rerank_limit
            data["search"]["reranker"] = rerank_config

        if self._summary_enabled:
            summary_config = {
                "response_language": self._summary_response_lang,
                "max_used_search_results": self._summary_num_results,
                "generation_preset_name": self._summary_prompt_name,
            }
            if self._prompt_text:
                summary_config["prompt_template"] = self._prompt_text
            if self._max_response_characters:
                summary_config[
                    "max_response_characters"
                ] = self._max_response_characters

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
                summary_config["model_parameters"] = model_paramters

            citations_config = {}
            if self._citations_style:
                if self._citations_style in ["NUMERIC", "NONE"]:
                    citations_config["style"] = self._citations_style

                elif self._citations_url_pattern:
                    citations_config["style"] = self._citations_style
                    citations_config["url_pattern"] = self._citations_url_pattern
                    citations_config["text_pattern"] = self._citations_text_pattern

            if len(citations_config) > 0:
                summary_config["citations"] = citations_config

            data["generation"] = summary_config

            ## NEED TO FIGURE OUT HOW CHAT WORKS IN APIV2
            if chat:
                data["query"][0]["summary"][0]["chat"] = {
                    "store": True,
                    "conversationId": chat_conv_id,
                }

        return data

    def _vectara_stream(
        self,
        query_bundle: QueryBundle,
        chat: bool = False,
        conv_id: Optional[str] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> TokenGen:
        """
        Query Vectara index to get for top k most similar nodes.

        Args:
            query_bundle: Query Bundle
            chat: whether to enable chat
            conv_id: conversation ID, if chat enabled
        """
        body = self._build_vectara_query_body(query_bundle.query_str)
        if verbose:
            print(f"Vectara streaming query request body: {body}")
        response = self._index._session.post(
            headers=self._get_post_headers(),
            url="https://api.vectara.io/v1/stream-query",
            data=json.dumps(body),
            timeout=self._index.vectara_api_timeout,
            stream=True,
        )

        if response.status_code != 200:
            print(
                "Query failed %s",
                f"(code {response.status_code}, reason {response.reason}, details "
                f"{response.text})",
            )
            return

        responses = []
        documents = []
        stream_response = CompletionResponse(
            text="", additional_kwargs={"fcs": None}, raw=None, delta=None
        )

        for line in response.iter_lines():
            if line:  # filter out keep-alive new lines
                data = json.loads(line.decode("utf-8"))
                result = data["result"]
                response_set = result["responseSet"]
                if response_set is None:
                    summary = result.get("summary", None)
                    if summary is None:
                        continue
                    if len(summary.get("status")) > 0:
                        print(
                            f"Summary generation failed with status {summary.get('status')[0].get('statusDetail')}"
                        )
                        continue

                    # Store conversation ID for chat, if applicable
                    chat = summary.get("chat", None)
                    if chat and chat.get("status", None):
                        st_code = chat["status"]
                        print(f"Chat query failed with code {st_code}")
                        if st_code == "RESOURCE_EXHAUSTED":
                            self.conv_id = None
                            print("Sorry, Vectara chat turns exceeds plan limit.")
                            continue

                    conv_id = chat.get("conversationId", None) if chat else None
                    if conv_id:
                        self.conv_id = conv_id

                    # if factual consistency score is provided, pull that from the JSON response
                    if summary.get("factualConsistency", None):
                        fcs = summary.get("factualConsistency", {}).get("score", None)
                        stream_response.additional_kwargs["fcs"] = fcs
                        continue

                    # Yield the summary chunk
                    chunk = urllib.parse.unquote(summary["text"])
                    stream_response.text += chunk
                    stream_response.delta = chunk
                    yield stream_response
                else:
                    metadatas = []
                    for x in responses:
                        md = {m["name"]: m["value"] for m in x["metadata"]}
                        doc_num = x["documentIndex"]
                        doc_md = {
                            m["name"]: m["value"]
                            for m in documents[doc_num]["metadata"]
                        }
                        md.update(doc_md)
                        metadatas.append(md)

                    top_nodes = []
                    for x, md in zip(responses, metadatas):
                        doc_inx = x["documentIndex"]
                        doc_id = documents[doc_inx]["id"]
                        node = NodeWithScore(
                            node=TextNode(text=x["text"], id_=doc_id, metadata=md), score=x["score"]  # type: ignore
                        )
                        top_nodes.append(node)
                    stream_response.additional_kwargs["top_nodes"] = top_nodes[
                        : self._similarity_top_k
                    ]
                    stream_response.delta = None
                    yield stream_response
        return

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
            chat: whether to enable chat in Vectara
            conv_id: conversation ID, if chat enabled
            verbose: whether to print verbose output (e.g. for debugging)
            Additional keyword arguments

        Returns:
            List[NodeWithScore]: list of nodes with scores
            Dict: summary
            str: conversation ID, if applicable
        """
        data = self._build_vectara_query_body(query_bundle.query_str, chat, conv_id)

        if verbose:
            print(f"Vectara query request body: {data}")
        response = self._index._session.post(
            headers=self._get_post_headers(),
            url="https://api.vectara.io/v1/query",
            data=json.dumps(data),
            timeout=self._index.vectara_api_timeout,
        )

        if response.status_code != 200:
            _logger.error(
                "Query failed %s",
                f"(code {response.status_code}, reason {response.reason}, details "
                f"{response.text})",
            )
            return [], {"text": ""}, ""

        result = response.json()
        if verbose:
            print(f"Vectara query response: {result}")
        status = result["responseSet"][0]["status"]
        if len(status) > 0 and status[0]["code"] != "OK":
            _logger.error(
                f"Query failed (code {status[0]['code']}, msg={status[0]['statusDetail']}"
            )
            return [], {"text": ""}, ""

        responses = result["responseSet"][0]["response"]
        documents = result["responseSet"][0]["document"]

        if self._summary_enabled:
            summaryJson = result["responseSet"][0]["summary"][0]
            if len(summaryJson["status"]) > 0:
                print(
                    f"Summary generation failed with error: '{summaryJson['status'][0]['statusDetail']}'"
                )
                return [], {"text": ""}, ""

            summary = {
                "text": (
                    urllib.parse.unquote(summaryJson["text"])
                    if self._summary_enabled
                    else None
                ),
                "fcs": summaryJson["factualConsistency"]["score"],
            }
            if summaryJson.get("chat", None):
                conv_id = summaryJson["chat"]["conversationId"]
            else:
                conv_id = None
        else:
            summary = None

        metadatas = []
        for x in responses:
            md = {m["name"]: m["value"] for m in x["metadata"]}
            doc_num = x["documentIndex"]
            doc_md = {m["name"]: m["value"] for m in documents[doc_num]["metadata"]}
            md.update(doc_md)
            metadatas.append(md)

        top_nodes = []
        for x, md in zip(responses, metadatas):
            doc_inx = x["documentIndex"]
            doc_id = documents[doc_inx]["id"]
            node = NodeWithScore(
                node=TextNode(text=x["text"], id_=doc_id, metadata=md), score=x["score"]  # type: ignore
            )
            top_nodes.append(node)

        return top_nodes[: self._similarity_top_k], summary, conv_id

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
            chat: whether to enable chat in Vectara
            conv_id: conversation ID, if chat enabled
            verbose: whether to print verbose output (e.g. for debugging)
            Additional keyword arguments

        Returns:
            List[NodeWithScore]: list of nodes with scores
            Dict: summary
            str: conversation ID, if applicable
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
