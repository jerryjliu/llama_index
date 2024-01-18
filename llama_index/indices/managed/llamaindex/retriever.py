from typing import Any, Dict, List, Optional

from llama_index_client import TextNodeWithScore

from llama_index.core.base_retriever import BaseRetriever
from llama_index.indices.managed.llamaindex.utils import get_aclient, get_client
from llama_index.ingestion.pipeline import DEFAULT_PROJECT_NAME
from llama_index.schema import NodeWithScore, QueryBundle, TextNode


class PlatformRetriever(BaseRetriever):
    def __init__(
        self,
        name: str,
        project_name: str = DEFAULT_PROJECT_NAME,
        dense_similarity_top_k: Optional[int] = None,
        sparse_similarity_top_k: Optional[int] = None,
        enable_reranking: Optional[bool] = None,
        rerank_top_n: Optional[int] = None,
        alpha: Optional[float] = None,
        search_filters: Optional[Dict[str, List[Any]]] = None,
        platform_api_key: Optional[str] = None,
        platform_base_url: Optional[str] = None,
        platform_app_url: Optional[str] = None,
        timeout: int = 60,
        **kwargs: Any,
    ) -> None:
        """Initialize the Platform Retriever."""
        self.name = name
        self.project_name = project_name
        self._client = get_client(
            platform_api_key, platform_base_url, platform_app_url, timeout
        )
        self._aclient = get_aclient(
            platform_api_key, platform_base_url, platform_app_url, timeout
        )

        projects = self._client.project.list_projects(project_name=project_name)
        if len(projects) == 0:
            raise ValueError(f"No project found with name {project_name}")

        self._dense_similarity_top_k = dense_similarity_top_k
        self._sparse_similarity_top_k = sparse_similarity_top_k
        self._enable_reranking = enable_reranking
        self._rerank_top_n = rerank_top_n
        self._alpha = alpha
        self._search_filters = search_filters

        super().__init__(**kwargs)

    def _result_nodes_to_node_with_score(
        self, result_nodes: List[TextNodeWithScore]
    ) -> List[NodeWithScore]:
        nodes = []
        for res in result_nodes:
            text_node = TextNode.parse_obj(res.node.dict())
            nodes.append(NodeWithScore(node=text_node, score=res.score))

        return nodes

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve from the platform."""
        pipelines = self._client.pipeline.get_pipeline_by_name(
            pipeline_name=self.name, project_name=self.project_name
        )
        assert len(pipelines) == 1
        pipeline = pipelines[0]

        if pipeline.id is None:
            raise ValueError(
                f"No pipeline found with name {self.name} in project {self.project_name}"
            )

        # TODO: janky default values
        results = self._client.retrieval.run_search(
            query=query_bundle.query_str,
            pipeline_id=pipeline.id,
            dense_similarity_top_k=self._dense_similarity_top_k or 4,
            sparse_similarity_top_k=self._sparse_similarity_top_k or 4,
            enable_reranking=self._enable_reranking or True,
            rerank_top_n=self._rerank_top_n or 2,
            alpha=self._alpha or 0.5,
            search_filters=self._search_filters or {},
        )

        result_nodes = results.retrieval_nodes

        return self._result_nodes_to_node_with_score(result_nodes)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Asynchronously retrieve from the platform."""
        pipeline = await self._aclient.pipeline.get_pipeline_by_name(
            pipeline_name=self.name, project_name=self.project_name
        )
        if pipeline.id is None:
            raise ValueError(
                f"No pipeline found with name {self.name} in project {self.project_name}"
            )

        results = await self._aclient.retrieval.run_search(
            query=query_bundle.query_str,
            pipeline_id=pipeline.id,
            dense_similarity_top_k=self._dense_similarity_top_k,
            sparse_similarity_top_k=self._sparse_similarity_top_k,
            enable_reranking=self._enable_reranking,
            rerank_top_n=self._rerank_top_n,
            alpha=self._alpha,
            search_filters=self._search_filters,
        )

        result_nodes = results.retrieval_nodes

        return self._result_nodes_to_node_with_score(result_nodes)
