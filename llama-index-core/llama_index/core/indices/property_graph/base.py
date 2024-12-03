import asyncio
from typing import Any, Dict, List, Optional, Sequence, Type, TYPE_CHECKING

from llama_index.core.data_structs import IndexLPG
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms import LLM
from llama_index.core.embeddings.utils import EmbedType, resolve_embed_model
from llama_index.core.callbacks import CallbackManager
from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import (
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    VECTOR_SOURCE_KEY,
)
from llama_index.core.vector_stores.simple import DEFAULT_VECTOR_STORE
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices.property_graph.transformations import (
    SimpleLLMPathExtractor,
    ImplicitPathExtractor,
)
from llama_index.core.ingestion.pipeline import (
    run_transformations,
    arun_transformations,
)
from llama_index.core.graph_stores.types import (
    LabelledNode,
    Relation,
    PropertyGraphStore,
    TRIPLET_SOURCE_KEY,
)
from llama_index.core.storage.docstore.types import RefDocInfo
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.schema import BaseNode, MetadataMode, TextNode, TransformComponent
from llama_index.core.settings import Settings
from llama_index.core.vector_stores.types import BasePydanticVectorStore

if TYPE_CHECKING:
    from llama_index.core.indices.property_graph.sub_retrievers.base import (
        BasePGRetriever,
    )


class PropertyGraphIndex(BaseIndex[IndexLPG]):
    """An index for a property graph.

    Args:
        nodes (Optional[Sequence[BaseNode]]):
            A list of nodes to insert into the index.
        llm (Optional[LLM]):
            The language model to use for extracting triplets. Defaults to `Settings.llm`.
        kg_extractors (Optional[List[TransformComponent]]):
            A list of transformations to apply to the nodes to extract triplets.
            Defaults to `[SimpleLLMPathExtractor(llm=llm), ImplicitEdgeExtractor()]`.
        property_graph_store (Optional[PropertyGraphStore]):
            The property graph store to use. If not provided, a new `SimplePropertyGraphStore` will be created.
        vector_store (Optional[BasePydanticVectorStore]):
            The vector store index to use, if the graph store does not support vector queries.
        use_async (bool):
            Whether to use async for transformations. Defaults to `True`.
        embed_model (Optional[EmbedType]):
            The embedding model to use for embedding nodes.
            If not provided, `Settings.embed_model` will be used if `embed_kg_nodes=True`.
        embed_kg_nodes (bool):
            Whether to embed the KG nodes. Defaults to `True`.
        callback_manager (Optional[CallbackManager]):
            The callback manager to use.
        transformations (Optional[List[TransformComponent]]):
            A list of transformations to apply to the nodes before inserting them into the index.
            These are applied prior to the `kg_extractors`.
        storage_context (Optional[StorageContext]):
            The storage context to use.
        show_progress (bool):
            Whether to show progress bars for transformations. Defaults to `False`.
    """

    index_struct_cls = IndexLPG

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        llm: Optional[LLM] = None,
        kg_extractors: Optional[List[TransformComponent]] = None,
        property_graph_store: Optional[PropertyGraphStore] = None,
        # vector related params
        vector_store: Optional[BasePydanticVectorStore] = None,
        use_async: bool = True,
        embed_model: Optional[EmbedType] = None,
        embed_kg_nodes: bool = True,
        # parent class params
        callback_manager: Optional[CallbackManager] = None,
        transformations: Optional[List[TransformComponent]] = None,
        storage_context: Optional[StorageContext] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        storage_context = storage_context or StorageContext.from_defaults(
            property_graph_store=property_graph_store
        )

        # lazily initialize the graph store on the storage context
        if property_graph_store is not None:
            storage_context.property_graph_store = property_graph_store
        elif storage_context.property_graph_store is None:
            storage_context.property_graph_store = SimplePropertyGraphStore()

        if vector_store is not None:
            storage_context.vector_stores[DEFAULT_VECTOR_STORE] = vector_store

        if embed_kg_nodes and (
            storage_context.property_graph_store.supports_vector_queries
            or embed_kg_nodes
        ):
            self._embed_model = (
                resolve_embed_model(embed_model)
                if embed_model
                else Settings.embed_model
            )
        else:
            self._embed_model = None  # type: ignore

        self._kg_extractors = kg_extractors or [
            SimpleLLMPathExtractor(llm=llm or Settings.llm),
            ImplicitPathExtractor(),
        ]
        self._use_async = use_async
        self._llm = llm
        self._embed_kg_nodes = embed_kg_nodes
        self._override_vector_store = (
            vector_store is not None
            or not storage_context.property_graph_store.supports_vector_queries
        )

        super().__init__(
            nodes=nodes,
            callback_manager=callback_manager,
            storage_context=storage_context,
            transformations=transformations,
            show_progress=show_progress,
            **kwargs,
        )

    @classmethod
    def from_existing(
        cls: Type["PropertyGraphIndex"],
        property_graph_store: PropertyGraphStore,
        vector_store: Optional[BasePydanticVectorStore] = None,
        # general params
        llm: Optional[LLM] = None,
        kg_extractors: Optional[List[TransformComponent]] = None,
        # vector related params
        use_async: bool = True,
        embed_model: Optional[EmbedType] = None,
        embed_kg_nodes: bool = True,
        # parent class params
        callback_manager: Optional[CallbackManager] = None,
        transformations: Optional[List[TransformComponent]] = None,
        storage_context: Optional[StorageContext] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> "PropertyGraphIndex":
        """Create an index from an existing property graph store (and optional vector store)."""
        return cls(
            nodes=[],  # no nodes to insert
            property_graph_store=property_graph_store,
            vector_store=vector_store,
            llm=llm,
            kg_extractors=kg_extractors,
            use_async=use_async,
            embed_model=embed_model,
            embed_kg_nodes=embed_kg_nodes,
            callback_manager=callback_manager,
            transformations=transformations,
            storage_context=storage_context,
            show_progress=show_progress,
            **kwargs,
        )

    @property
    def property_graph_store(self) -> PropertyGraphStore:
        """Get the labelled property graph store."""
        assert self.storage_context.property_graph_store is not None

        return self.storage_context.property_graph_store

    @property
    def vector_store(self) -> Optional[BasePydanticVectorStore]:
        if self._embed_kg_nodes and self._override_vector_store:
            return self.storage_context.vector_store
        else:
            return None

    def _insert_nodes(self, nodes: Sequence[BaseNode]) -> Sequence[BaseNode]:
        """Insert nodes to the index struct."""
        if len(nodes) == 0:
            return nodes

        # run transformations on nodes to extract triplets
        if self._use_async:
            nodes = asyncio.run(
                arun_transformations(
                    nodes, self._kg_extractors, show_progress=self._show_progress
                )
            )
        else:
            nodes = run_transformations(
                nodes, self._kg_extractors, show_progress=self._show_progress
            )

        # ensure all nodes have nodes and/or relations in metadata
        assert all(
            node.metadata.get(KG_NODES_KEY) is not None
            or node.metadata.get(KG_RELATIONS_KEY) is not None
            for node in nodes
        )

        kg_nodes_to_insert: List[LabelledNode] = []
        kg_rels_to_insert: List[Relation] = []
        for node in nodes:
            # remove nodes and relations from metadata
            kg_nodes = node.metadata.pop(KG_NODES_KEY, [])
            kg_rels = node.metadata.pop(KG_RELATIONS_KEY, [])

            # add source id to properties
            for kg_node in kg_nodes:
                kg_node.properties[TRIPLET_SOURCE_KEY] = node.id_
            for kg_rel in kg_rels:
                kg_rel.properties[TRIPLET_SOURCE_KEY] = node.id_

            # add nodes and relations to insert lists
            kg_nodes_to_insert.extend(kg_nodes)
            kg_rels_to_insert.extend(kg_rels)

        # filter out duplicate kg nodes
        kg_node_ids = {node.id for node in kg_nodes_to_insert}
        existing_kg_nodes = self.property_graph_store.get(ids=list(kg_node_ids))
        existing_kg_node_ids = {node.id for node in existing_kg_nodes}
        kg_nodes_to_insert = [
            node for node in kg_nodes_to_insert if node.id not in existing_kg_node_ids
        ]

        # filter out duplicate llama nodes
        existing_nodes = self.property_graph_store.get_llama_nodes(
            [node.id_ for node in nodes]
        )
        existing_node_hashes = {node.hash for node in existing_nodes}
        nodes = [node for node in nodes if node.hash not in existing_node_hashes]

        # embed nodes (if needed)
        if self._embed_kg_nodes:
            # embed llama-index nodes
            node_texts = [
                node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes
            ]

            if self._use_async:
                embeddings = asyncio.run(
                    self._embed_model.aget_text_embedding_batch(
                        node_texts, show_progress=self._show_progress
                    )
                )
            else:
                embeddings = self._embed_model.get_text_embedding_batch(
                    node_texts, show_progress=self._show_progress
                )

            for node, embedding in zip(nodes, embeddings):
                node.embedding = embedding

            # embed kg nodes
            kg_node_texts = [str(kg_node) for kg_node in kg_nodes_to_insert]

            if self._use_async:
                kg_embeddings = asyncio.run(
                    self._embed_model.aget_text_embedding_batch(
                        kg_node_texts, show_progress=self._show_progress
                    )
                )
            else:
                kg_embeddings = self._embed_model.get_text_embedding_batch(
                    kg_node_texts,
                    show_progress=self._show_progress,
                )

            for kg_node, embedding in zip(kg_nodes_to_insert, kg_embeddings):
                kg_node.embedding = embedding

        # if graph store doesn't support vectors, or the vector index was provided, use it
        if self.vector_store is not None and len(kg_nodes_to_insert) > 0:
            self._insert_nodes_to_vector_index(kg_nodes_to_insert)

        if len(nodes) > 0:
            self.property_graph_store.upsert_llama_nodes(nodes)

        if len(kg_nodes_to_insert) > 0:
            self.property_graph_store.upsert_nodes(kg_nodes_to_insert)

        # important: upsert relations after nodes
        if len(kg_rels_to_insert) > 0:
            self.property_graph_store.upsert_relations(kg_rels_to_insert)

        # refresh schema if needed
        if self.property_graph_store.supports_structured_queries:
            self.property_graph_store.get_schema(refresh=True)

        return nodes

    def _insert_nodes_to_vector_index(self, nodes: List[LabelledNode]) -> None:
        """Insert vector nodes."""
        assert self.vector_store is not None

        llama_nodes: List[TextNode] = []
        for node in nodes:
            if node.embedding is not None:
                llama_nodes.append(
                    TextNode(
                        text=str(node),
                        metadata={VECTOR_SOURCE_KEY: node.id, **node.properties},
                        embedding=[*node.embedding],
                    )
                )
                if not self.vector_store.stores_text:
                    llama_nodes[-1].id_ = node.id

            # clear the embedding to save memory, its not used now
            node.embedding = None

        self.vector_store.add(llama_nodes)

    def _build_index_from_nodes(
        self, nodes: Optional[Sequence[BaseNode]], **build_kwargs: Any
    ) -> IndexLPG:
        """Build index from nodes."""
        nodes = self._insert_nodes(nodes or [])

        # this isn't really used or needed
        return IndexLPG()

    def as_retriever(
        self,
        sub_retrievers: Optional[List["BasePGRetriever"]] = None,
        include_text: bool = True,
        **kwargs: Any,
    ) -> BaseRetriever:
        """Return a retriever for the index.

        Args:
            sub_retrievers (Optional[List[BasePGRetriever]]):
                A list of sub-retrievers to use. If not provided, a default list will be used:
                `[LLMSynonymRetriever, VectorContextRetriever]` if the graph store supports vector queries.
            include_text (bool):
                Whether to include source-text in the retriever results.
            **kwargs:
                Additional kwargs to pass to the retriever.
        """
        from llama_index.core.indices.property_graph.retriever import (
            PGRetriever,
        )
        from llama_index.core.indices.property_graph.sub_retrievers.vector import (
            VectorContextRetriever,
        )
        from llama_index.core.indices.property_graph.sub_retrievers.llm_synonym import (
            LLMSynonymRetriever,
        )

        if sub_retrievers is None:
            sub_retrievers = [
                LLMSynonymRetriever(
                    graph_store=self.property_graph_store,
                    include_text=include_text,
                    llm=self._llm,
                    **kwargs,
                ),
            ]

            if self._embed_model and (
                self.property_graph_store.supports_vector_queries or self.vector_store
            ):
                sub_retrievers.append(
                    VectorContextRetriever(
                        index_id=self.index_id,
                        graph_store=self.property_graph_store,
                        vector_store=self.vector_store,
                        include_text=include_text,
                        embed_model=self._embed_model,
                        **kwargs,
                    )
                )

        return PGRetriever(sub_retrievers, use_async=self._use_async, **kwargs)

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        """Delete a node."""
        self.property_graph_store.delete(ids=[node_id])

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Index-specific logic for inserting nodes to the index struct."""
        self._insert_nodes(nodes)

    @property
    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        """Retrieve a dict mapping of ingested documents and their nodes+metadata."""
        raise NotImplementedError(
            "Ref doc info not implemented for PropertyGraphIndex. "
            "All inserts are already upserts."
        )
