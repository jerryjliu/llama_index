"""
Base vector store index.

An index that is built on top of an existing vector store.

"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence


from llama_index.core.async_utils import run_async_tasks
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.embeddings.utils import EmbedType, resolve_embed_model
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices.utils import async_embed_nodes, embed_nodes
from llama_index.core.schema import (
    BaseNode,
    ImageNode,
    IndexNode,
    MetadataMode,
    TransformComponent,
)
from llama_index.core.settings import Settings
from llama_index.core.storage.docstore.types import RefDocInfo
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.utils import iter_batch
from llama_index.core.vector_stores.types import BasePydanticVectorStore

logger = logging.getLogger(__name__)


class VectorStoreIndex(BaseIndex[IndexDict]):
    """
    Vector Store Index.

    Args:
        use_async (bool): Whether to use asynchronous calls. Defaults to False.
        show_progress (bool): Whether to show tqdm progress bars. Defaults to False.
        store_nodes_override (bool): set to True to always store Node objects in index
            store and document store even if vector store keeps text. Defaults to False
    """

    index_struct_cls = IndexDict

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        # vector store index params
        use_async: bool = False,
        store_nodes_override: bool = False,
        embed_model: Optional[EmbedType] = None,
        insert_batch_size: int = 2048,
        # parent class params
        objects: Optional[Sequence[IndexNode]] = None,
        index_struct: Optional[IndexDict] = None,
        storage_context: Optional[StorageContext] = None,
        callback_manager: Optional[CallbackManager] = None,
        transformations: Optional[List[TransformComponent]] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._use_async = use_async
        self._store_nodes_override = store_nodes_override
        self._embed_model = (
            resolve_embed_model(embed_model, callback_manager=callback_manager)
            if embed_model
            else Settings.embed_model
        )

        self._insert_batch_size = insert_batch_size
        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            storage_context=storage_context,
            show_progress=show_progress,
            objects=objects,
            callback_manager=callback_manager,
            transformations=transformations,
            **kwargs,
        )

    @classmethod
    def from_vector_store(
        cls,
        vector_store: BasePydanticVectorStore,
        embed_model: Optional[EmbedType] = None,
        **kwargs: Any,
    ) -> "VectorStoreIndex":
        if not vector_store.stores_text:
            raise ValueError(
                "Cannot initialize from a vector store that does not store text."
            )

        kwargs.pop("storage_context", None)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        return cls(
            nodes=[],
            embed_model=embed_model,
            storage_context=storage_context,
            **kwargs,
        )

    @property
    def vector_store(self) -> BasePydanticVectorStore:
        return self._vector_store

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        # NOTE: lazy import
        from llama_index.core.indices.vector_store.retrievers import (
            VectorIndexRetriever,
        )

        return VectorIndexRetriever(
            self,
            node_ids=list(self.index_struct.nodes_dict.values()),
            callback_manager=self._callback_manager,
            object_map=self._object_map,
            **kwargs,
        )

    def _get_node_with_embedding(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """
        Get tuples of id, node, and embedding.

        Allows us to store these nodes in a vector store.
        Embeddings are called in batches.

        """
        id_to_embed_map = embed_nodes(
            nodes, self._embed_model, show_progress=show_progress
        )

        results = []
        for node in nodes:
            embedding = id_to_embed_map[node.node_id]
            result = node.model_copy()
            result.embedding = embedding
            results.append(result)
        return results

    async def _aget_node_with_embedding(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """
        Asynchronously get tuples of id, node, and embedding.

        Allows us to store these nodes in a vector store.
        Embeddings are called in batches.

        """
        id_to_embed_map = await async_embed_nodes(
            nodes=nodes,
            embed_model=self._embed_model,
            show_progress=show_progress,
        )

        results = []
        for node in nodes:
            embedding = id_to_embed_map[node.node_id]
            result = node.model_copy()
            result.embedding = embedding
            results.append(result)
        return results

    async def _async_add_nodes_to_index(
        self,
        index_struct: IndexDict,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **insert_kwargs: Any,
    ) -> None:
        """Asynchronously add nodes to index."""
        if not nodes:
            return

        for nodes_batch in iter_batch(nodes, self._insert_batch_size):
            nodes_batch = await self._aget_node_with_embedding(
                nodes_batch, show_progress
            )
            new_ids = await self._vector_store.async_add(nodes_batch, index_id=index_struct.index_id, **insert_kwargs)

            # if the vector store doesn't store text, we need to add the nodes to the
            # index struct and document store
            if not self._vector_store.stores_text or self._store_nodes_override:
                for node, new_id in zip(nodes_batch, new_ids):
                    # NOTE: remove embedding from node to avoid duplication
                    node_without_embedding = node.model_copy()
                    node_without_embedding.embedding = None

                    index_struct.add_node(node_without_embedding, text_id=new_id)
                    self._docstore.add_documents(
                        [node_without_embedding], allow_update=True
                    )
            else:
                # NOTE: if the vector store keeps text,
                # we only need to add image and index nodes
                for node, new_id in zip(nodes_batch, new_ids):
                    if isinstance(node, (ImageNode, IndexNode)):
                        # NOTE: remove embedding from node to avoid duplication
                        node_without_embedding = node.model_copy()
                        node_without_embedding.embedding = None

                        index_struct.add_node(node_without_embedding, text_id=new_id)
                        self._docstore.add_documents(
                            [node_without_embedding], allow_update=True
                        )

    def _add_nodes_to_index(
        self,
        index_struct: IndexDict,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **insert_kwargs: Any,
    ) -> None:
        """Add document to index."""
        if not nodes:
            return

        for nodes_batch in iter_batch(nodes, self._insert_batch_size):
            nodes_batch = self._get_node_with_embedding(nodes_batch, show_progress)
            new_ids = self._vector_store.add(nodes_batch, index_id=index_struct.index_id, **insert_kwargs)

            if not self._vector_store.stores_text or self._store_nodes_override:
                # NOTE: if the vector store doesn't store text,
                # we need to add the nodes to the index struct and document store
                for node, new_id in zip(nodes_batch, new_ids):
                    # NOTE: remove embedding from node to avoid duplication
                    node_without_embedding = node.model_copy()
                    node_without_embedding.embedding = None

                    index_struct.add_node(node_without_embedding, text_id=new_id)
                    self._docstore.add_documents(
                        [node_without_embedding], allow_update=True
                    )
            else:
                # NOTE: if the vector store keeps text,
                # we only need to add image and index nodes
                for node, new_id in zip(nodes_batch, new_ids):
                    if isinstance(node, (ImageNode, IndexNode)):
                        # NOTE: remove embedding from node to avoid duplication
                        node_without_embedding = node.model_copy()
                        node_without_embedding.embedding = None

                        index_struct.add_node(node_without_embedding, text_id=new_id)
                        self._docstore.add_documents(
                            [node_without_embedding], allow_update=True
                        )

    def _build_index_from_nodes(
        self,
        nodes: Sequence[BaseNode],
        **insert_kwargs: Any,
    ) -> IndexDict:
        """Build index from nodes."""
        index_struct = self.index_struct_cls()
        if self._use_async:
            tasks = [
                self._async_add_nodes_to_index(
                    index_struct,
                    nodes,
                    show_progress=self._show_progress,
                    **insert_kwargs,
                )
            ]
            run_async_tasks(tasks)
        else:
            self._add_nodes_to_index(
                index_struct,
                nodes,
                show_progress=self._show_progress,
                **insert_kwargs,
            )
        return index_struct

    def build_index_from_nodes(
        self,
        nodes: Sequence[BaseNode],
        **insert_kwargs: Any,
    ) -> IndexDict:
        """
        Build the index from nodes.

        NOTE: Overrides BaseIndex.build_index_from_nodes.
            VectorStoreIndex only stores nodes in document store
            if vector store does not store text
        """
        # Filter out the nodes that don't have content
        content_nodes = [
            node
            for node in nodes
            if node.get_content(metadata_mode=MetadataMode.EMBED) != ""
        ]

        # Report if some nodes are missing content
        if len(content_nodes) != len(nodes):
            print("Some nodes are missing content, skipping them...")

        return self._build_index_from_nodes(content_nodes, **insert_kwargs)

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Insert a document."""
        self._add_nodes_to_index(self._index_struct, nodes, **insert_kwargs)

    def insert_nodes(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """
        Insert nodes.

        NOTE: overrides BaseIndex.insert_nodes.
            VectorStoreIndex only stores nodes in document store
            if vector store does not store text
        """
        for node in nodes:
            if isinstance(node, IndexNode):
                try:
                    node.dict()
                except ValueError:
                    self._object_map[node.index_id] = node.obj
                    node.obj = None

        with self._callback_manager.as_trace("insert_nodes"):
            self._insert(nodes, **insert_kwargs)
            self._storage_context.index_store.add_index_struct(self._index_struct)

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        pass

    def delete_nodes(
        self,
        node_ids: List[str],
        delete_from_docstore: bool = False,
        **delete_kwargs: Any,
    ) -> None:
        """
        Delete a list of nodes from the index.

        Args:
            node_ids (List[str]): A list of node_ids from the nodes to delete

        """
        # delete nodes from vector store
        self._vector_store.delete_nodes(node_ids, **delete_kwargs)

        # delete from docstore only if needed
        if (
            not self._vector_store.stores_text or self._store_nodes_override
        ) and delete_from_docstore:
            for node_id in node_ids:
                self._docstore.delete_document(node_id, raise_error=False)

    def _delete_from_index_struct(self, ref_doc_id: str) -> None:
        # delete from index_struct only if needed
        if not self._vector_store.stores_text or self._store_nodes_override:
            ref_doc_info = self._docstore.get_ref_doc_info(ref_doc_id)
            if ref_doc_info is not None:
                for node_id in ref_doc_info.node_ids:
                    self._index_struct.delete(node_id)
                    self._vector_store.delete(node_id)

    def _delete_from_docstore(self, ref_doc_id: str) -> None:
        # delete from docstore only if needed
        if not self._vector_store.stores_text or self._store_nodes_override:
            self._docstore.delete_ref_doc(ref_doc_id, raise_error=False)

    def delete_ref_doc(
        self, ref_doc_id: str, delete_from_docstore: bool = False, **delete_kwargs: Any
    ) -> None:
        """Delete a document and it's nodes by using ref_doc_id."""
        self._vector_store.delete(ref_doc_id, **delete_kwargs)
        self._delete_from_index_struct(ref_doc_id)
        if delete_from_docstore:
            self._delete_from_docstore(ref_doc_id)
        self._storage_context.index_store.add_index_struct(self._index_struct)

    async def _adelete_from_index_struct(self, ref_doc_id: str) -> None:
        """Delete from index_struct only if needed."""
        if not self._vector_store.stores_text or self._store_nodes_override:
            ref_doc_info = await self._docstore.aget_ref_doc_info(ref_doc_id)
            if ref_doc_info is not None:
                for node_id in ref_doc_info.node_ids:
                    self._index_struct.delete(node_id)
                    self._vector_store.delete(node_id)

    async def _adelete_from_docstore(self, ref_doc_id: str) -> None:
        """Delete from docstore only if needed."""
        if not self._vector_store.stores_text or self._store_nodes_override:
            await self._docstore.adelete_ref_doc(ref_doc_id, raise_error=False)

    async def adelete_ref_doc(
        self, ref_doc_id: str, delete_from_docstore: bool = False, **delete_kwargs: Any
    ) -> None:
        """Delete a document and it's nodes by using ref_doc_id."""
        tasks = [
            self._vector_store.adelete(ref_doc_id, **delete_kwargs),
            self._adelete_from_index_struct(ref_doc_id),
        ]
        if delete_from_docstore:
            tasks.append(self._adelete_from_docstore(ref_doc_id))

        await asyncio.gather(*tasks)

        self._storage_context.index_store.add_index_struct(self._index_struct)

    @property
    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        """Retrieve a dict mapping of ingested documents and their nodes+metadata."""
        if not self._vector_store.stores_text or self._store_nodes_override:
            node_doc_ids = list(self.index_struct.nodes_dict.values())
            nodes = self.docstore.get_nodes(node_doc_ids)

            all_ref_doc_info = {}
            for node in nodes:
                ref_node = node.source_node
                if not ref_node:
                    continue

                ref_doc_info = self.docstore.get_ref_doc_info(ref_node.node_id)
                if not ref_doc_info:
                    continue

                all_ref_doc_info[ref_node.node_id] = ref_doc_info
            return all_ref_doc_info
        else:
            raise NotImplementedError(
                "Vector store integrations that store text in the vector store are "
                "not supported by ref_doc_info yet."
            )


GPTVectorStoreIndex = VectorStoreIndex
