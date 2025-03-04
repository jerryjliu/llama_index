"""Base index classes."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Sequence, Type, TypeVar

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.chat_engine.types import BaseChatEngine, ChatMode
from llama_index.core.data_structs.data_structs import IndexStruct
from llama_index.core.ingestion import run_transformations
from llama_index.core.llms.utils import LLMType, resolve_llm
from llama_index.core.schema import BaseNode, Document, IndexNode, TransformComponent
from llama_index.core.settings import Settings
from llama_index.core.storage.docstore.types import BaseDocumentStore, RefDocInfo
from llama_index.core.storage.storage_context import StorageContext

IS = TypeVar("IS", bound=IndexStruct)
IndexType = TypeVar("IndexType", bound="BaseIndex")

logger = logging.getLogger(__name__)


class BaseIndex(Generic[IS], ABC):
    """Base LlamaIndex.

    Args:
        nodes (List[Node]): List of nodes to index
        show_progress (bool): Whether to show tqdm progress bars. Defaults to False.
    """

    index_struct_cls: Type[IS]

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        objects: Optional[Sequence[IndexNode]] = None,
        index_struct: Optional[IS] = None,
        storage_context: Optional[StorageContext] = None,
        callback_manager: Optional[CallbackManager] = None,
        transformations: Optional[List[TransformComponent]] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize with parameters."""
        if index_struct is None and nodes is None and objects is None:
            raise ValueError("One of nodes, objects, or index_struct must be provided.")
        if index_struct is not None and nodes is not None and len(nodes) >= 1:
            raise ValueError("Only one of nodes or index_struct can be provided.")
        # This is to explicitly make sure that the old UX is not used
        if nodes is not None and len(nodes) >= 1 and not isinstance(nodes[0], BaseNode):
            if isinstance(nodes[0], Document):
                raise ValueError(
                    "The constructor now takes in a list of Node objects. "
                    "Since you are passing in a list of Document objects, "
                    "please use `from_documents` instead."
                )
            else:
                raise ValueError("nodes must be a list of Node objects.")

        self._storage_context = storage_context or StorageContext.from_defaults()
        self._docstore = self._storage_context.docstore
        self._show_progress = show_progress
        self._vector_store = self._storage_context.vector_store
        self._graph_store = self._storage_context.graph_store
        self._callback_manager = callback_manager or Settings.callback_manager

        objects = objects or []
        self._object_map = {obj.index_id: obj.obj for obj in objects}
        for obj in objects:
            obj.obj = None  # clear the object to avoid serialization issues

        with self._callback_manager.as_trace("index_construction"):
            if index_struct is None:
                nodes = nodes or []
                index_struct = self.build_index_from_nodes(
                    nodes + objects,  # type: ignore
                    **kwargs,  # type: ignore
                )
            self._index_struct = index_struct
            self._storage_context.index_store.add_index_struct(self._index_struct)

        self._transformations = transformations or Settings.transformations

    @classmethod
    def from_documents(
        cls: Type[IndexType],
        documents: Sequence[Document],
        storage_context: Optional[StorageContext] = None,
        show_progress: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        transformations: Optional[List[TransformComponent]] = None,
        **kwargs: Any,
    ) -> IndexType:
        """Create index from documents.

        Args:
            documents (Optional[Sequence[BaseDocument]]): List of documents to
                build the index from.

        """
        storage_context = storage_context or StorageContext.from_defaults()
        docstore = storage_context.docstore
        callback_manager = callback_manager or Settings.callback_manager
        transformations = transformations or Settings.transformations

        with callback_manager.as_trace("index_construction"):
            for doc in documents:
                docstore.set_document_hash(doc.get_doc_id(), doc.hash)

            nodes = run_transformations(
                documents,  # type: ignore
                transformations,
                show_progress=show_progress,
                **kwargs,
            )

            return cls(
                nodes=nodes,
                storage_context=storage_context,
                callback_manager=callback_manager,
                show_progress=show_progress,
                transformations=transformations,
                **kwargs,
            )

    @property
    def index_struct(self) -> IS:
        """Get the index struct."""
        return self._index_struct

    @property
    def index_id(self) -> str:
        """Get the index struct."""
        return self._index_struct.index_id

    def set_index_id(self, index_id: str) -> None:
        """Set the index id.

        NOTE: if you decide to set the index_id on the index_struct manually,
        you will need to explicitly call `add_index_struct` on the `index_store`
        to update the index store.

        Args:
            index_id (str): Index id to set.

        """
        # delete the old index struct
        old_id = self._index_struct.index_id
        self._storage_context.index_store.delete_index_struct(old_id)
        # add the new index struct
        self._index_struct.index_id = index_id
        self._storage_context.index_store.add_index_struct(self._index_struct)

    @property
    def docstore(self) -> BaseDocumentStore:
        """Get the docstore corresponding to the index."""
        return self._docstore

    @property
    def storage_context(self) -> StorageContext:
        return self._storage_context

    @property
    def summary(self) -> str:
        return str(self._index_struct.summary)

    @summary.setter
    def summary(self, new_summary: str) -> None:
        self._index_struct.summary = new_summary
        self._storage_context.index_store.add_index_struct(self._index_struct)

    @abstractmethod
    def _build_index_from_nodes(
        self, nodes: Sequence[BaseNode], **build_kwargs: Any
    ) -> IS:
        """Build the index from nodes."""

    def build_index_from_nodes(
        self, nodes: Sequence[BaseNode], **build_kwargs: Any
    ) -> IS:
        """Build the index from nodes."""
        self._docstore.add_documents(nodes, allow_update=True)
        return self._build_index_from_nodes(nodes, **build_kwargs)

    @abstractmethod
    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Index-specific logic for inserting nodes to the index struct."""

    def insert_nodes(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Insert nodes."""
        for node in nodes:
            if isinstance(node, IndexNode):
                try:
                    node.dict()
                except ValueError:
                    self._object_map[node.index_id] = node.obj
                    node.obj = None

        with self._callback_manager.as_trace("insert_nodes"):
            self.docstore.add_documents(nodes, allow_update=True)
            self._insert(nodes, **insert_kwargs)
            self._storage_context.index_store.add_index_struct(self._index_struct)

    def insert(
        self, documents: Document | Sequence[Document], **insert_kwargs: Any
    ) -> None:
        """Insert a document."""
        with self._callback_manager.as_trace("insert"):
            if isinstance(documents, Document):
                documents = [documents]

            nodes = run_transformations(
                documents,
                self._transformations,
                show_progress=self._show_progress,
                **insert_kwargs,
            )

            self.insert_nodes(nodes, **insert_kwargs)
            for doc in documents:
                self.docstore.set_document_hash(doc.get_doc_id(), doc.hash)

    @abstractmethod
    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        """Delete a node."""

    def delete_nodes(
        self,
        node_ids: List[str],
        delete_from_docstore: bool = False,
        **delete_kwargs: Any,
    ) -> None:
        """Delete a list of nodes from the index.

        Args:
            doc_ids (List[str]): A list of doc_ids from the nodes to delete

        """
        for node_id in node_ids:
            self._delete_node(node_id, **delete_kwargs)
            if delete_from_docstore:
                self.docstore.delete_document(node_id, raise_error=False)

        self._storage_context.index_store.add_index_struct(self._index_struct)

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document from the index.
        All nodes in the index related to the index will be deleted.

        Args:
            doc_id (str): A doc_id of the ingested document

        """
        logger.warning(
            "delete() is now deprecated, please refer to delete_ref_doc() to delete "
            "ingested documents+nodes or delete_nodes to delete a list of nodes."
        )
        self.delete_ref_doc(doc_id)

    def delete_ref_doc(
        self, ref_doc_id: str, delete_from_docstore: bool = False, **delete_kwargs: Any
    ) -> None:
        """Delete a document and it's nodes by using ref_doc_id."""
        ref_doc_info = self.docstore.get_ref_doc_info(ref_doc_id)
        if ref_doc_info is None:
            logger.warning(f"ref_doc_id {ref_doc_id} not found, nothing deleted.")
            return

        self.delete_nodes(
            ref_doc_info.node_ids,
            delete_from_docstore=False,
            **delete_kwargs,
        )

        if delete_from_docstore:
            self.docstore.delete_ref_doc(ref_doc_id, raise_error=False)

    def update(self, document: Document, **update_kwargs: Any) -> None:
        """Update a document and it's corresponding nodes.

        This is equivalent to deleting the document and then inserting it again.

        Args:
            document (Union[BaseDocument, BaseIndex]): document to update
            insert_kwargs (Dict): kwargs to pass to insert
            delete_kwargs (Dict): kwargs to pass to delete

        """
        logger.warning(
            "update() is now deprecated, please refer to update_ref_doc() to update "
            "ingested documents+nodes."
        )
        self.update_ref_doc(document, **update_kwargs)

    def update_ref_doc(self, document: Document, **update_kwargs: Any) -> None:
        """Update a document and it's corresponding nodes.

        This is equivalent to deleting the document and then inserting it again.

        Args:
            document (Union[BaseDocument, BaseIndex]): document to update
            insert_kwargs (Dict): kwargs to pass to insert
            delete_kwargs (Dict): kwargs to pass to delete

        """
        self.refresh_ref_docs([document], **update_kwargs)

    def refresh(
        self, documents: Sequence[Document], **update_kwargs: Any
    ) -> List[bool]:
        """Refresh an index with documents that have changed.

        This allows users to save LLM and Embedding model calls, while only
        updating documents that have any changes in text or metadata. It
        will also insert any documents that previously were not stored.
        """
        logger.warning(
            "refresh() is now deprecated, please refer to refresh_ref_docs() to "
            "refresh ingested documents+nodes with an updated list of documents."
        )
        return self.refresh_ref_docs(documents, **update_kwargs)

    def refresh_ref_docs(
        self, documents: Sequence[Document], **update_kwargs: Any
    ) -> List[bool]:
        """Refresh an index with documents that have changed.

        This allows users to save LLM and Embedding model calls, while only
        updating documents that have any changes in text or metadata. It
        will also insert any documents that previously were not stored.
        """
        with self._callback_manager.as_trace("refresh"):
            refreshed_documents = [False] * len(documents)
            for i, doc in enumerate(documents):
                existing_doc_hash = self._docstore.get_document_hash(doc.get_doc_id())
                if existing_doc_hash is None:
                    refreshed_documents[i] = True
                elif existing_doc_hash != doc.hash:
                    self.delete_ref_doc(
                        doc.get_doc_id(),
                        delete_from_docstore=True,
                        **update_kwargs.pop("delete_kwargs", {}),
                    )
                    refreshed_documents[i] = True

            documents = [
                doc for i, doc in enumerate(documents) if refreshed_documents[i]
            ]
            self.insert(documents, **update_kwargs.pop("insert_kwargs", {}))

            return refreshed_documents

    @property
    @abstractmethod
    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        """Retrieve a dict mapping of ingested documents and their nodes+metadata."""
        ...

    @abstractmethod
    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        ...

    def as_query_engine(
        self, llm: Optional[LLMType] = None, **kwargs: Any
    ) -> BaseQueryEngine:
        """Convert the index to a query engine.

        Calls `index.as_retriever(**kwargs)` to get the retriever and then wraps it in a
        `RetrieverQueryEngine.from_args(retriever, **kwrags)` call.
        """
        # NOTE: lazy import
        from llama_index.core.query_engine.retriever_query_engine import (
            RetrieverQueryEngine,
        )

        retriever = self.as_retriever(**kwargs)
        llm = (
            resolve_llm(llm, callback_manager=self._callback_manager)
            if llm
            else Settings.llm
        )

        return RetrieverQueryEngine.from_args(
            retriever,
            llm=llm,
            **kwargs,
        )

    def as_chat_engine(
        self,
        chat_mode: ChatMode = ChatMode.BEST,
        llm: Optional[LLMType] = None,
        **kwargs: Any,
    ) -> BaseChatEngine:
        """Convert the index to a chat engine.

        Calls `index.as_query_engine(llm=llm, **kwargs)` to get the query engine and then
        wraps it in a chat engine based on the chat mode.

        Chat modes:
            - `ChatMode.BEST` (default): Chat engine that uses an agent (react or openai) with a query engine tool
            - `ChatMode.CONTEXT`: Chat engine that uses a retriever to get context
            - `ChatMode.CONDENSE_QUESTION`: Chat engine that condenses questions
            - `ChatMode.CONDENSE_PLUS_CONTEXT`: Chat engine that condenses questions and uses a retriever to get context
            - `ChatMode.SIMPLE`: Simple chat engine that uses the LLM directly
            - `ChatMode.REACT`: Chat engine that uses a react agent with a query engine tool
            - `ChatMode.OPENAI`: Chat engine that uses an openai agent with a query engine tool
        """
        llm = (
            resolve_llm(llm, callback_manager=self._callback_manager)
            if llm
            else Settings.llm
        )

        query_engine = self.as_query_engine(llm=llm, **kwargs)

        # resolve chat mode
        if chat_mode in [ChatMode.REACT, ChatMode.OPENAI, ChatMode.BEST]:
            # use an agent with query engine tool in these chat modes
            # NOTE: lazy import
            from llama_index.core.agent import AgentRunner
            from llama_index.core.tools.query_engine import QueryEngineTool

            # convert query engine to tool
            query_engine_tool = QueryEngineTool.from_defaults(query_engine=query_engine)

            return AgentRunner.from_llm(
                tools=[query_engine_tool],
                llm=llm,
                **kwargs,
            )

        if chat_mode == ChatMode.CONDENSE_QUESTION:
            # NOTE: lazy import
            from llama_index.core.chat_engine import CondenseQuestionChatEngine

            return CondenseQuestionChatEngine.from_defaults(
                query_engine=query_engine,
                llm=llm,
                **kwargs,
            )
        elif chat_mode == ChatMode.CONTEXT:
            from llama_index.core.chat_engine import ContextChatEngine

            return ContextChatEngine.from_defaults(
                retriever=self.as_retriever(**kwargs),
                llm=llm,
                **kwargs,
            )

        elif chat_mode == ChatMode.CONDENSE_PLUS_CONTEXT:
            from llama_index.core.chat_engine import CondensePlusContextChatEngine

            return CondensePlusContextChatEngine.from_defaults(
                retriever=self.as_retriever(**kwargs),
                llm=llm,
                **kwargs,
            )

        elif chat_mode == ChatMode.SIMPLE:
            from llama_index.core.chat_engine import SimpleChatEngine

            return SimpleChatEngine.from_defaults(
                llm=llm,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown chat mode: {chat_mode}")


# legacy
BaseGPTIndex = BaseIndex
