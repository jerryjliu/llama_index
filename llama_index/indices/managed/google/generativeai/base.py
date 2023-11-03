"""Google GenerativeAI Semantic Vector Store & Attributed Question and Answering.

Google Generative AI Semantic Retriever API is a managed end to end service that
allows developers to create a corpus of documents to perform semantic search on
related passages given a user query.

Google Generative AI Attributed Question and Answering API is a managed
end-to-end service that allows developers to create responses grounded on
specified passages based on user queries.

For more information visit:
https://developers.generativeai.google/guide
"""

import datetime
import logging
from typing import Any, Optional, Sequence, Type, cast

from llama_index import VectorStoreIndex
from llama_index.data_structs.data_structs import IndexDict
from llama_index.indices.base import IndexType
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.managed.base import BaseManagedIndex
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.service_context import ServiceContext
from llama_index.response_synthesizers.google.generativeai import (
    GoogleTextSynthesizer,
)
from llama_index.schema import BaseNode, Document
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.google.generativeai import (
    GoogleVectorStore,
    google_service_context,
)

_logger = logging.getLogger(__name__)


class GoogleIndex(BaseManagedIndex):
    """Google's Generative AI Semantic vector store with AQA."""

    _store: GoogleVectorStore
    _index: VectorStoreIndex

    def __init__(
        self,
        vector_store: GoogleVectorStore,
        service_context: Optional[ServiceContext] = None,
        **kwargs: Any,
    ) -> None:
        """Creates an instance of GoogleIndex.

        Prefer to use the factories `from_corpus` or `create_corpus` instead.
        """
        actual_service_context = service_context or google_service_context

        self._store = vector_store
        self._index = VectorStoreIndex.from_vector_store(
            vector_store, service_context=actual_service_context, **kwargs
        )

        super().__init__(
            index_struct=self._index.index_struct,
            service_context=actual_service_context,
            **kwargs,
        )

    @classmethod
    def from_corpus(
        cls: Type[IndexType], *, corpus_id: str, **kwargs: Any
    ) -> IndexType:
        """Creates a GoogleIndex from an existing corpus.

        Args:
            corpus_id: ID of an existing corpus on Google's server.

        Returns:
            An instance of GoogleIndex pointing to the specified corpus.
        """
        _logger.debug(f"\n\nGoogleIndex.from_corpus(corpus_id={corpus_id})")
        return cls(
            vector_store=GoogleVectorStore.from_corpus(corpus_id=corpus_id), **kwargs
        )

    @classmethod
    def create_corpus(
        cls: Type[IndexType],
        *,
        corpus_id: Optional[str] = None,
        display_name: Optional[str] = None,
        **kwargs: Any,
    ) -> IndexType:
        """Creates a GoogleIndex from a new corpus.

        Args:
            corpus_id: ID of the new corpus to be created. If not provided,
                Google server will provide one.
            display_name: Title of the new corpus. If not provided, Google
                server will provide one.

        Returns:
            An instance of GoogleIndex pointing to the specified corpus.
        """
        _logger.debug(
            f"\n\nGoogleIndex.from_new_corpus(new_corpus_id={corpus_id}, new_display_name={display_name})"
        )
        return cls(
            vector_store=GoogleVectorStore.create_corpus(
                corpus_id=corpus_id, display_name=display_name
            ),
            **kwargs,
        )

    @classmethod
    def from_documents(
        cls: Type[IndexType],
        documents: Sequence[Document],
        storage_context: Optional[StorageContext] = None,
        service_context: Optional[ServiceContext] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> IndexType:
        """Build an index from a sequence of documents."""
        _logger.debug(f"\n\nGoogleIndex.from_documents(...)")

        new_display_name = f"Corpus created on {datetime.datetime.now()}"
        c = cls(
            vector_store=GoogleVectorStore.create_corpus(display_name=new_display_name),
            **kwargs,
        )

        index = cast(GoogleIndex, c)
        index.insert_documents(documents=documents)

        return c

    @property
    def corpus_id(self) -> str:
        """Returns the corpus ID being used by this GoogleIndex."""
        return self._store.corpus_id

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Inserts a set of nodes."""
        self._index.insert_nodes(nodes=nodes)

    def insert_documents(self, documents: Sequence[Document], **kwargs: Any) -> None:
        """Inserts a set of documents."""
        for document in documents:
            self.insert(document=document, **kwargs)

    def delete_ref_doc(
        self, ref_doc_id: str, delete_from_docstore: bool = False, **delete_kwargs: Any
    ) -> None:
        """Deletes a document and its nodes by using ref_doc_id."""
        self._index.delete_ref_doc(ref_doc_id=ref_doc_id)

    def update_ref_doc(self, document: Document, **update_kwargs: Any) -> None:
        """Updates a document and its corresponding nodes."""
        self._index.update(document=document)

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Returns a Retriever for this managed index."""
        return self._index.as_retriever(**kwargs)

    def as_query_engine(
        self, *, answer_style: int = 1, **kwargs: Any
    ) -> BaseQueryEngine:
        """Returns the AQA engine for this index.

        Args:
            answer_style: See `google.ai.generativelanguage.AnswerStyle`

        Returns:
            A query engine that uses Google's AQA model.
        """
        return super().as_query_engine(
            retriever=self.as_retriever(**kwargs),
            response_synthesizer=GoogleTextSynthesizer(answer_style=answer_style),
            **kwargs,
        )

    def _build_index_from_nodes(self, nodes: Sequence[BaseNode]) -> IndexDict:
        """Build the index from nodes."""
        return self._index._build_index_from_nodes(nodes)
