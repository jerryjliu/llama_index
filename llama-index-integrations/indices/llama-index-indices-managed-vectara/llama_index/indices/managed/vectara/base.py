"""
Managed index.

A managed Index - where the index is accessible via some API that
interfaces a managed service.
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Sequence, Type, Dict
from functools import lru_cache

import requests
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.data_structs.data_structs import IndexDict, IndexStructType
from llama_index.core.indices.managed.base import BaseManagedIndex, IndexType
from llama_index.core.llms.utils import LLMType, resolve_llm
from llama_index.core.schema import (
    Document,
    Node,
    TransformComponent,
)
from llama_index.core.settings import Settings
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core import get_response_synthesizer


_logger = logging.getLogger(__name__)


class VectaraIndexStruct(IndexDict):
    """Vectara Index Struct."""

    @classmethod
    def get_type(cls) -> IndexStructType:
        """Get index struct type."""
        return IndexStructType.VECTARA


class VectaraIndex(BaseManagedIndex):
    """
    Vectara Index.

    The Vectara index implements a managed index that uses Vectara as the backend.
    Vectara performs a lot of the functions in traditional indexes in the backend:
    - breaks down a document into chunks (nodes)
    - Creates the embedding for each chunk (node)
    - Performs the search for the top k most similar nodes to a query
    - Optionally can perform summarization of the top k nodes

    Args:
        show_progress (bool): Whether to show tqdm progress bars. Defaults to False.

    """

    def __init__(
        self,
        show_progress: bool = False,
        vectara_corpus_key: Optional[str] = None,
        vectara_api_key: Optional[str] = None,
        parallelize_ingest: bool = False,
        x_source_str: str = "llama_index",
        vectara_base_url: str = "https://api.vectara.io",
        vectara_verify_ssl: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the Vectara API."""
        self.parallelize_ingest = parallelize_ingest
        self._base_url = vectara_base_url.rstrip("/")

        index_struct = VectaraIndexStruct(
            index_id=str(vectara_corpus_key),
            summary="Vectara Index",
        )

        super().__init__(
            show_progress=show_progress,
            index_struct=index_struct,
            **kwargs,
        )

        self._vectara_corpus_key = vectara_corpus_key or str(
            os.environ.get("VECTARA_CORPUS_KEY")
        )

        self._vectara_api_key = vectara_api_key or os.environ.get("VECTARA_API_KEY")
        if self._vectara_corpus_key is None or self._vectara_api_key is None:
            _logger.warning(
                "Can't find Vectara credentials or corpus_key in environment."
            )
            raise ValueError("Missing Vectara credentials")
        else:
            _logger.debug(f"Using corpus key {self._vectara_corpus_key}")

        # identifies usage source for internal measurement
        self._x_source_str = x_source_str

        # setup requests session with max 3 retries and 90s timeout
        # for calling Vectara API
        self._session = requests.Session()  # to reuse connections
        if not vectara_verify_ssl:
            self._session.verify = False  # to ignore SSL verification
        adapter = requests.adapters.HTTPAdapter(max_retries=3)
        self._session.mount("https://", adapter)
        self.vectara_api_timeout = 90
        self.doc_ids: List[str] = []

    @lru_cache(maxsize=None)
    def _get_corpus_key(self, corpus_key: str) -> str:
        """
        Get the corpus key to use for the index.
        If corpus_key is provided, check if it is one of the valid corpus keys.
        If not, use the first corpus key in the list.
        """
        if corpus_key is not None:
            if corpus_key in self._vectara_corpus_key.split(","):
                return corpus_key
        return self._vectara_corpus_key.split(",")[0]

    def _get_post_headers(self) -> dict:
        """Returns headers that should be attached to each post request."""
        return {
            "x-api-key": self._vectara_api_key,
            "Content-Type": "application/json",
            "X-Source": self._x_source_str,
        }

    def _delete_doc(self, doc_id: str, corpus_key: Optional[str] = None) -> bool:
        """
        Delete a document from the Vectara corpus.

        Args:
            doc_id (str): ID of the document to delete.
            corpus_key (str): corpus key to delete the document from.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        valid_corpus_key = self._get_corpus_key(corpus_key)
        body = {}
        response = self._session.delete(
            f"{self._base_url}/v2/corpora/{valid_corpus_key}/documents/{doc_id}",
            data=json.dumps(body),
            verify=True,
            headers=self._get_post_headers(),
            timeout=self.vectara_api_timeout,
        )

        if response.status_code != 204:
            _logger.error(
                f"Delete request failed for doc_id = {doc_id} with status code "
                f"{response.status_code}, text {response.json()['messages'][0]}"
            )
            return False
        return True

    def _index_doc(self, doc: dict, corpus_key) -> str:
        response = self._session.post(
            headers=self._get_post_headers(),
            url=f"{self._base_url}/v2/corpora/{corpus_key}/documents",
            data=json.dumps(doc),
            timeout=self.vectara_api_timeout,
            verify=True,
        )

        status_code = response.status_code
        if status_code == 201:
            return "E_SUCCEEDED"

        result = response.json()
        return result["messages"][0]

    def _insert(
        self,
        document: Optional[Document] = None,
        nodes: Optional[Sequence[Node]] = None,
        corpus_key: Optional[str] = None,
        **insert_kwargs: Any,
    ) -> None:
        """
        Insert a document into a corpus using Vectara's indexing API.

        Args:
            document (Document): a document to index using Vectara's Structured Document type.
            nodes (Sequence[Node]): a list of nodes representing document parts to index a document using Vectara's Core Document type.
            corpus_key (str): If multiple corpora are provided for this index, the corpus_key of the corpus you want to add the document to.
        """
        if document:
            # Use Structured Document type
            metadata = document.metadata.copy()
            metadata["framework"] = "llama_index"
            doc = {
                "id": document.id_,
                "type": "structured",
                "metadata": metadata,
                "sections": [{"text": document.text_resource.text}],
            }

            if "title" in insert_kwargs and insert_kwargs["title"]:
                doc["title"] = insert_kwargs["title"]

            if "description" in insert_kwargs and insert_kwargs["description"]:
                doc["description"] = insert_kwargs["description"]

            if (
                "max_chars_per_chunk" in insert_kwargs
                and insert_kwargs["max_chars_per_chunk"]
            ):
                doc["chunking_strategy"] = {
                    "type": "max_chars_chunking_strategy",
                    "max_chars_per_chunk": insert_kwargs["max_chars_per_chunk"],
                }

        elif nodes:
            # Use Core Document type
            metadata = insert_kwargs["doc_metadata"]
            metadata["framework"] = "llama_index"
            doc = {
                "id": insert_kwargs["doc_id"],
                "type": "core",
                "metadata": metadata,
                "document_parts": [
                    {"text": node.text_resource.text, "metadata": node.metadata}
                    for node in nodes
                ],
            }

        else:
            _logger.error(
                "Error indexing document. Must provide either a document or a list of nodes."
            )
            return

        valid_corpus_key = self._get_corpus_key(corpus_key)
        if self.parallelize_ingest:
            with ThreadPoolExecutor() as executor:
                future = executor.submit(self._index_doc, doc, valid_corpus_key)
                ecode = future.result()
                if ecode != "E_SUCCEEDED":
                    _logger.error(
                        f"Error indexing document in Vectara with error code {ecode}"
                    )
            self.doc_ids.append(doc["id"])
        else:
            ecode = self._index_doc(doc, valid_corpus_key)
            if ecode != "E_SUCCEEDED":
                _logger.error(
                    f"Error indexing document in Vectara with error code {ecode}"
                )
            self.doc_ids.append(doc["id"])

    def add_document(
        self,
        doc: Document,
        corpus_key: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        max_chars_per_chunk: Optional[int] = None,
    ) -> None:
        """
        Indexes a document into a corpus using the Vectara Structured Document format.

        Full API Docs: https://docs.vectara.com/docs/api-reference/indexing-apis/indexing#structured-document-object-definition

        Args:
            doc (Document): The document object to be indexed.
                You should provide the value you want for the document id in the corpus as the id_ member of this object.
                You should provide any document_metadata in the metadata member of this object.
            corpus_key (str): If multiple corpora are provided for this index, the corpus_key of the corpus you want to add the document to.
            title (str): The title of the document.
            description (str): The description of the document.
            max_chars_per_chunk (int): The maximum number of characters per chunk.
        """
        self._insert(
            document=doc,
            corpus_key=corpus_key,
            title=title,
            description=description,
            max_chars_per_chunk=max_chars_per_chunk,
        )

    def add_nodes(
        self,
        nodes: Sequence[Node],
        document_id: str,
        document_metadata: Optional[Dict] = {},
        corpus_key: Optional[str] = None,
    ) -> None:
        """
        Indexes a document into a corpus using the Vectara Core Document format.

        Full API Docs: https://docs.vectara.com/docs/api-reference/indexing-apis/indexing#core-document-object-definition

        Args:
            nodes (Sequence[Node]): The user-specified document parts.
                You should provide any part_metadata in the metadata member of each node.
            document_id (str): The document id (must be unique for the corpus).
            document_metadata (Dict): The document_metadata to be associated with this document.
            corpus_key (str): If multiple corpora are provided for this index, the corpus_key of the corpus you want to add the document to.
        """
        self._insert(
            nodes=nodes,
            corpus_key=corpus_key,
            doc_id=document_id,
            doc_metadata=document_metadata,
        )

    def insert_file(
        self,
        file_path: str,
        metadata: Optional[dict] = None,
        chunking_strategy: Optional[dict] = None,
        enable_table_extraction: Optional[bool] = False,
        filename: Optional[str] = None,
        corpus_key: Optional[str] = None,
        **insert_kwargs: Any,
    ) -> Optional[str]:
        """
        Vectara provides a way to add files (binary or text) directly via our API
        where pre-processing and chunking occurs internally in an optimal way
        This method provides a way to use that API in Llama_index.

        # ruff: noqa: E501
        Full API Docs: https://docs.vectara.com/docs/rest-api/upload-file

        Args:
            file_path: local file path
                Files could be text, HTML, PDF, markdown, doc/docx, ppt/pptx, etc.
                see API docs for full list
            metadata: Optional dict of metadata associated with the file
            chunking_strategy: Optional dict specifying max number of characters per chunk
            enable_table_extraction: Optional bool specifying whether or not to extract tables from document
            filename: Optional string specifying the filename


        Returns:
            List of ids associated with each of the files indexed
        """
        if not os.path.exists(file_path):
            _logger.error(f"File {file_path} does not exist")
            return None

        if filename is None:
            filename = file_path.split("/")[-1]

        files = {"file": (filename, open(file_path, "rb"))}

        if metadata:
            metadata["framework"] = "llama_index"
            files["metadata"] = (None, json.dumps(metadata), "application/json")

        if chunking_strategy:
            files["chunking_strategy"] = (
                None,
                json.dumps(chunking_strategy),
                "application/json",
            )

        if enable_table_extraction:
            files["table_extraction_config"] = (
                None,
                json.dumps({"extract_tables": enable_table_extraction}),
                "application/json",
            )

        headers = self._get_post_headers()
        headers.pop("Content-Type")
        valid_corpus_key = self._get_corpus_key(corpus_key)
        response = self._session.post(
            f"{self._base_url}/v2/corpora/{valid_corpus_key}/upload_file",
            files=files,
            verify=True,
            headers=headers,
            timeout=self.vectara_api_timeout,
        )

        res = response.json()
        if response.status_code == 201:
            doc_id = res["id"]
            self.doc_ids.append(doc_id)
            return doc_id
        elif response.status_code == 400:
            _logger.info(f"File upload failed with error message {res['field_errors']}")
            return None
        else:
            _logger.info(f"File upload failed with error message {res['messages'][0]}")
            return None

    def delete_ref_doc(
        self, ref_doc_id: str, delete_from_docstore: bool = True, **delete_kwargs: Any
    ) -> None:
        """
        Delete a document from a Vectara corpus.

        Args:
            ref_doc_id (str): ID of the document to delete
            delete_from_docstore (bool): Whether to delete the document from the corpus.
                If False, no change is made to the index or corpus.
            corpus_key (str): corpus key to delete the document from.
                This should be specified if there are multiple corpora in the index.
        """
        if delete_from_docstore:
            if "corpus_key" in delete_kwargs:
                self._delete_doc(
                    doc_id=ref_doc_id, corpus_key=delete_kwargs["corpus_key"]
                )
            else:
                self._delete_doc(doc_id=ref_doc_id)

    def update_ref_doc(self, document: Document, **update_kwargs: Any) -> None:
        """
        Update a document's metadata in a Vectara corpus.

        Args:
            document (Document): The document to update.
                Make sure to include id_ argument for proper identification within the corpus.
            corpus_key (str): corpus key to modify the document from.
                This should be specified if there are multiple corpora in the index.
            metadata (dict): dictionary specifying any modifications or additions to the document's metadata.
        """
        if "metadata" in update_kwargs:
            if "corpus_key" in update_kwargs:
                valid_corpus_key = self._get_corpus_key(update_kwargs["corpus_key"])
            else:
                valid_corpus_key = self._get_corpus_key(corpus_key=None)

            doc_id = document.doc_id
            body = {"metadata": update_kwargs["metadata"]}
            response = self._session.patch(
                f"{self._base_url}/v2/corpora/{valid_corpus_key}/documents/{doc_id}",
                data=json.dumps(body),
                verify=True,
                headers=self._get_post_headers(),
                timeout=self.vectara_api_timeout,
            )

            if response.status_code != 200:
                _logger.error(
                    f"Update request failed for doc_id = {doc_id} with status code "
                    f"{response.status_code}, text {response.json()['messages'][0]}"
                )

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Return a Retriever for this managed index."""
        from llama_index.indices.managed.vectara.retriever import (
            VectaraRetriever,
        )

        return VectaraRetriever(self, **kwargs)

    def as_chat_engine(self, **kwargs: Any) -> BaseChatEngine:
        kwargs["summary_enabled"] = True
        retriever = self.as_retriever(**kwargs)
        kwargs.pop("summary_enabled")
        from llama_index.indices.managed.vectara.query import (
            VectaraChatEngine,
        )

        return VectaraChatEngine.from_args(retriever, **kwargs)  # type: ignore

    def as_query_engine(
        self, llm: Optional[LLMType] = None, **kwargs: Any
    ) -> BaseQueryEngine:
        if kwargs.get("summary_enabled", True):
            from llama_index.indices.managed.vectara.query import (
                VectaraQueryEngine,
            )

            kwargs["summary_enabled"] = True
            retriever = self.as_retriever(**kwargs)
            return VectaraQueryEngine.from_args(retriever=retriever, **kwargs)  # type: ignore
        else:
            from llama_index.core.query_engine.retriever_query_engine import (
                RetrieverQueryEngine,
            )

            llm = (
                resolve_llm(llm, callback_manager=self._callback_manager)
                or Settings.llm
            )

            retriever = self.as_retriever(**kwargs)
            response_synthesizer = get_response_synthesizer(
                response_mode=ResponseMode.COMPACT,
                llm=llm,
            )
            return RetrieverQueryEngine.from_args(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
                **kwargs,
            )

    @classmethod
    def from_documents(
        cls: Type[IndexType],
        documents: Sequence[Document],
        show_progress: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        transformations: Optional[List[TransformComponent]] = None,
        **kwargs: Any,
    ) -> IndexType:
        """Build a Vectara index from a sequence of documents."""
        index = cls(
            show_progress=show_progress,
            **kwargs,
        )

        for doc in documents:
            index.add_document(doc)

        return index
