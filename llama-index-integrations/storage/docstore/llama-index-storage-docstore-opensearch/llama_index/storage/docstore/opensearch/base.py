from typing import Optional

from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.core.storage.docstore.types import DEFAULT_BATCH_SIZE
from llama_index.storage.kvstore.opensearch import OpensearchKVStore


class OpensearchDocumentStore(KVDocumentStore):
    """Opensearch Document (Node) store.

    An Opensearch store for Document and Node objects.

    Args:
        opensearch_kvstore (OpensearchKVStore): Opensearch key-value store
        namespace (str): namespace for the docstore

    """

    def __init__(
        self,
        opensearch_kvstore: OpensearchKVStore,
        namespace: Optional[str] = None,
        node_collection_index: str = None,
        ref_doc_collection_index: str = None,
        metadata_collection_index: str = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Init a OpensearchDocumentStore."""
        super().__init__(opensearch_kvstore, namespace=namespace, batch_size=batch_size)
        if node_collection_index:
            self._node_collection = node_collection_index
        else:
            self._node_collection = f"llama_index-docstore.data-{self._namespace}"

        if ref_doc_collection_index:
            self._ref_doc_collection = ref_doc_collection_index
        else:
            self._ref_doc_collection = (
                f"llama_index-docstore.ref_doc_info-{self._namespace}"
            )

        if metadata_collection_index:
            self._metadata_collection = metadata_collection_index
        else:
            self._metadata_collection = (
                f"llama_index-docstore.metadata-{self._namespace}"
            )
