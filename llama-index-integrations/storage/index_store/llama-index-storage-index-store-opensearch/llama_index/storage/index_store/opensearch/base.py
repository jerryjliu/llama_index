from typing import Optional

from llama_index.core.storage.index_store.keyval_index_store import KVIndexStore
from llama_index.storage.kvstore.opensearch import OpensearchKVStore


class OpensearchIndexStore(KVIndexStore):
    """Opensearch Index store.

    Args:
        opensearch_kvstore (OpensearchKVStore): Opensearch key-value store
        namespace (str): namespace for the index store

    """

    def __init__(
        self,
        opensearch_kvstore: OpensearchKVStore,
        collection_index: Optional[str] = None,
        namespace: Optional[str] = None,
        collection_suffix: Optional[str] = None,
    ) -> None:
        """Init a OpensearchIndexStore."""
        super().__init__(
            opensearch_kvstore,
            namespace=namespace,
            collection_suffix=collection_suffix,
        )
        if collection_index:
            self._collection = collection_index
        else:
            self._collection = f"llama_index-index_store.data-{self._namespace}"
