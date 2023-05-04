import os
from typing import Optional
from llama_index.storage.index_store.keyval_index_store import KVIndexStore
from llama_index.storage.kvstore.simple_kvstore import SimpleKVStore
from llama_index.storage.kvstore.types import BaseInMemoryKVStore
from llama_index.storage.index_store.types import (
    DEFAULT_PERSIST_DIR,
    DEFAULT_PERSIST_FNAME,
    DEFAULT_PERSIST_PATH,
)


class SimpleIndexStore(KVIndexStore):
    """Simple in-memory Index store.

    Args:
        simple_kvstore (SimpleKVStore): simple key-value store

    """

    def __init__(self, simple_kvstore: Optional[SimpleKVStore] = None) -> None:
        simple_kvstore = simple_kvstore or SimpleKVStore()
        super().__init__(simple_kvstore)

    @classmethod
    def from_persist_dir(
        cls, persist_dir: str = DEFAULT_PERSIST_DIR
    ) -> "SimpleIndexStore":
        """Create a SimpleIndexStore from a persist directory."""
        persist_path = os.path.join(persist_dir, DEFAULT_PERSIST_FNAME)
        return cls.from_persist_path(persist_path)

    @classmethod
    def from_persist_path(cls, persist_path: str) -> "SimpleIndexStore":
        """Create a SimpleIndexStore from a persist path."""
        simple_kvstore = SimpleKVStore.from_persist_path(persist_path)
        return cls(simple_kvstore)

    def persist(self, persist_path: str = DEFAULT_PERSIST_PATH) -> None:
        """Persist the store."""
        if isinstance(self._kvstore, BaseInMemoryKVStore):
            self._kvstore.persist(persist_path)
    
    @classmethod
    def from_dict(cls, save_dict: dict) -> "SimpleIndexStore":
        simple_kvstore = SimpleKVStore.from_dict(save_dict)
        return cls(simple_kvstore)

    def to_dict(self) -> dict:
        assert isinstance(self._kvstore, SimpleKVStore)
        return self._kvstore.to_dict()
