from llama_index.core.storage.kvstore.types import BaseKVStore
from llama_index.storage.kvstore.opensearch import OpensearchKVStore


def test_class():
    names_of_base_classes = [b.__name__ for b in OpensearchKVStore.__mro__]
    assert BaseKVStore.__name__ in names_of_base_classes
