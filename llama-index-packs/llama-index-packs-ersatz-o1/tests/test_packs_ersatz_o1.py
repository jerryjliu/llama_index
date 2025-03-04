from llama_index.core.query_engine import CustomQueryEngine
from llama_index.packs.ersatz_o1 import ErsatzO1QueryEngine


def test_class():
    names_of_base_classes = [b.__name__ for b in ErsatzO1QueryEngine.__mro__]
    assert CustomQueryEngine.__name__ in names_of_base_classes
