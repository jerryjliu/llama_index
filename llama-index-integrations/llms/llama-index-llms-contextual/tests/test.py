from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.contextual import Contextual

def test_llm_class():
    names_of_base_classes = [b.__name__ for b in Contextual.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes
