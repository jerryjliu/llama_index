import os

import jwt  # noqa
import pytest
from llama_index.core import Document
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.vector_stores.deeplake import DeepLakeVectorStore

if os.getenv("GITHUB_ACTIONS") == "true":
    pytest.skip("tests are flaky on Github runners", allow_module_level=True)


def test_class():
    names_of_base_classes = [b.__name__ for b in DeepLakeVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


@pytest.fixture()
def vs_ids():
    vs = DeepLakeVectorStore(dataset_path="mem://test", overwrite=True)
    ids = vs.add(
        nodes=[
            Document(text="Doc 1", embedding=[1, 2, 1], metadata={"a": "1", "b": 10}),
            Document(text="Doc 2", embedding=[1, 2, 2], metadata={"a": "2", "b": 11}),
            Document(text="Doc 3", embedding=[1, 2, 3], metadata={"a": "3", "b": 12}),
        ]
    )
    yield (vs, ids)
    vs.clear()


def test_filters(vs_ids):
    vs, ids = vs_ids

    nodes = vs.get_nodes(node_ids=[ids[0], ids[2]])
    assert [x.text for x in nodes] == ["Doc 1", "Doc 3"]

    nodes = vs.get_nodes(node_ids=["a"])
    assert len(nodes) == 0

    nodes = vs.get_nodes(
        filters=MetadataFilters(filters=[MetadataFilter(key="a", value="2")])
    )
    assert [x.text for x in nodes] == ["Doc 2"]

    nodes = vs.get_nodes(
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="a", value="2"),
                MetadataFilter(key="a", value="3"),
            ]
        )
    )
    assert [x.text for x in nodes] == []

    nodes = vs.get_nodes(
        filters=MetadataFilters(
            condition=FilterCondition.OR,
            filters=[
                MetadataFilter(key="a", value="2"),
                MetadataFilter(key="a", value="3"),
            ],
        )
    )
    assert [x.text for x in nodes] == ["Doc 2", "Doc 3"]

    nodes = vs.get_nodes(
        filters=MetadataFilters(
            condition=FilterCondition.OR,
            filters=[
                MetadataFilter(key="a", value="2"),
                MetadataFilter(key="a", value="3"),
            ],
        )
    )
    assert [x.text for x in nodes] == ["Doc 2", "Doc 3"]

    nodes = vs.get_nodes(
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="b", value=10, operator=FilterOperator.GT),
            ]
        )
    )
    assert [x.text for x in nodes] == ["Doc 2", "Doc 3"]

    nodes = vs.get_nodes(
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="b", value=11, operator=FilterOperator.LTE),
            ]
        )
    )
    assert [x.text for x in nodes] == ["Doc 1", "Doc 2"]


def test_delete_id(vs_ids):
    vs, ids = vs_ids
    vs.delete_nodes(node_ids=[ids[0], ids[2]])
    assert [x.text for x in vs.get_nodes()] == ["Doc 2"]


def test_delete_filter(vs_ids):
    vs, ids = vs_ids
    vs.delete_nodes(
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="b", value=10, operator=FilterOperator.GT),
            ]
        )
    )
    assert [x.text for x in vs.get_nodes()] == ["Doc 1"]
