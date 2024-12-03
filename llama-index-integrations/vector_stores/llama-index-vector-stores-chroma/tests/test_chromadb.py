import os
from typing import Dict, List, Generator

import pytest
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)


PARAMS: Dict[str, str] = {
    "host": os.environ.get("CHROMADB_HOST", "localhost"),
    "port": os.environ.get("CHROMADB_PORT", "8000"),
}
COLLECTION_NAME = "llama_collection"

try:
    import chromadb

    # connection check
    conn__ = chromadb.HttpClient(**PARAMS)  # type: ignore
    conn__.get_or_create_collection(COLLECTION_NAME)

    http_client_chromadb_mode = True
except (ImportError, Exception):
    http_client_chromadb_mode = False


# To test chromadb http-client functionality do:
# cd tests
# docker-compose up
#
@pytest.mark.skipif(
    http_client_chromadb_mode is False,
    reason="chromadb is not running in http client mode",
)
def test_instance_creation_from_http_params() -> None:
    store = ChromaVectorStore.from_params(
        host=PARAMS["host"],
        port=PARAMS["port"],
        collection_name=COLLECTION_NAME,
        collection_kwargs={},
    )
    assert isinstance(store, ChromaVectorStore)


def test_instance_creation_from_collection() -> None:
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    store = ChromaVectorStore.from_collection(collection)
    assert isinstance(store, ChromaVectorStore)


def test_instance_creation_from_persist_dir() -> None:
    store = ChromaVectorStore.from_params(
        persist_dir="./data",
        collection_name=COLLECTION_NAME,
        collection_kwargs={},
    )
    assert isinstance(store, ChromaVectorStore)


@pytest.fixture()
def vector_store() -> Generator[ChromaVectorStore, None, None]:
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    yield ChromaVectorStore(chroma_collection=collection)
    chroma_client.delete_collection(name=COLLECTION_NAME)


@pytest.fixture(scope="session")
def node_embeddings() -> List[TextNode]:
    return [
        TextNode(
            text="lorem ipsum",
            id_="c330d77f-90bd-4c51-9ed2-57d8d693b3b0",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")},
            metadata={
                "author": "Stephen King",
                "theme": "Friendship",
            },
            embedding=[1.0, 0.0, 0.0],
        ),
        TextNode(
            text="lorem ipsum",
            id_="c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-1")},
            metadata={
                "director": "Francis Ford Coppola",
                "theme": "Mafia",
            },
            embedding=[0.0, 1.0, 0.0],
        ),
        TextNode(
            text="lorem ipsum",
            id_="c3ew11cd-8fb4-4b8f-b7ea-7fa96038d39d",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-2")},
            metadata={
                "director": "Christopher Nolan",
            },
            embedding=[0.0, 0.0, 1.0],
        ),
        TextNode(
            text="I was taught that the way of progress was neither swift nor easy.",
            id_="0b31ae71-b797-4e88-8495-031371a7752e",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="text-3")},
            metadata={
                "author": "Marie Curie",
            },
            embedding=[0.0, 0.0, 0.9],
        ),
        TextNode(
            text=(
                "The important thing is not to stop questioning."
                + " Curiosity has its own reason for existing."
            ),
            id_="bd2e080b-159a-4030-acc3-d98afd2ba49b",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="text-4")},
            metadata={
                "author": "Albert Einstein",
            },
            embedding=[0.0, 0.0, 0.5],
        ),
        TextNode(
            text=(
                "I am no bird; and no net ensnares me;"
                + " I am a free human being with an independent will."
            ),
            id_="f658de3b-8cef-4d1c-8bed-9a263c907251",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="text-5")},
            metadata={
                "author": "Charlotte Bronte",
            },
            embedding=[0.0, 0.0, 0.3],
        ),
    ]


@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_chromadb_and_query(
    vector_store: ChromaVectorStore,
    node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    if use_async:
        await vector_store.async_add(node_embeddings, index_id="42")
        res = await vector_store.aquery(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1), index_id="42"
        )
    else:
        vector_store.add(node_embeddings, index_id="42")
        res = vector_store.query(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1), index_id="42"
        )
    assert res.nodes
    assert res.nodes[0].get_content() == "lorem ipsum"


@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_chromadb_and_query_by_metafilters_only(
    vector_store: ChromaVectorStore,
    node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="author", value="Marie Curie", operator=FilterOperator.EQ
            )
        ],
        condition=FilterCondition.AND,
    )

    if use_async:
        await vector_store.async_add(node_embeddings, index_id="42")
        res = await vector_store.aquery(
            VectorStoreQuery(filters=filters, similarity_top_k=1),
            index_id="42"
        )
    else:
        vector_store.add(node_embeddings, index_id="42")
        res = vector_store.query(VectorStoreQuery(filters=filters, similarity_top_k=1), index_id="42")

    assert (
        res.nodes[0].get_content()
        == "I was taught that the way of progress was neither swift nor easy."
    )


def test_get_nodes(
    vector_store: ChromaVectorStore, node_embeddings: List[TextNode]
) -> None:
    vector_store.add(node_embeddings, index_id="42")
    res = vector_store.get_nodes(
        node_ids=[
            "c330d77f-90bd-4c51-9ed2-57d8d693b3b0",
            "c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d",
            "c3ew11cd-8fb4-4b8f-b7ea-7fa96038d39d",
        ],
        index_id="42"
    )
    assert len(res) == 3
    assert res[0].get_content() == "lorem ipsum"
    assert res[1].get_content() == "lorem ipsum"
    assert res[2].get_content() == "lorem ipsum"


def test_delete_nodes(
    vector_store: ChromaVectorStore, node_embeddings: List[TextNode]
) -> None:
    vector_store.add(node_embeddings, index_id="42")
    vector_store.delete_nodes(
        node_ids=[
            "c330d77f-90bd-4c51-9ed2-57d8d693b3b0",
            "c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d",
        ],
        index_id="42"
    )
    res = vector_store.get_nodes(
        node_ids=[
            "c330d77f-90bd-4c51-9ed2-57d8d693b3b0",
            "c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d",
            "c3ew11cd-8fb4-4b8f-b7ea-7fa96038d39d",
        ],
        index_id = "42"
    )
    assert len(res) == 1
    assert res[0].get_content() == "lorem ipsum"
    assert res[0].id_ == "c3ew11cd-8fb4-4b8f-b7ea-7fa96038d39d"


def test_clear(
    vector_store: ChromaVectorStore, node_embeddings: List[TextNode]
) -> None:
    vector_store.add(node_embeddings, index_id="42")
    vector_store.clear(index_id="42")
    res = vector_store.get_nodes(
        node_ids=[
            "c330d77f-90bd-4c51-9ed2-57d8d693b3b0",
            "c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d",
            "c3ew11cd-8fb4-4b8f-b7ea-7fa96038d39d",
        ],
        index_id="42"
    )
    assert len(res) == 0
