import pytest
from unittest.mock import MagicMock, patch

from llama_index.core.schema import TextNode
from llama_index.vector_stores.falkordb import FalkorDBVectorStore
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    MetadataFilters,
    ExactMatchFilter,
)


# Mock FalkorDB client
class MockFalkorDBClient:
    def __init__(self) -> None:
        self.nodes = {}
        self.query_results = []
        self.index_exists = False

    def query(self, query, params=None):
        if "CREATE VECTOR INDEX" in query:
            self.index_exists = True
            return MagicMock()
        elif "SHOW INDEXES" in query:
            if self.index_exists:
                return MagicMock(result_set=[{"name": "test_index"}])
            return MagicMock(result_set=[])
        elif "MATCH (n:Chunk)" in query and "RETURN" in query:
            # Query operation
            return MagicMock(result_set=self.query_results)
        elif "MERGE (c:Chunk" in query:
            # Add operation
            if params and "data" in params:
                for node in params["data"]:
                    self.nodes[node["id"]] = node
            return MagicMock()
        elif "MATCH (n:Chunk) WHERE n.id" in query and "DELETE" in query:
            # Delete operation
            node_id = params["id"]
            if node_id in self.nodes:
                del self.nodes[node_id]
            return MagicMock()
        elif "MATCH (n:Chunk) WHERE n.id" in query:
            # Get operation
            node_id = params["id"]
            if node_id in self.nodes:
                return MagicMock(result_set=[self.nodes[node_id]])
            return MagicMock(result_set=[])
        return MagicMock()

    def set_query_results(self, results):
        self.query_results = results


@pytest.fixture()
def mock_falkordb():
    with patch("falkordb.FalkorDB") as mock:
        client = MockFalkorDBClient()
        mock.return_value = client
        mock.from_url.return_value.select_graph.return_value = client
        yield client


@pytest.fixture()
def falkordb_store(mock_falkordb):
    store = FalkorDBVectorStore(
        url="redis://localhost:6379",
        database="testdb",
        index_name="test_index",
        node_label="Chunk",
        embedding_node_property="embedding",
        text_node_property="text",
    )
    # Ensure store uses the mock client
    store._client = mock_falkordb
    return store


def test_falkordb_add(falkordb_store):
    nodes = [
        TextNode(
            text="Hello world",
            id_="1",
            embedding=[1.0, 0.0, 0.0],
            metadata={"key": "value"},
        ),
        TextNode(
            text="Hello world 2",
            id_="2",
            embedding=[0.0, 1.0, 0.0],
            metadata={"key2": "value2"},
        ),
    ]
    ids = falkordb_store.add(nodes)
    assert ids == ["1", "2"]
    assert len(falkordb_store._client.nodes) == 2
    assert falkordb_store._client.nodes["1"]["text"] == "Hello world"
    assert falkordb_store._client.nodes["2"]["text"] == "Hello world 2"


def test_falkordb_delete(falkordb_store):
    node = TextNode(
        text="Hello world",
        id_="test_node",
        embedding=[1.0, 0.0, 0.0],
    )
    falkordb_store.add([node])
    assert "test_node" in falkordb_store._client.nodes

    falkordb_store.delete("test_node")
    assert "test_node" not in falkordb_store._client.nodes


def test_falkordb_query(falkordb_store):
    mock_results = [
        {
            "n": {
                "text": "Hello world",
                "id": "1",
                "embedding": [1.0, 0.0, 0.0],
                "metadata": {"key": "value"},
            },
            "score": 0.9,
        },
        {
            "n": {
                "text": "Hello world 2",
                "id": "2",
                "embedding": [0.0, 1.0, 0.0],
                "metadata": {"key2": "value2"},
            },
            "score": 0.7,
        },
    ]
    falkordb_store._client.set_query_results(mock_results)

    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0],
        similarity_top_k=2,
    )
    results = falkordb_store.query(query)

    assert len(results.nodes) == 2
    assert results.nodes[0].text == "Hello world"
    assert results.nodes[1].text == "Hello world 2"
    assert results.similarities == [0.9, 0.7]


def test_falkordb_query_with_filters(falkordb_store):
    mock_results = [
        {
            "n": {
                "text": "Hello world",
                "id": "1",
                "embedding": [1.0, 0.0, 0.0],
                "metadata": {"key": "value"},
            },
            "score": 0.9,
        },
    ]
    falkordb_store._client.set_query_results(mock_results)

    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0],
        similarity_top_k=2,
        filters=MetadataFilters(filters=[ExactMatchFilter(key="key", value="value")]),
    )
    results = falkordb_store.query(query)

    assert len(results.nodes) == 1
    assert results.nodes[0].text == "Hello world"
    assert results.similarities == [0.9]


def test_falkordb_update(falkordb_store):
    node = TextNode(
        text="Original text",
        id_="update_node",
        embedding=[1.0, 0.0, 0.0],
    )
    falkordb_store.add([node])

    updated_node = TextNode(
        text="Updated text",
        id_="update_node",
        embedding=[0.0, 1.0, 0.0],
    )
    falkordb_store.update(updated_node)

    assert falkordb_store._client.nodes["update_node"]["text"] == "Updated text"
    assert falkordb_store._client.nodes["update_node"]["embedding"] == [0.0, 1.0, 0.0]


def test_falkordb_get(falkordb_store):
    node = TextNode(
        text="Get test",
        id_="get_node",
        embedding=[1.0, 1.0, 1.0],
    )
    falkordb_store.add([node])

    retrieved_node = falkordb_store.get("get_node")
    assert retrieved_node is not None
    assert retrieved_node.text == "Get test"
    assert retrieved_node.embedding == [1.0, 1.0, 1.0]


def test_falkordb_nonexistent_get(falkordb_store):
    retrieved_node = falkordb_store.get("nonexistent_node")
    assert retrieved_node is None


if __name__ == "__main__":
    pytest.main()
