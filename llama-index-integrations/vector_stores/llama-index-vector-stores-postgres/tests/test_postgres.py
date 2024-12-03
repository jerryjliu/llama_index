import asyncio
from typing import Any, Dict, Generator, List, Union, Optional

import pytest
from llama_index.core.schema import (
    BaseNode,
    IndexNode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.postgres import PGVectorStore

# from testing find install here https://github.com/pgvector/pgvector#installation-notes


PARAMS: Dict[str, Union[str, int]] = {
    "host": "localhost",
    "user": "postgres",
    "password": "password",
    "port": 5432,
}
TEST_DB = "test_vector_db"
TEST_DB_HNSW = "test_vector_db_hnsw"
TEST_TABLE_NAME = "lorem_ipsum"
TEST_SCHEMA_NAME = "test"
TEST_EMBED_DIM = 2

try:
    import asyncpg  # noqa
    import pgvector  # noqa
    import psycopg2
    import sqlalchemy
    import sqlalchemy.ext.asyncio  # noqa

    # connection check
    conn__ = psycopg2.connect(**PARAMS)  # type: ignore
    conn__.close()

    postgres_not_available = False
except (ImportError, Exception):
    postgres_not_available = True


def _get_sample_vector(num: float) -> List[float]:
    """
    Get sample embedding vector of the form [num, 1, 1, ..., 1]
    where the length of the vector is TEST_EMBED_DIM.
    """
    return [num] + [1.0] * (TEST_EMBED_DIM - 1)


@pytest.fixture(scope="session")
def conn() -> Any:
    import psycopg2

    return psycopg2.connect(**PARAMS)  # type: ignore


@pytest.fixture()
def db(conn: Any) -> Generator:
    conn.autocommit = True

    with conn.cursor() as c:
        c.execute(f"DROP DATABASE IF EXISTS {TEST_DB}")
        c.execute(f"CREATE DATABASE {TEST_DB}")
        conn.commit()
    yield
    with conn.cursor() as c:
        c.execute(f"DROP DATABASE {TEST_DB}")
        conn.commit()


@pytest.fixture()
def db_hnsw(conn: Any) -> Generator:
    conn.autocommit = True

    with conn.cursor() as c:
        c.execute(f"DROP DATABASE IF EXISTS {TEST_DB_HNSW}")
        c.execute(f"CREATE DATABASE {TEST_DB_HNSW}")
        conn.commit()
    yield
    with conn.cursor() as c:
        c.execute(f"DROP DATABASE {TEST_DB_HNSW}")
        conn.commit()


@pytest.fixture()
def pg(db: None) -> Any:
    pg = PGVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB,
        table_name=TEST_TABLE_NAME,
        schema_name=TEST_SCHEMA_NAME,
        embed_dim=TEST_EMBED_DIM,
    )

    yield pg

    asyncio.run(pg.close())


@pytest.fixture()
def pg_hybrid(db: None) -> Any:
    pg = PGVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB,
        table_name=TEST_TABLE_NAME,
        schema_name=TEST_SCHEMA_NAME,
        hybrid_search=True,
        embed_dim=TEST_EMBED_DIM,
    )

    yield pg

    asyncio.run(pg.close())


@pytest.fixture()
def pg_hnsw(db_hnsw: None) -> Any:
    pg = PGVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB_HNSW,
        table_name=TEST_TABLE_NAME,
        schema_name=TEST_SCHEMA_NAME,
        embed_dim=TEST_EMBED_DIM,
        hnsw_kwargs={"hnsw_m": 16, "hnsw_ef_construction": 64, "hnsw_ef_search": 40},
    )

    yield pg

    asyncio.run(pg.close())


@pytest.fixture()
def pg_hnsw_hybrid(db_hnsw: None) -> Any:
    pg = PGVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB_HNSW,
        table_name=TEST_TABLE_NAME,
        schema_name=TEST_SCHEMA_NAME,
        embed_dim=TEST_EMBED_DIM,
        hybrid_search=True,
        hnsw_kwargs={"hnsw_m": 16, "hnsw_ef_construction": 64, "hnsw_ef_search": 40},
    )

    yield pg

    asyncio.run(pg.close())


@pytest.fixture()
def pg_hnsw_multiple(db_hnsw: None) -> Generator[List[PGVectorStore], None, None]:
    """
    This creates multiple instances of PGVectorStore.
    """
    pgs = []
    for _ in range(2):
        pg = PGVectorStore.from_params(
            **PARAMS,  # type: ignore
            database=TEST_DB_HNSW,
            table_name=TEST_TABLE_NAME,
            schema_name=TEST_SCHEMA_NAME,
            embed_dim=TEST_EMBED_DIM,
            hnsw_kwargs={
                "hnsw_m": 16,
                "hnsw_ef_construction": 64,
                "hnsw_ef_search": 40,
            },
        )
        pgs.append(pg)

    yield pgs

    for pg in pgs:
        asyncio.run(pg.close())


@pytest.fixture(scope="session")
def node_embeddings() -> List[TextNode]:
    return [
        TextNode(
            text="lorem ipsum",
            id_="aaa",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="aaa")},
            extra_info={"test_num": 1},
            embedding=_get_sample_vector(1.0),
        ),
        TextNode(
            text="dolor sit amet",
            id_="bbb",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="bbb")},
            extra_info={"test_key": "test_value"},
            embedding=_get_sample_vector(0.1),
        ),
        TextNode(
            text="consectetur adipiscing elit",
            id_="ccc",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ccc")},
            extra_info={"test_key_list": ["test_value"]},
            embedding=_get_sample_vector(0.1),
        ),
        TextNode(
            text="sed do eiusmod tempor",
            id_="ddd",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ccc")},
            extra_info={"test_key_2": "test_val_2"},
            embedding=_get_sample_vector(0.1),
        ),
    ]


@pytest.fixture(scope="session")
def hybrid_node_embeddings() -> List[TextNode]:
    return [
        TextNode(
            text="lorem ipsum",
            id_="aaa",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="aaa")},
            embedding=_get_sample_vector(0.1),
        ),
        TextNode(
            text="dolor sit amet",
            id_="bbb",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="bbb")},
            extra_info={"test_key": "test_value"},
            embedding=_get_sample_vector(1.0),
        ),
        TextNode(
            text="The quick brown fox jumped over the lazy dog.",
            id_="ccc",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ccc")},
            embedding=_get_sample_vector(5.0),
        ),
        TextNode(
            text="The fox and the hound",
            id_="ddd",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ddd")},
            extra_info={"test_key": "test_value"},
            embedding=_get_sample_vector(10.0),
        ),
    ]


@pytest.fixture(scope="session")
def index_node_embeddings() -> List[TextNode]:
    return [
        TextNode(
            text="lorem ipsum",
            id_="aaa",
            embedding=_get_sample_vector(0.1),
        ),
        TextNode(
            text="dolor sit amet",
            id_="bbb",
            extra_info={"test_key": "test_value"},
            embedding=_get_sample_vector(1.0),
        ),
        IndexNode(
            text="The quick brown fox jumped over the lazy dog.",
            id_="aaa_ref",
            index_id="aaa",
            embedding=_get_sample_vector(5.0),
        ),
    ]


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio()
async def test_instance_creation(db: None) -> None:
    pg = PGVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB,
        table_name=TEST_TABLE_NAME,
        schema_name=TEST_SCHEMA_NAME,
    )
    assert isinstance(pg, PGVectorStore)
    assert not hasattr(pg, "_engine")
    assert pg.client is None
    await pg.close()


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_db_and_query(
    pg: PGVectorStore, node_embeddings: List[TextNode], use_async: bool
) -> None:
    if use_async:
        await pg.async_add(node_embeddings, index_id="42")
    else:
        pg.add(node_embeddings, index_id="42")
    assert isinstance(pg, PGVectorStore)
    assert hasattr(pg, "_engine")
    q = VectorStoreQuery(query_embedding=_get_sample_vector(1.0), similarity_top_k=1)
    if use_async:
        res = await pg.aquery(q, index_id="42")
    else:
        res = pg.query(q, index_id="42")
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "aaa"


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [True, False])
async def test_query_hnsw(
    pg_hnsw: PGVectorStore, node_embeddings: List[TextNode], use_async: bool
):
    if use_async:
        await pg_hnsw.async_add(node_embeddings, index_id="42")
    else:
        pg_hnsw.add(node_embeddings, index_id="42")

    assert isinstance(pg_hnsw, PGVectorStore)
    assert hasattr(pg_hnsw, "_engine")

    q = VectorStoreQuery(query_embedding=_get_sample_vector(1.0), similarity_top_k=1)
    if use_async:
        res = await pg_hnsw.aquery(q, index_id="42")
    else:
        res = pg_hnsw.query(q, index_id="42")
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "aaa"


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_db_and_query_with_metadata_filters(
    pg: PGVectorStore, node_embeddings: List[TextNode], use_async: bool
) -> None:
    if use_async:
        await pg.async_add(node_embeddings, index_id="42")
    else:
        pg.add(node_embeddings, index_id="42")
    assert isinstance(pg, PGVectorStore)
    assert hasattr(pg, "_engine")
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="test_key", value="test_value")]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )
    if use_async:
        res = await pg.aquery(q, index_id="42")
    else:
        res = pg.query(q, index_id="42")
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "bbb"


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_db_and_query_with_metadata_filters_with_in_operator(
    pg: PGVectorStore, node_embeddings: List[TextNode], use_async: bool
) -> None:
    if use_async:
        await pg.async_add(node_embeddings, index_id="42")
    else:
        pg.add(node_embeddings, index_id="42")
    assert isinstance(pg, PGVectorStore)
    assert hasattr(pg, "_engine")
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_key",
                value=["test_value", "another_value"],
                operator=FilterOperator.IN,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )
    if use_async:
        res = await pg.aquery(q, index_id="42")
    else:
        res = pg.query(q, index_id="42")
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "bbb"


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_db_and_query_with_metadata_filters_with_in_operator_and_single_element(
    pg: PGVectorStore, node_embeddings: List[TextNode], use_async: bool
) -> None:
    if use_async:
        await pg.async_add(node_embeddings, index_id="42")
    else:
        pg.add(node_embeddings, index_id="42")
    assert isinstance(pg, PGVectorStore)
    assert hasattr(pg, "_engine")
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_key",
                value=["test_value"],
                operator=FilterOperator.IN,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )
    if use_async:
        res = await pg.aquery(q, index_id="42")
    else:
        res = pg.query(q, index_id="42")
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "bbb"


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_db_and_query_with_metadata_filters_with_contains_operator(
    pg: PGVectorStore, node_embeddings: List[TextNode], use_async: bool
) -> None:
    if use_async:
        await pg.async_add(node_embeddings, index_id="42")
    else:
        pg.add(node_embeddings, index_id="42")
    assert isinstance(pg, PGVectorStore)
    assert hasattr(pg, "_engine")
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_key_list",
                value="test_value",
                operator=FilterOperator.CONTAINS,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )
    if use_async:
        res = await pg.aquery(q, index_id="42")
    else:
        res = pg.query(q, index_id="42")
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "ccc"


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_db_query_and_delete(
    pg: PGVectorStore, node_embeddings: List[TextNode], use_async: bool
) -> None:
    if use_async:
        await pg.async_add(node_embeddings, index_id="42")
    else:
        pg.add(node_embeddings, index_id="42")
    assert isinstance(pg, PGVectorStore)
    assert hasattr(pg, "_engine")

    q = VectorStoreQuery(query_embedding=_get_sample_vector(0.1), similarity_top_k=1)

    if use_async:
        res = await pg.aquery(q, index_id="42")
    else:
        res = pg.query(q, index_id="42")
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "bbb"


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [True, False])
async def test_sparse_query(
    pg_hybrid: PGVectorStore,
    hybrid_node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    if use_async:
        await pg_hybrid.async_add(hybrid_node_embeddings, index_id="42")
    else:
        pg_hybrid.add(hybrid_node_embeddings, index_id="42")
    assert isinstance(pg_hybrid, PGVectorStore)
    assert hasattr(pg_hybrid, "_engine")

    # text search should work when query is a sentence and not just a single word
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="who is the fox?",
        sparse_top_k=2,
        mode=VectorStoreQueryMode.SPARSE,
    )

    if use_async:
        res = await pg_hybrid.aquery(q, index_id="42")
    else:
        res = pg_hybrid.query(q, index_id="42")
    assert res.nodes
    assert len(res.nodes) == 2
    assert res.nodes[0].node_id == "ccc"
    assert res.nodes[1].node_id == "ddd"


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [True, False])
async def test_hybrid_query(
    pg_hybrid: PGVectorStore,
    hybrid_node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    if use_async:
        await pg_hybrid.async_add(hybrid_node_embeddings, index_id="42")
    else:
        pg_hybrid.add(hybrid_node_embeddings, index_id="42")
    assert isinstance(pg_hybrid, PGVectorStore)
    assert hasattr(pg_hybrid, "_engine")

    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="fox",
        similarity_top_k=2,
        mode=VectorStoreQueryMode.HYBRID,
        sparse_top_k=1,
    )

    if use_async:
        res = await pg_hybrid.aquery(q, index_id="42")
    else:
        res = pg_hybrid.query(q, index_id="42")
    assert res.nodes
    assert len(res.nodes) == 3
    assert res.nodes[0].node_id == "aaa"
    assert res.nodes[1].node_id == "bbb"
    assert res.nodes[2].node_id == "ccc"

    # if sparse_top_k is not specified, it should default to similarity_top_k
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="fox",
        similarity_top_k=2,
        mode=VectorStoreQueryMode.HYBRID,
    )

    if use_async:
        res = await pg_hybrid.aquery(q, index_id="42")
    else:
        res = pg_hybrid.query(q, index_id="42")
    assert res.nodes
    assert len(res.nodes) == 4
    assert res.nodes[0].node_id == "aaa"
    assert res.nodes[1].node_id == "bbb"
    assert res.nodes[2].node_id == "ccc"
    assert res.nodes[3].node_id == "ddd"

    # text search should work when query is a sentence and not just a single word
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="who is the fox?",
        similarity_top_k=2,
        mode=VectorStoreQueryMode.HYBRID,
    )

    if use_async:
        res = await pg_hybrid.aquery(q, index_id="42")
    else:
        res = pg_hybrid.query(q, index_id="42")
    assert res.nodes
    assert len(res.nodes) == 4
    assert res.nodes[0].node_id == "aaa"
    assert res.nodes[1].node_id == "bbb"
    assert res.nodes[2].node_id == "ccc"
    assert res.nodes[3].node_id == "ddd"


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [True, False])
async def test_hybrid_query(
    pg_hnsw_hybrid: PGVectorStore,
    hybrid_node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    if use_async:
        await pg_hnsw_hybrid.async_add(hybrid_node_embeddings, index_id="42")
    else:
        pg_hnsw_hybrid.add(hybrid_node_embeddings, index_id="42")
    assert isinstance(pg_hnsw_hybrid, PGVectorStore)
    assert hasattr(pg_hnsw_hybrid, "_engine")

    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="fox",
        similarity_top_k=2,
        mode=VectorStoreQueryMode.HYBRID,
        sparse_top_k=1,
    )

    if use_async:
        res = await pg_hnsw_hybrid.aquery(q, index_id="42")
    else:
        res = pg_hnsw_hybrid.query(q, index_id="42")
    assert res.nodes
    assert len(res.nodes) == 3
    assert res.nodes[0].node_id == "aaa"
    assert res.nodes[1].node_id == "bbb"
    assert res.nodes[2].node_id == "ccc"

    # if sparse_top_k is not specified, it should default to similarity_top_k
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="fox",
        similarity_top_k=2,
        mode=VectorStoreQueryMode.HYBRID,
    )

    if use_async:
        res = await pg_hnsw_hybrid.aquery(q, index_id="42")
    else:
        res = pg_hnsw_hybrid.query(q, index_id="42")
    assert res.nodes
    assert len(res.nodes) == 4
    assert res.nodes[0].node_id == "aaa"
    assert res.nodes[1].node_id == "bbb"
    assert res.nodes[2].node_id == "ccc"
    assert res.nodes[3].node_id == "ddd"

    # text search should work when query is a sentence and not just a single word
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="who is the fox?",
        similarity_top_k=2,
        mode=VectorStoreQueryMode.HYBRID,
    )

    if use_async:
        res = await pg_hnsw_hybrid.aquery(q, index_id="42")
    else:
        res = pg_hnsw_hybrid.query(q, index_id="42")
    assert res.nodes
    assert len(res.nodes) == 4
    assert res.nodes[0].node_id == "aaa"
    assert res.nodes[1].node_id == "bbb"
    assert res.nodes[2].node_id == "ccc"
    assert res.nodes[3].node_id == "ddd"


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_db_and_hybrid_query_with_metadata_filters(
    pg_hybrid: PGVectorStore,
    hybrid_node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    if use_async:
        await pg_hybrid.async_add(hybrid_node_embeddings, index_id="42")
    else:
        pg_hybrid.add(hybrid_node_embeddings, index_id="42")
    assert isinstance(pg_hybrid, PGVectorStore)
    assert hasattr(pg_hybrid, "_engine")
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="test_key", value="test_value")]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="fox",
        similarity_top_k=10,
        filters=filters,
        mode=VectorStoreQueryMode.HYBRID,
    )
    if use_async:
        res = await pg_hybrid.aquery(q, index_id="42")
    else:
        res = pg_hybrid.query(q, index_id="42")
    assert res.nodes
    assert len(res.nodes) == 2
    assert res.nodes[0].node_id == "bbb"
    assert res.nodes[1].node_id == "ddd"


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
def test_hybrid_query_fails_if_no_query_str_provided(
    pg_hybrid: PGVectorStore, hybrid_node_embeddings: List[TextNode]
) -> None:
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(1.0),
        similarity_top_k=10,
        mode=VectorStoreQueryMode.HYBRID,
    )

    with pytest.raises(Exception) as exc:
        pg_hybrid.query(q, index_id="42")

        assert str(exc) == "query_str must be specified for a sparse vector query."


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_db_and_query_index_nodes(
    pg: PGVectorStore, index_node_embeddings: List[BaseNode], use_async: bool
) -> None:
    if use_async:
        await pg.async_add(index_node_embeddings, index_id="42")
    else:
        pg.add(index_node_embeddings, index_id="42")
    assert isinstance(pg, PGVectorStore)
    assert hasattr(pg, "_engine")
    q = VectorStoreQuery(query_embedding=_get_sample_vector(5.0), similarity_top_k=2)
    if use_async:
        res = await pg.aquery(q, index_id="42")
    else:
        res = pg.query(q, index_id="42")
    assert res.nodes
    assert len(res.nodes) == 2
    assert res.nodes[0].node_id == "aaa_ref"
    assert isinstance(res.nodes[0], IndexNode)
    assert hasattr(res.nodes[0], "index_id")
    assert res.nodes[1].node_id == "bbb"
    assert isinstance(res.nodes[1], TextNode)


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [True, False])
async def test_delete_nodes(
    pg: PGVectorStore, node_embeddings: List[BaseNode], use_async: bool
) -> None:
    if use_async:
        await pg.async_add(node_embeddings, index_id="42")
    else:
        pg.add(node_embeddings, index_id="42")

    assert isinstance(pg, PGVectorStore)
    assert hasattr(pg, "_engine")

    q = VectorStoreQuery(query_embedding=_get_sample_vector(0.5), similarity_top_k=10)

    # test deleting nothing
    if use_async:
        await pg.adelete_nodes(index_id="42")
    else:
        pg.delete_nodes(index_id="42")
    if use_async:
        res = await pg.aquery(q, index_id="42")
    else:
        res = pg.query(q, index_id="42")
    assert all(i in res.ids for i in ["aaa", "bbb", "ccc"])

    # test deleting element that doesn't exist
    if use_async:
        await pg.adelete_nodes(["asdf"], index_id="42")
    else:
        pg.delete_nodes(["asdf"], index_id="42")
    if use_async:
        res = await pg.aquery(q, index_id="42")
    else:
        res = pg.query(q, index_id="42")
    assert all(i in res.ids for i in ["aaa", "bbb", "ccc"])

    # test deleting list
    if use_async:
        await pg.adelete_nodes(["aaa", "bbb"], index_id="42")
    else:
        pg.delete_nodes(["aaa", "bbb"], index_id="42")
    if use_async:
        res = await pg.aquery(q, index_id="42")
    else:
        res = pg.query(q, index_id="42")
    assert all(i not in res.ids for i in ["aaa", "bbb"])
    assert "ccc" in res.ids


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [True, False])
async def test_delete_nodes_metadata(
    pg: PGVectorStore, node_embeddings: List[BaseNode], use_async: bool
) -> None:
    if use_async:
        await pg.async_add(node_embeddings, index_id="42")
    else:
        pg.add(node_embeddings, index_id="42")

    assert isinstance(pg, PGVectorStore)
    assert hasattr(pg, "_engine")

    q = VectorStoreQuery(query_embedding=_get_sample_vector(0.5), similarity_top_k=10)

    # test deleting multiple IDs but only one satisfies filter
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_key",
                value=["test_value", "another_value"],
                operator=FilterOperator.IN,
            )
        ]
    )
    if use_async:
        await pg.adelete_nodes(["aaa", "bbb"], filters=filters, index_id="42")
    else:
        pg.delete_nodes(["aaa", "bbb"], filters=filters, index_id="42")
    if use_async:
        res = await pg.aquery(q, index_id="42")
    else:
        res = pg.query(q, index_id="42")
    assert all(i in res.ids for i in ["aaa", "ccc", "ddd"])
    assert "bbb" not in res.ids

    # test deleting one ID which satisfies the filter
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_num",
                value=1,
                operator=FilterOperator.EQ,
            )
        ]
    )
    if use_async:
        await pg.adelete_nodes(["aaa"], filters=filters, index_id="42")
    else:
        pg.delete_nodes(["aaa"], filters=filters, index_id="42")
    if use_async:
        res = await pg.aquery(q, index_id="42")
    else:
        res = pg.query(q, index_id="42")
    assert all(i not in res.ids for i in ["bbb", "aaa"])
    assert all(i in res.ids for i in ["ccc", "ddd"])

    # test deleting one ID which doesn't satisfy the filter
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_num",
                value=1,
                operator=FilterOperator.EQ,
            )
        ]
    )
    if use_async:
        await pg.adelete_nodes(["ccc"], filters=filters, index_id="42")
    else:
        pg.delete_nodes(["ccc"], filters=filters, index_id="42")
    if use_async:
        res = await pg.aquery(q, index_id="42")
    else:
        res = pg.query(q, index_id="42")
    assert all(i not in res.ids for i in ["bbb", "aaa"])
    assert all(i in res.ids for i in ["ccc", "ddd"])

    # test deleting purely based on filters
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_key_2",
                value="test_val_2",
                operator=FilterOperator.EQ,
            )
        ]
    )
    if use_async:
        await pg.adelete_nodes(filters=filters, index_id="42")
    else:
        pg.delete_nodes(filters=filters, index_id="42")
    if use_async:
        res = await pg.aquery(q, index_id="42")
    else:
        res = pg.query(q, index_id="42")
    assert all(i not in res.ids for i in ["bbb", "aaa", "ddd"])
    assert "ccc" in res.ids


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [True, False])
async def test_hnsw_index_creation(
    pg_hnsw_multiple: List[PGVectorStore],
    node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    """
    This test will make sure that creating multiple PGVectorStores handles db initialization properly.
    """
    # calling add will make the db initialization run
    for pg in pg_hnsw_multiple:
        if use_async:
            await pg.async_add(node_embeddings, index_id="42")
        else:
            pg.add(node_embeddings, index_id="42")

    # these are the actual table and index names that PGVectorStore automatically created
    data_test_table_name = f"data_{TEST_TABLE_NAME}"
    data_test_index_name = f"data_{TEST_TABLE_NAME}_embedding_idx"

    # create a connection to the TEST_DB_HNSW database to make sure that one, and only one, index was created
    with psycopg2.connect(**PARAMS, database=TEST_DB_HNSW) as hnsw_conn:
        with hnsw_conn.cursor() as c:
            c.execute(
                f"SELECT COUNT(*) FROM pg_indexes WHERE schemaname = '{TEST_SCHEMA_NAME}' AND tablename = '{data_test_table_name}' AND indexname LIKE '{data_test_index_name}%';"
            )
            index_count = c.fetchone()[0]

    assert (
        index_count == 1
    ), f"Expected exactly one '{data_test_index_name}' index, but found {index_count}."


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [True, False])
async def test_clear(
    pg: PGVectorStore, node_embeddings: List[BaseNode], use_async: bool
) -> None:
    if use_async:
        await pg.async_add(node_embeddings, index_id="42")
    else:
        pg.add(node_embeddings, index_id="42")

    assert isinstance(pg, PGVectorStore)
    assert hasattr(pg, "_engine")

    q = VectorStoreQuery(query_embedding=_get_sample_vector(0.5), similarity_top_k=10)

    if use_async:
        res = await pg.aquery(q, index_id="42")
    else:
        res = pg.query(q, index_id="42")
    assert all(i in res.ids for i in ["bbb", "aaa", "ddd", "ccc"])

    if use_async:
        await pg.aclear(index_id="42")
    else:
        pg.clear(index_id="42")

    if use_async:
        res = await pg.aquery(q, index_id="42")
    else:
        res = pg.query(q, index_id="42")
    assert all(i not in res.ids for i in ["bbb", "aaa", "ddd", "ccc"])
    assert len(res.ids) == 0


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.parametrize(
    ("node_ids", "filters", "expected_node_ids"),
    [
        (["aaa", "bbb"], None, ["aaa", "bbb"]),
        (
            None,
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="test_num",
                        value=1,
                        operator=FilterOperator.EQ,
                    )
                ]
            ),
            ["aaa"],
        ),
        (
            ["bbb", "ccc"],
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="test_key",
                        value="test_value",
                        operator=FilterOperator.EQ,
                    )
                ]
            ),
            ["bbb"],
        ),
        (
            ["ccc"],
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="test_key",
                        value="test_value",
                        operator=FilterOperator.EQ,
                    )
                ]
            ),
            [],
        ),
        (
            ["aaa", "bbb"],
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="test_num",
                        value=999,
                        operator=FilterOperator.EQ,
                    )
                ]
            ),
            [],
        ),
    ],
)
def test_get_nodes_parametrized(
    pg: PGVectorStore,
    node_embeddings: List[TextNode],
    node_ids: Optional[List[str]],
    filters: Optional[MetadataFilters],
    expected_node_ids: List[str],
) -> None:
    """Test get_nodes method with various combinations of node_ids and filters."""
    pg.add(node_embeddings, index_id="42")
    nodes = pg.get_nodes(node_ids=node_ids, filters=filters, index_id="42")
    retrieved_ids = [node.node_id for node in nodes]
    assert set(retrieved_ids) == set(expected_node_ids)
    assert len(retrieved_ids) == len(expected_node_ids)
