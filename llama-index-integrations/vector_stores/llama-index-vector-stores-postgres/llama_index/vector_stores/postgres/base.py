import logging
import re
from typing import Any, Dict, List, NamedTuple, Optional, Type, Union, TYPE_CHECKING

import asyncpg  # noqa
import pgvector  # noqa
from pgvector.sqlalchemy import HALFVEC
import psycopg2  # noqa
import sqlalchemy
import sqlalchemy.ext.asyncio
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterOperator,
    MetadataFilters,
    MetadataFilter,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

if TYPE_CHECKING:
    from sqlalchemy.sql.selectable import Select


class DBEmbeddingRow(NamedTuple):
    node_id: str  # FIXME: verify this type hint
    text: str
    metadata: dict
    similarity: float


_logger = logging.getLogger(__name__)


def get_data_model(
    base: Type,
    index_name: str,
    schema_name: str,
    hybrid_search: bool,
    text_search_config: str,
    cache_okay: bool,
    embed_dim: int = 1536,
    use_jsonb: bool = False,
    use_halfvec: bool = False,
) -> Any:
    """
    This part create a dynamic sqlalchemy model with a new table.
    """
    from pgvector.sqlalchemy import Vector
    from sqlalchemy import Column, Computed
    from sqlalchemy.dialects.postgresql import BIGINT, JSON, JSONB, TSVECTOR, VARCHAR
    from sqlalchemy.schema import Index
    from sqlalchemy.types import TypeDecorator

    class TSVector(TypeDecorator):
        impl = TSVECTOR
        cache_ok = cache_okay

    tablename = "data_%s" % index_name  # dynamic table name
    class_name = "Data%s" % index_name  # dynamic class name
    indexname = "%s_idx" % index_name  # dynamic class name

    metadata_dtype = JSONB if use_jsonb else JSON

    if use_halfvec:
        embedding_col = Column(HALFVEC(embed_dim))  # type: ignore
    else:
        embedding_col = Column(Vector(embed_dim))  # type: ignore

    if hybrid_search:

        class HybridAbstractData(base):  # type: ignore
            __abstract__ = True  # this line is necessary
            id = Column(BIGINT, primary_key=True, autoincrement=True)
            text = Column(VARCHAR, nullable=False)
            metadata_ = Column(metadata_dtype)
            node_id = Column(VARCHAR)
            embedding = embedding_col
            text_search_tsv = Column(  # type: ignore
                TSVector(),
                Computed(
                    "to_tsvector('%s', text)" % text_search_config, persisted=True
                ),
            )

        model = type(
            class_name,
            (HybridAbstractData,),
            {"__tablename__": tablename, "__table_args__": {"schema": schema_name}},
        )

        Index(
            indexname,
            model.text_search_tsv,  # type: ignore
            postgresql_using="gin",
        )
    else:

        class AbstractData(base):  # type: ignore
            __abstract__ = True  # this line is necessary
            id = Column(BIGINT, primary_key=True, autoincrement=True)
            text = Column(VARCHAR, nullable=False)
            metadata_ = Column(metadata_dtype)
            node_id = Column(VARCHAR)
            embedding = embedding_col

        model = type(
            class_name,
            (AbstractData,),
            {"__tablename__": tablename, "__table_args__": {"schema": schema_name}},
        )

    return model


class PGVectorStore(BasePydanticVectorStore):
    """Postgres Vector Store.

    Examples:
        `pip install llama-index-vector-stores-postgres`

        ```python
        from llama_index.vector_stores.postgres import PGVectorStore

        # Create PGVectorStore instance
        vector_store = PGVectorStore.from_params(
            database="vector_db",
            host="localhost",
            password="password",
            port=5432,
            user="postgres",
            table_name="paul_graham_essay",
            embed_dim=1536  # openai embedding dimension
            use_halfvec=True  # Enable half precision

        )
        ```
    """

    stores_text: bool = True
    flat_metadata: bool = False

    connection_string: str
    async_connection_string: str
    table_name: str
    schema_name: str
    embed_dim: int
    hybrid_search: bool
    text_search_config: str
    cache_ok: bool
    perform_setup: bool
    debug: bool
    use_jsonb: bool
    create_engine_kwargs: Dict
    initialization_fail_on_error: bool = False

    hnsw_kwargs: Optional[Dict[str, Any]]

    use_halfvec: bool = False

    _base: Any = PrivateAttr()
    _table_class: Any = PrivateAttr()
    _engine: Any = PrivateAttr()
    _session: Any = PrivateAttr()
    _async_engine: Any = PrivateAttr()
    _async_session: Any = PrivateAttr()
    _is_initialized: bool = PrivateAttr(default=False)

    def __init__(
        self,
        connection_string: Union[str, sqlalchemy.engine.URL],
        async_connection_string: Union[str, sqlalchemy.engine.URL],
        table_name: str,
        schema_name: str,
        hybrid_search: bool = False,
        text_search_config: str = "english",
        embed_dim: int = 1536,
        cache_ok: bool = False,
        perform_setup: bool = True,
        debug: bool = False,
        use_jsonb: bool = False,
        hnsw_kwargs: Optional[Dict[str, Any]] = None,
        create_engine_kwargs: Optional[Dict[str, Any]] = None,
        initialization_fail_on_error: bool = False,
        use_halfvec: bool = False,
    ) -> None:
        """Constructor.

        Args:
            connection_string (Union[str, sqlalchemy.engine.URL]): Connection string to postgres db.
            async_connection_string (Union[str, sqlalchemy.engine.URL]): Connection string to async pg db.
            table_name (str): Table name.
            schema_name (str): Schema name.
            hybrid_search (bool, optional): Enable hybrid search. Defaults to False.
            text_search_config (str, optional): Text search configuration. Defaults to "english".
            embed_dim (int, optional): Embedding dimensions. Defaults to 1536.
            cache_ok (bool, optional): Enable cache. Defaults to False.
            perform_setup (bool, optional): If db should be set up. Defaults to True.
            debug (bool, optional): Debug mode. Defaults to False.
            use_jsonb (bool, optional): Use JSONB instead of JSON. Defaults to False.
            hnsw_kwargs (Optional[Dict[str, Any]], optional): HNSW kwargs, a dict that
                contains "hnsw_ef_construction", "hnsw_ef_search", "hnsw_m", and optionally "hnsw_dist_method". Defaults to None,
                which turns off HNSW search.
            create_engine_kwargs (Optional[Dict[str, Any]], optional): Engine parameters to pass to create_engine. Defaults to None.
            use_halfvec (bool, optional): If `True`, use half-precision vectors. Defaults to False.
        """
        table_name = table_name.lower()
        schema_name = schema_name.lower()

        if hybrid_search and text_search_config is None:
            raise ValueError(
                "Sparse vector index creation requires "
                "a text search configuration specification."
            )

        from sqlalchemy.orm import declarative_base

        super().__init__(
            connection_string=str(connection_string),
            async_connection_string=str(async_connection_string),
            table_name=table_name,
            schema_name=schema_name,
            hybrid_search=hybrid_search,
            text_search_config=text_search_config,
            embed_dim=embed_dim,
            cache_ok=cache_ok,
            perform_setup=perform_setup,
            debug=debug,
            use_jsonb=use_jsonb,
            hnsw_kwargs=hnsw_kwargs,
            create_engine_kwargs=create_engine_kwargs or {},
            initialization_fail_on_error=initialization_fail_on_error,
            use_halfvec=use_halfvec,
        )

        # sqlalchemy model
        self._base = declarative_base()
        self._table_class = get_data_model(
            self._base,
            table_name,
            schema_name,
            hybrid_search,
            text_search_config,
            cache_ok,
            embed_dim=embed_dim,
            use_jsonb=use_jsonb,
            use_halfvec=use_halfvec,
        )

    async def close(self) -> None:
        if not self._is_initialized:
            return

        self._session.close_all()
        self._engine.dispose()

        await self._async_engine.dispose()

    @classmethod
    def class_name(cls) -> str:
        return "PGVectorStore"

    @classmethod
    def from_params(
        cls,
        host: Optional[str] = None,
        port: Optional[str] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        table_name: str = "llamaindex",
        schema_name: str = "public",
        connection_string: Optional[Union[str, sqlalchemy.engine.URL]] = None,
        async_connection_string: Optional[Union[str, sqlalchemy.engine.URL]] = None,
        hybrid_search: bool = False,
        text_search_config: str = "english",
        embed_dim: int = 1536,
        cache_ok: bool = False,
        perform_setup: bool = True,
        debug: bool = False,
        use_jsonb: bool = False,
        hnsw_kwargs: Optional[Dict[str, Any]] = None,
        create_engine_kwargs: Optional[Dict[str, Any]] = None,
        use_halfvec: bool = False,
    ) -> "PGVectorStore":
        """Construct from params.

        Args:
            host (Optional[str], optional): Host of postgres connection. Defaults to None.
            port (Optional[str], optional): Port of postgres connection. Defaults to None.
            database (Optional[str], optional): Postgres DB name. Defaults to None.
            user (Optional[str], optional): Postgres username. Defaults to None.
            password (Optional[str], optional): Postgres password. Defaults to None.
            table_name (str): Table name. Defaults to "llamaindex".
            schema_name (str): Schema name. Defaults to "public".
            connection_string (Union[str, sqlalchemy.engine.URL]): Connection string to postgres db
            async_connection_string (Union[str, sqlalchemy.engine.URL]): Connection string to async pg db
            hybrid_search (bool, optional): Enable hybrid search. Defaults to False.
            text_search_config (str, optional): Text search configuration. Defaults to "english".
            embed_dim (int, optional): Embedding dimensions. Defaults to 1536.
            cache_ok (bool, optional): Enable cache. Defaults to False.
            perform_setup (bool, optional): If db should be set up. Defaults to True.
            debug (bool, optional): Debug mode. Defaults to False.
            use_jsonb (bool, optional): Use JSONB instead of JSON. Defaults to False.
            hnsw_kwargs (Optional[Dict[str, Any]], optional): HNSW kwargs, a dict that
                contains "hnsw_ef_construction", "hnsw_ef_search", "hnsw_m", and optionally "hnsw_dist_method". Defaults to None,
                which turns off HNSW search.
            create_engine_kwargs (Optional[Dict[str, Any]], optional): Engine parameters to pass to create_engine. Defaults to None.
            use_halfvec (bool, optional): If `True`, use half-precision vectors. Defaults to False.

        Returns:
            PGVectorStore: Instance of PGVectorStore constructed from params.
        """
        conn_str = (
            connection_string
            or f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        )
        async_conn_str = async_connection_string or (
            f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
        )
        return cls(
            connection_string=conn_str,
            async_connection_string=async_conn_str,
            table_name=table_name,
            schema_name=schema_name,
            hybrid_search=hybrid_search,
            text_search_config=text_search_config,
            embed_dim=embed_dim,
            cache_ok=cache_ok,
            perform_setup=perform_setup,
            debug=debug,
            use_jsonb=use_jsonb,
            hnsw_kwargs=hnsw_kwargs,
            create_engine_kwargs=create_engine_kwargs,
            use_halfvec=use_halfvec,
        )

    @property
    def client(self) -> Any:
        if not self._is_initialized:
            return None
        return self._engine

    def _connect(self) -> Any:
        from sqlalchemy import create_engine
        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
        from sqlalchemy.orm import sessionmaker

        self._engine = create_engine(
            self.connection_string, echo=self.debug, **self.create_engine_kwargs
        )
        self._session = sessionmaker(self._engine)

        self._async_engine = create_async_engine(
            self.async_connection_string, **self.create_engine_kwargs
        )
        self._async_session = sessionmaker(self._async_engine, class_=AsyncSession)  # type: ignore

    def _create_schema_if_not_exists(self) -> bool:
        """
        Create the schema if it does not exist.
        Returns True if the schema was created, False if it already existed.
        """
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", self.schema_name):
            raise ValueError(f"Invalid schema_name: {self.schema_name}")
        with self._session() as session, session.begin():
            # Check if the specified schema exists with "CREATE" statement
            check_schema_statement = sqlalchemy.text(
                f"SELECT schema_name FROM information_schema.schemata WHERE schema_name = :schema_name"
            ).bindparams(schema_name=self.schema_name)
            result = session.execute(check_schema_statement).fetchone()

            # If the schema does not exist, then create it
            schema_doesnt_exist = result is None
            if schema_doesnt_exist:
                create_schema_statement = sqlalchemy.text(
                    # DDL won't tolerate quoted string literal here for schema_name,
                    # so use a format string to embed the schema_name directly, instead of a param.
                    f"CREATE SCHEMA IF NOT EXISTS {self.schema_name}"
                )
                session.execute(create_schema_statement)

            session.commit()
            return schema_doesnt_exist

    def _create_tables_if_not_exists(self) -> None:
        with self._session() as session, session.begin():
            self._base.metadata.create_all(session.connection())

    def _create_extension(self) -> None:
        import sqlalchemy

        with self._session() as session, session.begin():
            statement = sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector")
            session.execute(statement)
            session.commit()

    def _create_hnsw_index(self) -> None:
        import sqlalchemy

        if (
            "hnsw_ef_construction" not in self.hnsw_kwargs
            or "hnsw_m" not in self.hnsw_kwargs
        ):
            raise ValueError(
                "Make sure hnsw_ef_search, hnsw_ef_construction, and hnsw_m are in hnsw_kwargs."
            )

        hnsw_ef_construction = self.hnsw_kwargs.pop("hnsw_ef_construction")
        hnsw_m = self.hnsw_kwargs.pop("hnsw_m")

        # If user didn’t specify an operator, pick a default based on whether halfvec is used
        if "hnsw_dist_method" in self.hnsw_kwargs:
            hnsw_dist_method = self.hnsw_kwargs.pop("hnsw_dist_method")
        else:
            if self.use_halfvec:
                hnsw_dist_method = "halfvec_l2_ops"
            else:
                # Default to vector_cosine_ops
                hnsw_dist_method = "vector_cosine_ops"

        index_name = f"{self._table_class.__tablename__}_embedding_idx"

        with self._session() as session, session.begin():
            statement = sqlalchemy.text(
                f"CREATE INDEX IF NOT EXISTS {index_name} "
                f"ON {self.schema_name}.{self._table_class.__tablename__} "
                f"USING hnsw (embedding {hnsw_dist_method}) "
                f"WITH (m = {hnsw_m}, ef_construction = {hnsw_ef_construction})"
            )
            session.execute(statement)
            session.commit()

    def _initialize(self) -> None:
        fail_on_error = self.initialization_fail_on_error
        if not self._is_initialized:
            self._connect()
            if self.perform_setup:
                try:
                    self._create_schema_if_not_exists()
                except Exception as e:
                    _logger.warning(f"PG Setup: Error creating schema: {e}")
                    if fail_on_error:
                        raise
                try:
                    self._create_extension()
                except Exception as e:
                    _logger.warning(f"PG Setup: Error creating extension: {e}")
                    if fail_on_error:
                        raise
                try:
                    self._create_tables_if_not_exists()
                except Exception as e:
                    _logger.warning(f"PG Setup: Error creating tables: {e}")
                    if fail_on_error:
                        raise
                if self.hnsw_kwargs is not None:
                    try:
                        self._create_hnsw_index()
                    except Exception as e:
                        _logger.warning(f"PG Setup: Error creating HNSW index: {e}")
                        if fail_on_error:
                            raise
            self._is_initialized = True

    def _node_to_table_row(self, node: BaseNode) -> Any:
        return self._table_class(
            node_id=node.node_id,
            embedding=node.get_embedding(),
            text=node.get_content(metadata_mode=MetadataMode.NONE),
            metadata_=node_to_metadata_dict(
                node,
                remove_text=True,
                flat_metadata=self.flat_metadata,
            ),
        )

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        self._initialize()
        ids = []
        with self._session() as session, session.begin():
            for node in nodes:
                ids.append(node.node_id)
                item = self._node_to_table_row(node)
                session.add(item)
            session.commit()
        return ids

    async def async_add(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        self._initialize()
        ids = []
        async with self._async_session() as session, session.begin():
            for node in nodes:
                ids.append(node.node_id)
                item = self._node_to_table_row(node)
                session.add(item)
            await session.commit()
        return ids

    def _to_postgres_operator(self, operator: FilterOperator) -> str:
        if operator == FilterOperator.EQ:
            return "="
        elif operator == FilterOperator.GT:
            return ">"
        elif operator == FilterOperator.LT:
            return "<"
        elif operator == FilterOperator.NE:
            return "!="
        elif operator == FilterOperator.GTE:
            return ">="
        elif operator == FilterOperator.LTE:
            return "<="
        elif operator == FilterOperator.IN:
            return "IN"
        elif operator == FilterOperator.NIN:
            return "NOT IN"
        elif operator == FilterOperator.CONTAINS:
            return "@>"
        elif operator == FilterOperator.TEXT_MATCH:
            return "LIKE"
        elif operator == FilterOperator.TEXT_MATCH_INSENSITIVE:
            return "ILIKE"
        else:
            _logger.warning(f"Unknown operator: {operator}, fallback to '='")
            return "="

    def _build_filter_clause(self, filter_: MetadataFilter) -> Any:
        from sqlalchemy import text

        if filter_.operator in [FilterOperator.IN, FilterOperator.NIN]:
            # Expects a single value in the metadata, and a list to compare

            # In Python, to create a tuple with a single element, you need to include a comma after the element
            # This code will correctly format the IN clause whether there is one element or multiple elements in the list:
            filter_value = ", ".join(f"'{e}'" for e in filter_.value)

            return text(
                f"metadata_->>'{filter_.key}' "
                f"{self._to_postgres_operator(filter_.operator)} "
                f"({filter_value})"
            )
        elif filter_.operator == FilterOperator.CONTAINS:
            # Expects a list stored in the metadata, and a single value to compare
            return text(
                f"metadata_::jsonb->'{filter_.key}' "
                f"{self._to_postgres_operator(filter_.operator)} "
                f"'[\"{filter_.value}\"]'"
            )
        elif (
            filter_.operator == FilterOperator.TEXT_MATCH
            or filter_.operator == FilterOperator.TEXT_MATCH_INSENSITIVE
        ):
            # Where the operator is text_match or ilike, we need to wrap the filter in '%' characters
            return text(
                f"metadata_->>'{filter_.key}' "
                f"{self._to_postgres_operator(filter_.operator)} "
                f"'%{filter_.value}%'"
            )
        else:
            # Check if value is a number. If so, cast the metadata value to a float
            # This is necessary because the metadata is stored as a string
            try:
                return text(
                    f"(metadata_->>'{filter_.key}')::float "
                    f"{self._to_postgres_operator(filter_.operator)} "
                    f"{float(filter_.value)}"
                )
            except ValueError:
                # If not a number, then treat it as a string
                return text(
                    f"metadata_->>'{filter_.key}' "
                    f"{self._to_postgres_operator(filter_.operator)} "
                    f"'{filter_.value}'"
                )

    def _recursively_apply_filters(self, filters: List[MetadataFilters]) -> Any:
        """
        Returns a sqlalchemy where clause.
        """
        import sqlalchemy

        sqlalchemy_conditions = {
            "or": sqlalchemy.sql.or_,
            "and": sqlalchemy.sql.and_,
        }

        if filters.condition not in sqlalchemy_conditions:
            raise ValueError(
                f"Invalid condition: {filters.condition}. "
                f"Must be one of {list(sqlalchemy_conditions.keys())}"
            )

        return sqlalchemy_conditions[filters.condition](
            *(
                (
                    self._build_filter_clause(filter_)
                    if not isinstance(filter_, MetadataFilters)
                    else self._recursively_apply_filters(filter_)
                )
                for filter_ in filters.filters
            )
        )

    def _apply_filters_and_limit(
        self,
        stmt: "Select",
        limit: int,
        metadata_filters: Optional[MetadataFilters] = None,
    ) -> Any:
        if metadata_filters:
            stmt = stmt.where(  # type: ignore
                self._recursively_apply_filters(metadata_filters)
            )
        return stmt.limit(limit)  # type: ignore

    def _build_query(
        self,
        embedding: Optional[List[float]],
        limit: int = 10,
        metadata_filters: Optional[MetadataFilters] = None,
    ) -> Any:
        from sqlalchemy import select, text

        stmt = select(  # type: ignore
            self._table_class.id,
            self._table_class.node_id,
            self._table_class.text,
            self._table_class.metadata_,
            self._table_class.embedding.cosine_distance(embedding).label("distance"),
        ).order_by(text("distance asc"))

        return self._apply_filters_and_limit(stmt, limit, metadata_filters)

    def _query_with_score(
        self,
        embedding: Optional[List[float]],
        limit: int = 10,
        metadata_filters: Optional[MetadataFilters] = None,
        **kwargs: Any,
    ) -> List[DBEmbeddingRow]:
        stmt = self._build_query(embedding, limit, metadata_filters)
        with self._session() as session, session.begin():
            from sqlalchemy import text

            if kwargs.get("ivfflat_probes"):
                ivfflat_probes = kwargs.get("ivfflat_probes")
                session.execute(
                    text(f"SET ivfflat.probes = :ivfflat_probes"),
                    {"ivfflat_probes": ivfflat_probes},
                )
            if self.hnsw_kwargs:
                hnsw_ef_search = (
                    kwargs.get("hnsw_ef_search") or self.hnsw_kwargs["hnsw_ef_search"]
                )
                session.execute(
                    text(f"SET hnsw.ef_search = :hnsw_ef_search"),
                    {"hnsw_ef_search": hnsw_ef_search},
                )

            res = session.execute(
                stmt,
            )
            return [
                DBEmbeddingRow(
                    node_id=item.node_id,
                    text=item.text,
                    metadata=item.metadata_,
                    similarity=(1 - item.distance) if item.distance is not None else 0,
                )
                for item in res.all()
            ]

    async def _aquery_with_score(
        self,
        embedding: Optional[List[float]],
        limit: int = 10,
        metadata_filters: Optional[MetadataFilters] = None,
        **kwargs: Any,
    ) -> List[DBEmbeddingRow]:
        stmt = self._build_query(embedding, limit, metadata_filters)
        async with self._async_session() as async_session, async_session.begin():
            from sqlalchemy import text

            if self.hnsw_kwargs:
                hnsw_ef_search = (
                    kwargs.get("hnsw_ef_search") or self.hnsw_kwargs["hnsw_ef_search"]
                )
                await async_session.execute(
                    text(f"SET hnsw.ef_search = {hnsw_ef_search}"),
                )
            if kwargs.get("ivfflat_probes"):
                ivfflat_probes = kwargs.get("ivfflat_probes")
                await async_session.execute(
                    text(f"SET ivfflat.probes = :ivfflat_probes"),
                    {"ivfflat_probes": ivfflat_probes},
                )

            res = await async_session.execute(stmt)
            return [
                DBEmbeddingRow(
                    node_id=item.node_id,
                    text=item.text,
                    metadata=item.metadata_,
                    similarity=(1 - item.distance) if item.distance is not None else 0,
                )
                for item in res.all()
            ]

    def _build_filters_query(
        self,
        limit: int,
        metadata_filters: Optional[MetadataFilters] = None,
    ) -> Any:
        from sqlalchemy import select
        from sqlalchemy.sql import text

        stmt = (
            select(  # type: ignore
                self._table_class.id,
                self._table_class.node_id,
                self._table_class.text,
                self._table_class.metadata_,
                self._table_class.text_search_tsv.label("rank"),
            )
            .order_by(text("rank desc"))
        )

        # type: ignore
        return self._apply_filters_and_limit(stmt, limit, metadata_filters)

    def _build_sparse_query(
        self,
        query_str: Optional[str],
        limit: int,
        metadata_filters: Optional[MetadataFilters] = None,
    ) -> Any:
        from sqlalchemy import select, type_coerce
        from sqlalchemy.sql import func, text
        from sqlalchemy.types import UserDefinedType

        class REGCONFIG(UserDefinedType):
            # The TypeDecorator.cache_ok class-level flag indicates if this custom TypeDecorator is safe to be used as part of a cache key.
            # If the TypeDecorator is not guaranteed to produce the same bind/result behavior and SQL generation every time,
            # this flag should be set to False; otherwise if the class produces the same behavior each time, it may be set to True.
            cache_ok = True

            def get_col_spec(self, **kw: Any) -> str:
                return "regconfig"

        if query_str is None:
            raise ValueError("query_str must be specified for a sparse vector query.")

        # Replace '&' with '|' to perform an OR search for higher recall
        ts_query = func.to_tsquery(
            func.replace(
                func.text(
                    func.plainto_tsquery(
                        type_coerce(self.text_search_config, REGCONFIG), query_str
                    )
                ),
                "&",
                "|",
            )
        )
        stmt = (
            select(  # type: ignore
                self._table_class.id,
                self._table_class.node_id,
                self._table_class.text,
                self._table_class.metadata_,
                func.ts_rank(self._table_class.text_search_tsv, ts_query).label("rank"),
            )
            .where(self._table_class.text_search_tsv.op("@@")(ts_query))
            .order_by(text("rank desc"))
        )

        # type: ignore
        return self._apply_filters_and_limit(stmt, limit, metadata_filters)

    async def _async_sparse_query_with_rank(
        self,
        query_str: Optional[str] = None,
        limit: int = 10,
        metadata_filters: Optional[MetadataFilters] = None,
    ) -> List[DBEmbeddingRow]:
        stmt = self._build_sparse_query(query_str, limit, metadata_filters)
        async with self._async_session() as async_session, async_session.begin():
            res = await async_session.execute(stmt)
            return [
                DBEmbeddingRow(
                    node_id=item.node_id,
                    text=item.text,
                    metadata=item.metadata_,
                    similarity=item.rank,
                )
                for item in res.all()
            ]

    def _sparse_query_with_rank(
        self,
        query_str: Optional[str] = None,
        limit: int = 10,
        metadata_filters: Optional[MetadataFilters] = None,
    ) -> List[DBEmbeddingRow]:
        stmt = self._build_sparse_query(query_str, limit, metadata_filters)
        with self._session() as session, session.begin():
            res = session.execute(stmt)
            return [
                DBEmbeddingRow(
                    node_id=item.node_id,
                    text=item.text,
                    metadata=item.metadata_,
                    similarity=item.rank,
                )
                for item in res.all()
            ]

    def _query_filters_only(
        self,
        limit: int = 10,
        metadata_filters: Optional[MetadataFilters] = None,
    ) -> List[DBEmbeddingRow]:
        stmt = self._build_filters_query(limit, metadata_filters)
        with self._session() as session, session.begin():
            res = session.execute(stmt)
            return [
                DBEmbeddingRow(
                    node_id=item.node_id,
                    text=item.text,
                    metadata=item.metadata_,
                    similarity=item.rank,
                )
                for item in res.all()
            ]

    async def _async_hybrid_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> List[DBEmbeddingRow]:
        import asyncio

        if query.alpha is not None:
            _logger.warning("postgres hybrid search does not support alpha parameter.")

        sparse_top_k = query.sparse_top_k or query.similarity_top_k

        results = await asyncio.gather(
            self._aquery_with_score(
                query.query_embedding,
                query.similarity_top_k,
                query.filters,
                **kwargs,
            ),
            self._async_sparse_query_with_rank(
                query.query_str, sparse_top_k, query.filters
            ),
        )

        dense_results, sparse_results = results
        all_results = dense_results + sparse_results
        return _dedup_results(all_results)

    def _hybrid_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> List[DBEmbeddingRow]:
        if query.alpha is not None:
            _logger.warning("postgres hybrid search does not support alpha parameter.")

        sparse_top_k = query.sparse_top_k or query.similarity_top_k

        dense_results = self._query_with_score(
            query.query_embedding,
            query.similarity_top_k,
            query.filters,
            **kwargs,
        )

        sparse_results = self._sparse_query_with_rank(
            query.query_str, sparse_top_k, query.filters
        )

        all_results = dense_results + sparse_results
        return _dedup_results(all_results)

    def _db_rows_to_query_result(
        self, rows: List[DBEmbeddingRow]
    ) -> VectorStoreQueryResult:
        nodes = []
        similarities = []
        ids = []
        for db_embedding_row in rows:
            try:
                node = metadata_dict_to_node(db_embedding_row.metadata)
                node.set_content(str(db_embedding_row.text))
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                node = TextNode(
                    id_=db_embedding_row.node_id,
                    text=db_embedding_row.text,
                    metadata=db_embedding_row.metadata,
                )
            similarities.append(db_embedding_row.similarity)
            ids.append(db_embedding_row.node_id)
            nodes.append(node)

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        self._initialize()
        if query.mode == VectorStoreQueryMode.HYBRID:
            results = await self._async_hybrid_query(query, **kwargs)
        elif query.mode == VectorStoreQueryMode.FILTERS:
            filters_top_k = query.filters_top_k or query.similarity_top_k
            results = self._query_filters_only(filters_top_k, query.filters)
        elif query.mode in [
            VectorStoreQueryMode.SPARSE,
            VectorStoreQueryMode.TEXT_SEARCH,
        ]:
            sparse_top_k = query.sparse_top_k or query.similarity_top_k
            results = await self._async_sparse_query_with_rank(
                query.query_str, sparse_top_k, query.filters
            )
        elif query.mode == VectorStoreQueryMode.DEFAULT:
            results = await self._aquery_with_score(
                query.query_embedding,
                query.similarity_top_k,
                query.filters,
                **kwargs,
            )
        else:
            raise ValueError(f"Invalid query mode: {query.mode}")

        return self._db_rows_to_query_result(results)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        self._initialize()
        if query.mode == VectorStoreQueryMode.HYBRID:
            results = self._hybrid_query(query, **kwargs)
        elif query.mode == VectorStoreQueryMode.FILTERS:
            filters_top_k = query.filters_top_k or query.similarity_top_k
            results = self._query_filters_only(filters_top_k, query.filters)
        elif query.mode in [
            VectorStoreQueryMode.SPARSE,
            VectorStoreQueryMode.TEXT_SEARCH,
        ]:
            sparse_top_k = query.sparse_top_k or query.similarity_top_k
            results = self._sparse_query_with_rank(
                query.query_str, sparse_top_k, query.filters
            )
        elif query.mode == VectorStoreQueryMode.DEFAULT:
            results = self._query_with_score(
                query.query_embedding,
                query.similarity_top_k,
                query.filters,
                **kwargs,
            )
        else:
            raise ValueError(f"Invalid query mode: {query.mode}")

        return self._db_rows_to_query_result(results)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        from sqlalchemy import delete

        self._initialize()
        with self._session() as session, session.begin():
            stmt = delete(self._table_class).where(
                self._table_class.metadata_["doc_id"].astext == ref_doc_id
            )

            session.execute(stmt)
            session.commit()

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Deletes nodes.

        Args:
            node_ids (Optional[List[str]], optional): IDs of nodes to delete. Defaults to None.
            filters (Optional[MetadataFilters], optional): Metadata filters. Defaults to None.
        """
        if not node_ids and not filters:
            return

        from sqlalchemy import delete

        self._initialize()
        with self._session() as session, session.begin():
            stmt = delete(self._table_class)

            if node_ids:
                stmt = stmt.where(self._table_class.node_id.in_(node_ids))

            if filters:
                stmt = stmt.where(self._recursively_apply_filters(filters))

            session.execute(stmt)
            session.commit()

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Deletes nodes asynchronously.

        Args:
            node_ids (Optional[List[str]], optional): IDs of nodes to delete. Defaults to None.
            filters (Optional[MetadataFilters], optional): Metadata filters. Defaults to None.
        """
        if not node_ids and not filters:
            return

        from sqlalchemy import delete

        self._initialize()
        async with self._async_session() as async_session, async_session.begin():
            stmt = delete(self._table_class)

            if node_ids:
                stmt = stmt.where(self._table_class.node_id.in_(node_ids))

            if filters:
                stmt = stmt.where(self._recursively_apply_filters(filters))

            await async_session.execute(stmt)
            await async_session.commit()

    def clear(self) -> None:
        """Clears table."""
        from sqlalchemy import delete

        self._initialize()
        with self._session() as session, session.begin():
            stmt = delete(self._table_class)

            session.execute(stmt)
            session.commit()

    async def aclear(self) -> None:
        """Asynchronously clears table."""
        from sqlalchemy import delete

        self._initialize()
        async with self._async_session() as async_session, async_session.begin():
            stmt = delete(self._table_class)

            await async_session.execute(stmt)
            await async_session.commit()

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Get nodes from vector store."""
        assert (
            node_ids is not None or filters is not None
        ), "Either node_ids or filters must be provided"

        self._initialize()
        from sqlalchemy import select

        stmt = select(
            self._table_class.node_id,
            self._table_class.text,
            self._table_class.metadata_,
            self._table_class.embedding,
        )

        if node_ids:
            stmt = stmt.where(self._table_class.node_id.in_(node_ids))

        if filters:
            filter_clause = self._recursively_apply_filters(filters)
            stmt = stmt.where(filter_clause)

        nodes: List[BaseNode] = []

        with self._session() as session, session.begin():
            res = session.execute(stmt).fetchall()
            for item in res:
                node_id = item.node_id
                text = item.text
                metadata = item.metadata_
                embedding = item.embedding

                try:
                    node = metadata_dict_to_node(metadata)
                    node.set_content(str(text))
                    node.embedding = embedding
                except Exception:
                    node = TextNode(
                        id_=node_id,
                        text=text,
                        metadata=metadata,
                        embedding=embedding,
                    )
                nodes.append(node)

        return nodes


def _dedup_results(results: List[DBEmbeddingRow]) -> List[DBEmbeddingRow]:
    seen_ids = set()
    deduped_results = []
    for result in results:
        if result.node_id not in seen_ids:
            deduped_results.append(result)
            seen_ids.add(result.node_id)
    return deduped_results
