from typing import Any, Dict, List, Optional
import logging

from falkordb import FalkorDB

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
    FilterOperator,
    MetadataFilters,
    FilterCondition,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

_logger = logging.getLogger(__name__)


def clean_params(params: List[BaseNode]) -> List[Dict[str, Any]]:
    clean_params = []
    for record in params:
        text = record.get_content(metadata_mode=MetadataMode.NONE)
        embedding = record.get_embedding()
        id = record.node_id
        metadata = node_to_metadata_dict(record, remove_text=True, flat_metadata=False)
        for k in ["document_id", "doc_id"]:
            if k in metadata:
                del metadata[k]
        clean_params.append(
            {"text": text, "embedding": embedding, "id": id, "metadata": metadata}
        )
    return clean_params


def _to_falkordb_operator(operator: FilterOperator) -> str:
    operator_map = {
        FilterOperator.EQ: "=",
        FilterOperator.GT: ">",
        FilterOperator.LT: "<",
        FilterOperator.NE: "<>",
        FilterOperator.GTE: ">=",
        FilterOperator.LTE: "<=",
        FilterOperator.IN: "IN",
        FilterOperator.NIN: "NOT IN",
        FilterOperator.CONTAINS: "CONTAINS",
    }
    return operator_map.get(operator, "=")


def construct_metadata_filter(filters: MetadataFilters):
    cypher_snippets = []
    params = {}
    for index, filter in enumerate(filters.filters):
        cypher_snippets.append(
            f"n.`{filter.key}` {_to_falkordb_operator(filter.operator)} $param_{index}"
        )
        params[f"param_{index}"] = filter.value

    condition = " OR " if filters.condition == FilterCondition.OR else " AND "
    return condition.join(cypher_snippets), params


class FalkorDBVectorStore(BasePydanticVectorStore):
    stores_text: bool = True
    flat_metadata: bool = True

    distance_strategy: str
    index_name: str
    node_label: str
    embedding_node_property: str
    text_node_property: str
    embedding_dimension: int

    _client: FalkorDB = PrivateAttr()
    _database: str = PrivateAttr()

    def __init__(
        self,
        url: Optional[str] = None,
        database: str = "falkor",
        index_name: str = "vector",
        node_label: str = "Chunk",
        embedding_node_property: str = "embedding",
        text_node_property: str = "text",
        distance_strategy: str = "cosine",
        embedding_dimension: int = 1536,
        driver: Optional[FalkorDB] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            distance_strategy=distance_strategy,
            index_name=index_name,
            node_label=node_label,
            embedding_node_property=embedding_node_property,
            text_node_property=text_node_property,
            embedding_dimension=embedding_dimension,
        )

        if distance_strategy not in ["cosine", "euclidean"]:
            raise ValueError("distance_strategy must be either 'euclidean' or 'cosine'")

        self._client = driver or FalkorDB.from_url(url).select_graph(database)
        self._database = database

        # Inline check_if_not_null function
        for prop, value in zip(
            [
                "index_name",
                "node_label",
                "embedding_node_property",
                "text_node_property",
            ],
            [index_name, node_label, embedding_node_property, text_node_property],
        ):
            if not value:
                raise ValueError(f"Parameter `{prop}` must not be None or empty string")

        if not self.retrieve_existing_index():
            self.create_new_index()

    @property
    def client(self) -> FalkorDB:
        return self._client

    def create_new_index(self) -> None:
        index_query = (
            f"CREATE VECTOR INDEX {self.index_name} "
            f"FOR (n:`{self.node_label}`) "
            f"ON (n.`{self.embedding_node_property}`) "
            f"OPTIONS {{dimension: {self.embedding_dimension}, metric: '{self.distance_strategy}'}}"
        )
        self._client.query(index_query)

    def retrieve_existing_index(self) -> bool:
        index_information = self._client.query(
            "CALL db.indexes() "
            "YIELD label, properties, types, options, entitytype "
            "WHERE types = ['VECTOR'] AND label = $index_name",
            params={"index_name": self.index_name},
        )
        if index_information.result_set:
            index = index_information.result_set[0]
            self.node_label = index["entitytype"]
            self.embedding_node_property = index["properties"][0]
            self.embedding_dimension = index["options"]["dimension"]
            self.distance_strategy = index["options"]["metric"]
            return True
        return False

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        ids = [r.node_id for r in nodes]
        import_query = (
            "UNWIND $data AS row "
            f"MERGE (c:`{self.node_label}` {{id: row.id}}) "
            f"SET c.`{self.embedding_node_property}` = row.embedding, "
            f"c.`{self.text_node_property}` = row.text, "
            "c += row.metadata"
        )

        self._client.query(import_query, params={"data": clean_params(nodes)})
        return ids

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        base_query = (
            f"MATCH (n:`{self.node_label}`) "
            f"WHERE n.`{self.embedding_node_property}` IS NOT NULL "
        )

        if query.filters:
            filter_snippets, filter_params = construct_metadata_filter(query.filters)
            base_query += f"AND {filter_snippets} "
        else:
            filter_params = {}

        similarity_query = (
            f"WITH n, vector.similarity.{self.distance_strategy}("
            f"n.`{self.embedding_node_property}`, $embedding) AS score "
            "ORDER BY score DESC LIMIT toInteger($k) "
        )

        return_query = (
            f"RETURN n.`{self.text_node_property}` AS text, score, "
            "n.id AS id, "
            f"n {{.*, `{self.text_node_property}`: NULL, "
            f"`{self.embedding_node_property}`: NULL, id: NULL}} AS metadata"
        )

        full_query = base_query + similarity_query + return_query

        parameters = {
            "k": query.similarity_top_k,
            "embedding": query.query_embedding,
            **filter_params,
        }

        results = self._client.query(full_query, params=parameters)

        nodes = []
        similarities = []
        ids = []
        for record in results.result_set:
            node = metadata_dict_to_node(record["metadata"])
            node.set_content(str(record["text"]))
            nodes.append(node)
            similarities.append(record["score"])
            ids.append(record["id"])

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        self._client.query(
            f"MATCH (n:`{self.node_label}`) WHERE n.ref_doc_id = $id DETACH DELETE n",
            params={"id": ref_doc_id},
        )
