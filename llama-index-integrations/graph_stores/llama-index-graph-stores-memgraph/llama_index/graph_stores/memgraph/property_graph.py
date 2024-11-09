from typing import Any, List, Dict, Optional, Tuple
from llama_index.core.graph_stores.prompts import DEFAULT_CYPHER_TEMPALTE
from llama_index.core.graph_stores.types import (
    PropertyGraphStore,
    Triplet,
    LabelledNode,
    Relation,
    EntityNode,
    ChunkNode,
)
from llama_index.core.graph_stores.utils import (
    clean_string_values,
    value_sanitize,
    LIST_LIMIT,
)

from llama_index.core.prompts import PromptTemplate
from llama_index.core.vector_stores.types import VectorStoreQuery
import neo4j


def remove_empty_values(input_dict):
    """
    Remove entries with empty values from the dictionary.
    """
    return {key: value for key, value in input_dict.items() if value}


BASE_ENTITY_LABEL = "__Entity__"
BASE_NODE_LABEL = "__Node__"
EXCLUDED_LABELS = ["_Bloom_Perspective_", "_Bloom_Scene_"]
EXCLUDED_RELS = ["_Bloom_HAS_SCENE_"]
EXHAUSTIVE_SEARCH_LIMIT = 10000
# Threshold for returning all available prop values in graph schema
DISTINCT_VALUE_LIMIT = 10
CHUNK_SIZE = 1000
# Threshold for max number of returned triplets
LIMIT = 100

node_properties_query = """
MATCH (n)
UNWIND labels(n) AS label
WITH label, COUNT(n) AS count
CALL schema.node_type_properties()
YIELD propertyName, nodeLabels, propertyTypes
WITH label, nodeLabels, count, collect({property: propertyName, type: propertyTypes[0]}) AS properties
WHERE label IN nodeLabels
RETURN {labels: label, count: count, properties: properties} AS output
ORDER BY count DESC
"""

rel_properties_query = """
CALL schema.rel_type_properties()
YIELD relType AS label, propertyName AS property, propertyTypes AS type
WITH label, collect({property: property, type: type}) AS properties
RETURN {type: label, properties: properties} AS output
"""

rel_query = """
MATCH (start_node)-[r]->(end_node)
WITH DISTINCT labels(start_node) AS start_labels, type(r) AS relationship_type, labels(end_node) AS end_labels, keys(r) AS relationship_properties
UNWIND start_labels AS start_label
UNWIND end_labels AS end_label
RETURN DISTINCT {start: start_label, type: relationship_type, end: end_label} AS output
"""


class MemgraphPropertyGraphStore(PropertyGraphStore):
    r"""
    Memgraph Property Graph Store.

    This class implements a Memgraph property graph store.

    Args:
        username (str): The username for the Memgraph database.
        password (str): The password for the Memgraph database.
        url (str): The URL for the Memgraph database.
        database (Optional[str]): The name of the database to connect to. Defaults to "memgraph".

    Examples:
        ```python
        from llama_index.core.indices.property_graph import PropertyGraphIndex
        from llama_index.graph_stores.memgraph import MemgraphPropertyGraphStore

        # Create a MemgraphPropertyGraphStore instance
        graph_store = MemgraphPropertyGraphStore(
            username="memgraph",
            password="password",
            url="bolt://localhost:7687",
            database="memgraph"
        )

        # Create the index
        index = PropertyGraphIndex.from_documents(
            documents,
            property_graph_store=graph_store,
        )

        # Close the Memgraph connection explicitly.
        graph_store.close()
        ```
    """

    supports_structured_queries: bool = True
    text_to_cypher_template: PromptTemplate = DEFAULT_CYPHER_TEMPALTE

    def __init__(
        self,
        username: str,
        password: str,
        url: str,
        database: Optional[str] = "memgraph",
        refresh_schema: bool = True,
        sanitize_query_output: bool = True,
        enhanced_schema: bool = False,
        **neo4j_kwargs: Any,
    ) -> None:
        self.sanitize_query_output = sanitize_query_output
        self.enhanced_schema = enhanced_schema
        self._driver = neo4j.GraphDatabase.driver(
            url, auth=(username, password), **neo4j_kwargs
        )
        self._database = database
        self.structured_schema = {}
        if refresh_schema:
            self.refresh_schema()

        # Create index for faster imports and retrieval
        self.structured_query(f"""CREATE INDEX ON :{BASE_NODE_LABEL}(id);""")
        self.structured_query(f"""CREATE INDEX ON :{BASE_ENTITY_LABEL}(id);""")

    @property
    def client(self):
        return self._driver

    def close(self) -> None:
        self._driver.close()

    def refresh_schema(self) -> None:
        """Refresh the schema."""
        # Leave schema empty if db is empty
        if self.structured_query("MATCH (n) RETURN n LIMIT 1") == []:
            return

        node_query_results = self.structured_query(
            node_properties_query,
            param_map={
                "EXCLUDED_LABELS": [
                    *EXCLUDED_LABELS,
                    BASE_ENTITY_LABEL,
                    BASE_NODE_LABEL,
                ]
            },
        )
        node_properties = {}
        for el in node_query_results:
            if el["output"]["labels"] in [
                *EXCLUDED_LABELS,
                BASE_ENTITY_LABEL,
                BASE_NODE_LABEL,
            ]:
                continue

            label = el["output"]["labels"]
            properties = el["output"]["properties"]
            if label in node_properties:
                node_properties[label]["properties"].extend(
                    prop
                    for prop in properties
                    if prop not in node_properties[label]["properties"]
                )
            else:
                node_properties[label] = {"properties": properties}

        node_properties = [
            {"labels": label, **value} for label, value in node_properties.items()
        ]
        rels_query_result = self.structured_query(
            rel_properties_query, param_map={"EXCLUDED_LABELS": EXCLUDED_RELS}
        )
        rel_properties = (
            [
                el["output"]
                for el in rels_query_result
                if any(prop["property"] for prop in el["output"].get("properties", []))
            ]
            if rels_query_result
            else []
        )
        rel_objs_query_result = self.structured_query(
            rel_query,
            param_map={
                "EXCLUDED_LABELS": [
                    *EXCLUDED_LABELS,
                    BASE_ENTITY_LABEL,
                    BASE_NODE_LABEL,
                ]
            },
        )
        relationships = [
            el["output"]
            for el in rel_objs_query_result
            if rel_objs_query_result
            and el["output"]["start"]
            not in [*EXCLUDED_LABELS, BASE_ENTITY_LABEL, BASE_NODE_LABEL]
            and el["output"]["end"]
            not in [*EXCLUDED_LABELS, BASE_ENTITY_LABEL, BASE_NODE_LABEL]
        ]
        self.structured_schema = {
            "node_props": {el["labels"]: el["properties"] for el in node_properties},
            "rel_props": {el["type"]: el["properties"] for el in rel_properties},
            "relationships": relationships,
        }
        schema_nodes = self.structured_query(
            "MATCH (n) UNWIND labels(n) AS label RETURN label AS node, COUNT(n) AS count ORDER BY count DESC"
        )
        schema_rels = self.structured_query(
            "MATCH ()-[r]->() RETURN TYPE(r) AS relationship_type, COUNT(r) AS count"
        )
        schema_counts = [
            {
                "nodes": [
                    {"name": item["node"], "count": item["count"]}
                    for item in schema_nodes
                ],
                "relationships": [
                    {"name": item["relationship_type"], "count": item["count"]}
                    for item in schema_rels
                ],
            }
        ]
        # Update node info
        for node in schema_counts[0].get("nodes", []):
            # Skip bloom labels
            if node["name"] in EXCLUDED_LABELS:
                continue
            node_props = self.structured_schema["node_props"].get(node["name"])
            if not node_props:  # The node has no properties
                continue

            enhanced_cypher = self._enhanced_schema_cypher(
                node["name"], node_props, node["count"] < EXHAUSTIVE_SEARCH_LIMIT
            )
            output = self.structured_query(enhanced_cypher)
            enhanced_info = output[0]["output"]
            for prop in node_props:
                if prop["property"] in enhanced_info:
                    prop.update(enhanced_info[prop["property"]])

        # Update rel info
        for rel in schema_counts[0].get("relationships", []):
            if rel["name"] in EXCLUDED_RELS:
                continue
            rel_props = self.structured_schema["rel_props"].get(f":`{rel['name']}`")
            if not rel_props:  # The rel has no properties
                continue
            enhanced_cypher = self._enhanced_schema_cypher(
                rel["name"],
                rel_props,
                rel["count"] < EXHAUSTIVE_SEARCH_LIMIT,
                is_relationship=True,
            )
            try:
                enhanced_info = self.structured_query(enhanced_cypher)[0]["output"]
                for prop in rel_props:
                    if prop["property"] in enhanced_info:
                        prop.update(enhanced_info[prop["property"]])
            except neo4j.exceptions.ClientError:
                pass

    def upsert_nodes(self, nodes: List[LabelledNode]) -> None:
        # Lists to hold separated types
        entity_dicts: List[dict] = []
        chunk_dicts: List[dict] = []

        # Sort by type
        for item in nodes:
            if isinstance(item, EntityNode):
                entity_dicts.append({**item.dict(), "id": item.id})
            elif isinstance(item, ChunkNode):
                chunk_dicts.append({**item.dict(), "id": item.id})
            else:
                pass
        if chunk_dicts:
            for index in range(0, len(chunk_dicts), CHUNK_SIZE):
                chunked_params = chunk_dicts[index : index + CHUNK_SIZE]
                for param in chunked_params:
                    formatted_properties = ", ".join(
                        [
                            f"{key}: {value!r}"
                            for key, value in param["properties"].items()
                        ]
                    )
                    self.structured_query(
                        f"""
                        MERGE (c:{BASE_NODE_LABEL} {{id: '{param["id"]}'}})
                        SET c.`text` = '{param["text"]}', c:Chunk
                        WITH c
                        SET c += {{{formatted_properties}}}
                        RETURN count(*)
                        """
                    )
        if entity_dicts:
            for index in range(0, len(entity_dicts), CHUNK_SIZE):
                chunked_params = entity_dicts[index : index + CHUNK_SIZE]
                for param in chunked_params:
                    formatted_properties = ", ".join(
                        [
                            f"{key}: {value!r}"
                            for key, value in param["properties"].items()
                        ]
                    )
                    self.structured_query(
                        f"""
                        MERGE (e:{BASE_NODE_LABEL} {{id: '{param["id"]}'}})
                        SET e += {{{formatted_properties}}}
                        SET e.name = '{param["name"]}', e:`{BASE_ENTITY_LABEL}`
                        WITH e
                        SET e :{param["label"]}
                        """
                    )
                    triplet_source_id = param["properties"].get("triplet_source_id")
                    if triplet_source_id:
                        self.structured_query(
                            f"""
                            MERGE (e:{BASE_NODE_LABEL} {{id: '{param["id"]}'}})
                            MERGE (c:{BASE_NODE_LABEL} {{id: '{triplet_source_id}'}})
                            MERGE (e)<-[:MENTIONS]-(c)
                            """
                        )

    def upsert_relations(self, relations: List[Relation]) -> None:
        """Add relations."""
        params = [r.dict() for r in relations]
        for index in range(0, len(params), CHUNK_SIZE):
            chunked_params = params[index : index + CHUNK_SIZE]
            for param in chunked_params:
                formatted_properties = ", ".join(
                    [f"{key}: {value!r}" for key, value in param["properties"].items()]
                )

                self.structured_query(
                    f"""
                    MERGE (source: {BASE_NODE_LABEL} {{id: '{param["source_id"]}'}})
                    ON CREATE SET source:Chunk
                    MERGE (target: {BASE_NODE_LABEL} {{id: '{param["target_id"]}'}})
                    ON CREATE SET target:Chunk
                    WITH source, target
                    MERGE (source)-[r:{param["label"]}]->(target)
                    SET r += {{{formatted_properties}}}
                    RETURN count(*)
                    """
                )

    def get(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        """Get nodes."""
        cypher_statement = f"MATCH (e:{BASE_NODE_LABEL}) "

        params = {}
        cypher_statement += "WHERE e.id IS NOT NULL "

        if ids:
            cypher_statement += "AND e.id IN $ids "
            params["ids"] = ids

        if properties:
            prop_list = []
            for i, prop in enumerate(properties):
                prop_list.append(f"e.`{prop}` = $property_{i}")
                params[f"property_{i}"] = properties[prop]
            cypher_statement += " AND " + " AND ".join(prop_list)

        return_statement = """
            RETURN
            e.id AS name,
            CASE
                WHEN labels(e)[0] IN ['__Entity__', '__Node__'] THEN
                    CASE
                        WHEN size(labels(e)) > 2 THEN labels(e)[2]
                        WHEN size(labels(e)) > 1 THEN labels(e)[1]
                        ELSE NULL
                    END
                ELSE labels(e)[0]
            END AS type,
            properties(e) AS properties
        """
        cypher_statement += return_statement
        response = self.structured_query(cypher_statement, param_map=params)
        response = response if response else []

        nodes = []
        for record in response:
            if "text" in record["properties"] or record["type"] is None:
                text = record["properties"].pop("text", "")
                nodes.append(
                    ChunkNode(
                        id_=record["name"],
                        text=text,
                        properties=remove_empty_values(record["properties"]),
                    )
                )
            else:
                nodes.append(
                    EntityNode(
                        name=record["name"],
                        label=record["type"],
                        properties=remove_empty_values(record["properties"]),
                    )
                )

        return nodes

    def get_triplets(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Triplet]:
        cypher_statement = f"MATCH (e:`{BASE_ENTITY_LABEL}`)-[r]->(t) "

        params = {}
        if entity_names or relation_names or properties or ids:
            cypher_statement += "WHERE "

        if entity_names:
            cypher_statement += "e.name in $entity_names "
            params["entity_names"] = entity_names

        if relation_names and entity_names:
            cypher_statement += f"AND "

        if relation_names:
            cypher_statement += "type(r) in $relation_names "
            params[f"relation_names"] = relation_names

        if ids:
            cypher_statement += "e.id in $ids "
            params["ids"] = ids

        if properties:
            prop_list = []
            for i, prop in enumerate(properties):
                prop_list.append(f"e.`{prop}` = $property_{i}")
                params[f"property_{i}"] = properties[prop]
            cypher_statement += " AND ".join(prop_list)

        if not (entity_names or properties or relation_names or ids):
            return_statement = """
                WHERE NOT ANY(label IN labels(e) WHERE label = 'Chunk')
                RETURN type(r) as type, properties(r) as rel_prop, e.id as source_id,
                CASE
                    WHEN labels(e)[0] IN ['__Entity__', '__Node__'] THEN
                        CASE
                            WHEN size(labels(e)) > 2 THEN labels(e)[2]
                            WHEN size(labels(e)) > 1 THEN labels(e)[1]
                            ELSE NULL
                        END
                    ELSE labels(e)[0]
                END AS source_type,
                properties(e) AS source_properties,
                t.id as target_id,
                CASE
                    WHEN labels(t)[0] IN ['__Entity__', '__Node__'] THEN
                        CASE
                            WHEN size(labels(t)) > 2 THEN labels(t)[2]
                            WHEN size(labels(t)) > 1 THEN labels(t)[1]
                            ELSE NULL
                        END
                    ELSE labels(t)[0]
                END AS target_type, properties(t) AS target_properties LIMIT 100;
            """
        else:
            return_statement = """
            AND NOT ANY(label IN labels(e) WHERE label = 'Chunk')
                RETURN type(r) as type, properties(r) as rel_prop, e.id as source_id,
                CASE
                    WHEN labels(e)[0] IN ['__Entity__', '__Node__'] THEN
                        CASE
                            WHEN size(labels(e)) > 2 THEN labels(e)[2]
                            WHEN size(labels(e)) > 1 THEN labels(e)[1]
                            ELSE NULL
                        END
                    ELSE labels(e)[0]
                END AS source_type,
                properties(e) AS source_properties,
                t.id as target_id,
                CASE
                    WHEN labels(t)[0] IN ['__Entity__', '__Node__'] THEN
                        CASE
                            WHEN size(labels(t)) > 2 THEN labels(t)[2]
                            WHEN size(labels(t)) > 1 THEN labels(t)[1]
                            ELSE NULL
                        END
                    ELSE labels(t)[0]
                END AS target_type, properties(t) AS target_properties LIMIT 100;
            """

        cypher_statement += return_statement
        data = self.structured_query(cypher_statement, param_map=params)
        data = data if data else []

        triplets = []
        for record in data:
            source = EntityNode(
                name=record["source_id"],
                label=record["source_type"],
                properties=remove_empty_values(record["source_properties"]),
            )
            target = EntityNode(
                name=record["target_id"],
                label=record["target_type"],
                properties=remove_empty_values(record["target_properties"]),
            )
            rel = Relation(
                source_id=record["source_id"],
                target_id=record["target_id"],
                label=record["type"],
                properties=remove_empty_values(record["rel_prop"]),
            )
            triplets.append([source, rel, target])
        return triplets

    def get_rel_map(
        self,
        graph_nodes: List[LabelledNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Get depth-aware rel map."""
        triples = []

        ids = [node.id for node in graph_nodes]
        response = self.structured_query(
            f"""
            WITH $ids AS id_list
            UNWIND range(0, size(id_list) - 1) AS idx
            MATCH (e:__Node__)
            WHERE e.id = id_list[idx]
            MATCH p=(e)-[r*1..{depth}]-(other)
            WHERE ALL(rel in relationships(p) WHERE type(rel) <> 'MENTIONS')
            UNWIND relationships(p) AS rel
            WITH DISTINCT rel, idx
            WITH startNode(rel) AS source,
                type(rel) AS type,
                rel{{.*}} AS rel_properties,
                endNode(rel) AS endNode,
                idx
            LIMIT toInteger($limit)
            RETURN source.id AS source_id,
                CASE
                    WHEN labels(source)[0] IN ['__Entity__', '__Node__'] THEN
                        CASE
                            WHEN size(labels(source)) > 2 THEN labels(source)[2]
                            WHEN size(labels(source)) > 1 THEN labels(source)[1]
                            ELSE NULL
                        END
                    ELSE labels(source)[0]
                END AS source_type,
                properties(source) AS source_properties,
                type,
                rel_properties,
                endNode.id AS target_id,
                CASE
                    WHEN labels(endNode)[0] IN ['__Entity__', '__Node__'] THEN
                        CASE
                            WHEN size(labels(endNode)) > 2 THEN labels(endNode)[2]
                            WHEN size(labels(endNode)) > 1 THEN labels(endNode)[1] ELSE NULL
                        END
                    ELSE labels(endNode)[0]
                END AS target_type,
                properties(endNode) AS target_properties,
                idx
            ORDER BY idx
            LIMIT toInteger($limit)
            """,
            param_map={"ids": ids, "limit": limit},
        )
        response = response if response else []

        ignore_rels = ignore_rels or []
        for record in response:
            if record["type"] in ignore_rels:
                continue

            source = EntityNode(
                name=record["source_id"],
                label=record["source_type"],
                properties=remove_empty_values(record["source_properties"]),
            )
            target = EntityNode(
                name=record["target_id"],
                label=record["target_type"],
                properties=remove_empty_values(record["target_properties"]),
            )
            rel = Relation(
                source_id=record["source_id"],
                target_id=record["target_id"],
                label=record["type"],
                properties=remove_empty_values(record["rel_properties"]),
            )
            triples.append([source, rel, target])

        return triples

    def structured_query(
        self, query: str, param_map: Optional[Dict[str, Any]] = None
    ) -> Any:
        param_map = param_map or {}

        with self._driver.session(database=self._database) as session:
            result = session.run(query, param_map)
            full_result = [d.data() for d in result]

        if self.sanitize_query_output:
            return [value_sanitize(el) for el in full_result]
        return full_result

    def vector_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> Tuple[List[LabelledNode], List[float]]:
        raise NotImplementedError(
            "Vector query is not currently implemented for MemgraphPropertyGraphStore."
        )

    def delete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Delete matching data."""
        if entity_names:
            self.structured_query(
                "MATCH (n) WHERE n.name IN $entity_names DETACH DELETE n",
                param_map={"entity_names": entity_names},
            )
        if ids:
            self.structured_query(
                "MATCH (n) WHERE n.id IN $ids DETACH DELETE n",
                param_map={"ids": ids},
            )
        if relation_names:
            for rel in relation_names:
                self.structured_query(f"MATCH ()-[r:`{rel}`]->() DELETE r")

        if properties:
            cypher = "MATCH (e) WHERE "
            prop_list = []
            params = {}
            for i, prop in enumerate(properties):
                prop_list.append(f"e.`{prop}` = $property_{i}")
                params[f"property_{i}"] = properties[prop]
            cypher += " AND ".join(prop_list)
            self.structured_query(cypher + " DETACH DELETE e", param_map=params)

    def _enhanced_schema_cypher(
        self,
        label_or_type: str,
        properties: List[Dict[str, Any]],
        exhaustive: bool,
        is_relationship: bool = False,
    ) -> str:
        if is_relationship:
            match_clause = f"MATCH ()-[n:`{label_or_type}`]->()"
        else:
            match_clause = f"MATCH (n:`{label_or_type}`)"

        with_clauses = []
        return_clauses = []
        output_dict = {}
        if exhaustive:
            for prop in properties:
                if prop["property"]:
                    prop_name = prop["property"]
                else:
                    prop_name = None
                if prop["type"]:
                    prop_type = prop["type"]
                else:
                    prop_type = None
                if prop_type == "String":
                    with_clauses.append(
                        f"collect(distinct substring(toString(n.`{prop_name}`), 0, 50)) "
                        f"AS `{prop_name}_values`"
                    )
                    return_clauses.append(
                        f"values:`{prop_name}_values`[..{DISTINCT_VALUE_LIMIT}],"
                        f" distinct_count: size(`{prop_name}_values`)"
                    )
                elif prop_type in [
                    "Int",
                    "Double",
                    "Date",
                    "LocalTime",
                    "LocalDateTime",
                ]:
                    with_clauses.append(f"min(n.`{prop_name}`) AS `{prop_name}_min`")
                    with_clauses.append(f"max(n.`{prop_name}`) AS `{prop_name}_max`")
                    with_clauses.append(
                        f"count(distinct n.`{prop_name}`) AS `{prop_name}_distinct`"
                    )
                    return_clauses.append(
                        f"min: toString(`{prop_name}_min`), "
                        f"max: toString(`{prop_name}_max`), "
                        f"distinct_count: `{prop_name}_distinct`"
                    )
                elif prop_type in ["List", "List[Any]"]:
                    with_clauses.append(
                        f"min(size(n.`{prop_name}`)) AS `{prop_name}_size_min`, "
                        f"max(size(n.`{prop_name}`)) AS `{prop_name}_size_max`"
                    )
                    return_clauses.append(
                        f"min_size: `{prop_name}_size_min`, "
                        f"max_size: `{prop_name}_size_max`"
                    )
                elif prop_type in ["Bool", "Duration"]:
                    continue
                if return_clauses:
                    output_dict[prop_name] = "{" + return_clauses.pop() + "}"
                else:
                    output_dict[prop_name] = None
        else:
            # Just sample 5 random nodes
            match_clause += " WITH n LIMIT 5"
            for prop in properties:
                prop_name = prop["property"]
                prop_type = prop["type"]

                # Check if indexed property, we can still do exhaustive
                prop_index = [
                    el
                    for el in self.structured_schema["metadata"]["index"]
                    if el["label"] == label_or_type
                    and el["properties"] == [prop_name]
                    and el["type"] == "RANGE"
                ]
                if prop_type == "String":
                    if (
                        prop_index
                        and prop_index[0].get("size") > 0
                        and prop_index[0].get("distinctValues") <= DISTINCT_VALUE_LIMIT
                    ):
                        distinct_values_query = f"""
                            MATCH (n:{label_or_type})
                            RETURN DISTINCT n.`{prop_name}` AS value
                            LIMIT {DISTINCT_VALUE_LIMIT}
                        """
                        distinct_values = self.query(distinct_values_query)

                        # Extract values from the result set
                        distinct_values = [
                            record["value"] for record in distinct_values
                        ]

                        return_clauses.append(
                            f"values: {distinct_values},"
                            f" distinct_count: {len(distinct_values)}"
                        )
                    else:
                        with_clauses.append(
                            f"collect(distinct substring(n.`{prop_name}`, 0, 50)) "
                            f"AS `{prop_name}_values`"
                        )
                        return_clauses.append(f"values: `{prop_name}_values`")
                elif prop_type in [
                    "Int",
                    "Double",
                    "Float",
                    "Date",
                    "LocalTime",
                    "LocalDateTime",
                ]:
                    if not prop_index:
                        with_clauses.append(
                            f"collect(distinct toString(n.`{prop_name}`)) "
                            f"AS `{prop_name}_values`"
                        )
                        return_clauses.append(f"values: `{prop_name}_values`")
                    else:
                        with_clauses.append(
                            f"min(n.`{prop_name}`) AS `{prop_name}_min`"
                        )
                        with_clauses.append(
                            f"max(n.`{prop_name}`) AS `{prop_name}_max`"
                        )
                        with_clauses.append(
                            f"count(distinct n.`{prop_name}`) AS `{prop_name}_distinct`"
                        )
                        return_clauses.append(
                            f"min: toString(`{prop_name}_min`), "
                            f"max: toString(`{prop_name}_max`), "
                            f"distinct_count: `{prop_name}_distinct`"
                        )

                elif prop_type in ["List", "List[Any]"]:
                    with_clauses.append(
                        f"min(size(n.`{prop_name}`)) AS `{prop_name}_size_min`, "
                        f"max(size(n.`{prop_name}`)) AS `{prop_name}_size_max`"
                    )
                    return_clauses.append(
                        f"min_size: `{prop_name}_size_min`, "
                        f"max_size: `{prop_name}_size_max`"
                    )
                elif prop_type in ["Bool", "Duration"]:
                    continue
                if return_clauses:
                    output_dict[prop_name] = "{" + return_clauses.pop() + "}"
                else:
                    output_dict[prop_name] = None

        with_clause = "WITH " + ",\n     ".join(with_clauses)
        return_clause = (
            "RETURN {"
            + ", ".join(f"`{k}`: {v}" for k, v in output_dict.items())
            + "} AS output"
        )
        # Combine all parts of the Cypher query
        return f"{match_clause}\n{with_clause}\n{return_clause}"

    def get_schema(self, refresh: bool = False) -> Any:
        if refresh:
            self.refresh_schema()

        return self.structured_schema

    def get_schema_str(self, refresh: bool = False) -> str:
        schema = self.get_schema(refresh=refresh)

        formatted_node_props = []
        formatted_rel_props = []

        if self.enhanced_schema:
            # Enhanced formatting for nodes
            for node_type, properties in schema["node_props"].items():
                formatted_node_props.append(f"- **{node_type}**")
                for prop in properties:
                    example = ""
                    if prop["type"] == "String" and prop.get("values"):
                        if prop.get("distinct_count", 11) > DISTINCT_VALUE_LIMIT:
                            example = (
                                f'Example: "{clean_string_values(prop["values"][0])}"'
                                if prop["values"]
                                else ""
                            )
                        else:  # If less than 10 possible values return all
                            example = (
                                (
                                    "Available options: "
                                    f'{[clean_string_values(el) for el in prop["values"]]}'
                                )
                                if prop["values"]
                                else ""
                            )

                    elif prop["type"] in [
                        "Int",
                        "Double",
                        "Float",
                        "Date",
                        "LocalTime",
                        "LocalDateTime",
                    ]:
                        if prop.get("min") is not None:
                            example = f'Min: {prop["min"]}, Max: {prop["max"]}'
                        else:
                            example = (
                                f'Example: "{prop["values"][0]}"'
                                if prop.get("values")
                                else ""
                            )
                    elif prop["type"] in ["List", "List[Any]"]:
                        # Skip embeddings
                        if not prop.get("min_size") or prop["min_size"] > LIST_LIMIT:
                            continue
                        example = f'Min Size: {prop["min_size"]}, Max Size: {prop["max_size"]}'
                    formatted_node_props.append(
                        f"  - `{prop['property']}`: {prop['type']} {example}"
                    )

            # Enhanced formatting for relationships
            for rel_type, properties in schema["rel_props"].items():
                formatted_rel_props.append(f"- **{rel_type}**")
                for prop in properties:
                    example = ""
                    if prop["type"] == "STRING":
                        if prop.get("distinct_count", 11) > DISTINCT_VALUE_LIMIT:
                            example = (
                                f'Example: "{clean_string_values(prop["values"][0])}"'
                                if prop.get("values")
                                else ""
                            )
                        else:  # If less than 10 possible values return all
                            example = (
                                (
                                    "Available options: "
                                    f'{[clean_string_values(el) for el in prop["values"]]}'
                                )
                                if prop.get("values")
                                else ""
                            )
                    elif prop["type"] in [
                        "Int",
                        "Double",
                        "Float",
                        "Date",
                        "LocalTime",
                        "LocalDateTime",
                    ]:
                        if prop.get("min"):  # If we have min/max
                            example = f'Min: {prop["min"]}, Max:  {prop["max"]}'
                        else:  # return a single value
                            example = (
                                f'Example: "{prop["values"][0]}"'
                                if prop.get("values")
                                else ""
                            )
                    elif prop["type"] == "List[Any]":
                        # Skip embeddings
                        if prop["min_size"] > LIST_LIMIT:
                            continue
                        example = f'Min Size: {prop["min_size"]}, Max Size: {prop["max_size"]}'
                    formatted_rel_props.append(
                        f"  - `{prop['property']}: {prop['type']}` {example}"
                    )
        else:
            # Format node properties
            for label, props in schema["node_props"].items():
                props_str = ", ".join(
                    [f"{prop['property']}: {prop['type']}" for prop in props]
                )
                formatted_node_props.append(f"{label} {{{props_str}}}")

            # Format relationship properties using structured_schema
            for type, props in schema["rel_props"].items():
                props_str = ", ".join(
                    [f"{prop['property']}: {prop['type']}" for prop in props]
                )
                formatted_rel_props.append(f"{type} {{{props_str}}}")

        # Format relationships
        formatted_rels = [
            f"(:{el['start']})-[:{el['type']}]->(:{el['end']})"
            for el in schema["relationships"]
        ]

        return "\n".join(
            [
                "Node properties:",
                "\n".join(formatted_node_props),
                "Relationship properties:",
                "\n".join(formatted_rel_props),
                "The relationships:",
                "\n".join(formatted_rels),
            ]
        )
