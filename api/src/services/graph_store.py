"""Memgraph graph operations via neo4j driver."""

import json
import re

from neo4j import AsyncDriver


def _serialize_meta(metadata: dict) -> str:
    """Serialize metadata dict to JSON string for Memgraph storage."""
    return json.dumps(metadata) if metadata else "{}"


def _parse_meta(raw: str | None) -> dict:
    """Parse metadata JSON string from Memgraph back to dict."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _sanitize_relation_type(relation_type: str) -> str:
    """Sanitize a relation type for safe Cypher interpolation.

    Cypher does not support parameterized relationship types, so we must
    interpolate the string. Only allow alphanumeric + underscore.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", relation_type.upper().strip())
    if not cleaned or not cleaned[0].isalpha():
        cleaned = "REL_" + cleaned
    return cleaned


async def ensure_indexes(driver: AsyncDriver) -> None:
    """Create unique constraint on concept names. Idempotent."""
    async with driver.session() as session:
        await session.run("CREATE CONSTRAINT ON (c:Concept) ASSERT c.name IS UNIQUE;")


async def create_concept(driver: AsyncDriver, name: str, type: str, description: str, metadata: dict) -> dict:
    """Create or update a concept node in the graph. Upserts by name."""
    async with driver.session() as session:
        result = await session.run(
            """
            MERGE (c:Concept {name: $name})
            ON CREATE SET c.type = $type, c.description = $description, c.metadata = $metadata
            ON MATCH SET c.type = $type, c.description = $description, c.metadata = $metadata
            RETURN id(c) AS id, c.name AS name, c.type AS type,
                   c.description AS description, c.metadata AS metadata
            """,
            name=name, type=type, description=description, metadata=_serialize_meta(metadata),
        )
        record = await result.single()
        data = dict(record)
        data["metadata"] = _parse_meta(data.get("metadata"))
        return data


async def get_concept(driver: AsyncDriver, concept_id: int) -> dict | None:
    """Get a concept by internal ID with its relations."""
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (c:Concept) WHERE id(c) = $id
            OPTIONAL MATCH (c)-[r]->(t:Concept)
            RETURN id(c) AS id, c.name AS name, c.type AS type,
                   c.description AS description, c.metadata AS metadata,
                   collect({relation_type: type(r), target: t.name, target_id: id(t)}) AS relations
            """,
            id=concept_id,
        )
        record = await result.single()
        if not record:
            return None
        data = dict(record)
        data["metadata"] = _parse_meta(data.get("metadata"))
        # Filter out empty relations from OPTIONAL MATCH
        data["relations"] = [r for r in data["relations"] if r["target"] is not None]
        return data


async def delete_concept(driver: AsyncDriver, concept_id: int) -> bool:
    """Delete a concept and all its edges."""
    async with driver.session() as session:
        result = await session.run(
            "MATCH (c:Concept) WHERE id(c) = $id DETACH DELETE c RETURN count(c) AS deleted",
            id=concept_id,
        )
        record = await result.single()
        return record["deleted"] > 0


async def create_relation(
    driver: AsyncDriver,
    source_name: str,
    target_name: str,
    relation_type: str,
    metadata: dict,
) -> dict:
    """Create a typed directed edge between two concepts (by name)."""
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (s:Concept {name: $source}), (t:Concept {name: $target})
            CREATE (s)-[r:"""
            + _sanitize_relation_type(relation_type)
            + """ {metadata: $metadata}]->(t)
            RETURN id(r) AS id, s.name AS source, t.name AS target,
                   type(r) AS relation_type, r.metadata AS metadata
            """,
            source=source_name,
            target=target_name,
            metadata=_serialize_meta(metadata),
        )
        record = await result.single()
        if not record:
            return None
        data = dict(record)
        data["metadata"] = _parse_meta(data.get("metadata"))
        return data


async def get_relations(driver: AsyncDriver, concept_name: str) -> list[dict]:
    """Get all relations for a concept (both directions)."""
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (c:Concept {name: $name})-[r]-(other:Concept)
            RETURN id(r) AS id, c.name AS source, other.name AS target,
                   type(r) AS relation_type, r.metadata AS metadata
            """,
            name=concept_name,
        )
        records = [dict(record) async for record in result]
        for r in records:
            r["metadata"] = _parse_meta(r.get("metadata"))
        return records


async def delete_relation(driver: AsyncDriver, relation_id: int) -> bool:
    """Delete a relation by internal ID."""
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH ()-[r]->() WHERE id(r) = $id
            DELETE r RETURN count(r) AS deleted
            """,
            id=relation_id,
        )
        record = await result.single()
        return record["deleted"] > 0


async def get_neighbors(driver: AsyncDriver, concept_id: int, depth: int = 2) -> list[dict]:
    """Multi-hop neighborhood traversal."""
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (start:Concept) WHERE id(start) = $id
            MATCH path = (start)-[*1.."""
            + str(min(depth, 5))
            + """]->(neighbor:Concept)
            RETURN DISTINCT id(neighbor) AS id, neighbor.name AS name,
                   neighbor.type AS type, length(path) AS depth
            ORDER BY depth, name
            """,
            id=concept_id,
        )
        return [dict(record) async for record in result]


async def get_shortest_path(driver: AsyncDriver, from_id: int, to_id: int) -> list[dict]:
    """Shortest path between two concepts."""
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (a:Concept) WHERE id(a) = $from_id
            MATCH (b:Concept) WHERE id(b) = $to_id
            MATCH p = (a)-[*BFS]-(b)
            UNWIND nodes(p) AS n
            RETURN id(n) AS id, n.name AS name, n.type AS type
            """,
            from_id=from_id,
            to_id=to_id,
        )
        return [dict(record) async for record in result]


async def get_communities(driver: AsyncDriver) -> list[dict]:
    """Community detection via Louvain algorithm (MAGE)."""
    async with driver.session() as session:
        result = await session.run(
            """
            CALL community_detection.get()
            YIELD node, community_id
            RETURN id(node) AS id, node.name AS name, node.type AS type, community_id
            ORDER BY community_id, name
            """
        )
        return [dict(record) async for record in result]


async def get_stats(driver: AsyncDriver) -> dict:
    """Graph statistics: node count, edge count."""
    async with driver.session() as session:
        nodes_result = await session.run("MATCH (n:Concept) RETURN count(n) AS count")
        nodes_record = await nodes_result.single()
        edges_result = await session.run("MATCH ()-[r]->() RETURN count(r) AS count")
        edges_record = await edges_result.single()
        return {
            "concepts": nodes_record["count"],
            "relations": edges_record["count"],
        }


async def graph_search_expansion(
    driver: AsyncDriver,
    entity_names: list[str],
    depth: int = 2,
) -> list[dict]:
    """Given entity names found in a query, expand their graph neighborhood.

    Returns chunk-like results with concept info for RRF fusion.
    """
    if not entity_names:
        return []

    async with driver.session() as session:
        result = await session.run(
            """
            UNWIND $names AS name
            MATCH (c:Concept) WHERE toLower(c.name) = toLower(name)
            MATCH path = (c)-[*1.."""
            + str(min(depth, 3))
            + """]->(neighbor:Concept)
            RETURN DISTINCT id(neighbor) AS concept_id, neighbor.name AS name,
                   neighbor.type AS type, neighbor.description AS description,
                   length(path) AS depth
            ORDER BY depth, name
            """,
            names=entity_names,
        )
        return [dict(record) async for record in result]
