# memgraph

Memgraph MAGE - lightweight graph database with 30+ built-in algorithms.

**Image:** `memgraph/memgraph-mage:3.9.0`
**Port:** 7687 Bolt protocol (configurable via `MEMGRAPH_PORT`)

## What it does

Stores the knowledge graph: concepts (nodes) and relations (edges). Supports Cypher queries for graph traversal, shortest path, and built-in algorithms like PageRank and community detection.

Concepts are upserted by name (unique constraint enforced at startup). Self-referencing relations are blocked by the API.

## Key algorithms (via MAGE)

```cypher
-- PageRank
CALL pagerank.get() YIELD node, rank;

-- Community detection (Louvain)
CALL community_detection.get() YIELD node, community_id;

-- Shortest path (built-in Cypher)
MATCH p = (a)-[*BFS]-(b) WHERE id(a) = 0 AND id(b) = 5 RETURN p;
```

## Connection

Bolt protocol only (no REST API). The API container connects using the `neo4j` Python async driver with `bolt://memgraph:7687`.

## Persistence

Data is bind-mounted to `./data/memgraph` in the rag-base repo (`/var/lib/memgraph` inside the container). Survives `docker compose down` AND `docker volume prune`. Wipe by deleting `./data/memgraph`. The `./data/` tree is gitignored.

**First-boot prereq.** Memgraph runs as UID 101 inside the container. The host bind dir must be owned by 101 or Memgraph SIGSEGVs at startup with banner-only logs:

```bash
mkdir -p ./data/memgraph
docker run --rm -v "$PWD/data/memgraph":/d alpine chown -R 101:101 /d
```

## Optional

This container only starts with the `graph` profile: `docker compose --profile graph up -d`. The API detects its absence and disables graph endpoints gracefully (returns 503).
