"""Unit tests for Phase 5 graph-only retrieval helpers.

These tests do not need Memgraph, GLiNER, or Postgres. They cover the pure-Python
helpers in `api/src/services/graph_only_search.py`: source_id splitting, doc_id
parsing, and the node ranking logic.
"""

from api.src.services.graph_only_search import (
    _parse_doc_id,
    _rank_nodes,
    _split_source_id,
)


# ── _split_source_id ─────────────────────────────────────────────────


def test_split_source_id_single_chunk():
    assert _split_source_id("chunk-abc123") == ["chunk-abc123"]


def test_split_source_id_multi_chunk():
    s = "chunk-abc123<SEP>chunk-def456<SEP>chunk-ghi789"
    assert _split_source_id(s) == ["chunk-abc123", "chunk-def456", "chunk-ghi789"]


def test_split_source_id_empty():
    assert _split_source_id("") == []
    assert _split_source_id(None) == []  # type: ignore[arg-type]


def test_split_source_id_strips_whitespace():
    assert _split_source_id(" chunk-x <SEP>  chunk-y  ") == ["chunk-x", "chunk-y"]


# ── _parse_doc_id ────────────────────────────────────────────────────


def test_parse_doc_id_normal():
    assert _parse_doc_id("doc_123") == 123
    assert _parse_doc_id("doc_0") == 0


def test_parse_doc_id_invalid_format():
    assert _parse_doc_id("document_123") is None
    assert _parse_doc_id("doc-123") is None
    assert _parse_doc_id("123") is None
    assert _parse_doc_id("") is None
    assert _parse_doc_id(None) is None


def test_parse_doc_id_garbage():
    assert _parse_doc_id("doc_abc") is None
    assert _parse_doc_id("doc_") is None


# ── _rank_nodes ──────────────────────────────────────────────────────


def _node(nid: str, ntype: str = "person") -> dict:
    return {"id": nid, "type": ntype, "description": "", "source_id": ""}


def _edge(src: str, tgt: str, rel: str = "rel") -> dict:
    return {"source": src, "target": tgt, "description": rel, "weight": 1.0, "keywords": ""}


def test_rank_nodes_none_preserves_order():
    nodes = [_node("A"), _node("B"), _node("C")]
    edges = [_edge("A", "B"), _edge("A", "C"), _edge("A", "C")]  # A has degree 3
    ranked = _rank_nodes(nodes, edges, seed_ids={"A"}, ranking="none")
    assert [n["id"] for n in ranked] == ["A", "B", "C"]
    # Degree is computed and attached even in 'none' mode for the response payload.
    assert all("_degree" in n for n in ranked)


def test_rank_nodes_degree_seeds_first_then_by_degree_desc():
    nodes = [_node("seed1"), _node("low"), _node("high"), _node("mid")]
    edges = [
        _edge("seed1", "high"),
        _edge("seed1", "high"),
        _edge("seed1", "high"),  # high: degree 3 (excluding seed1's contribution from seed)
        _edge("seed1", "mid"),
        _edge("seed1", "mid"),  # mid: degree 2
        _edge("seed1", "low"),  # low: degree 1
    ]
    ranked = _rank_nodes(nodes, edges, seed_ids={"seed1"}, ranking="degree")
    assert ranked[0]["id"] == "seed1"  # seed always first
    # Non-seeds in degree desc order
    assert [n["id"] for n in ranked[1:]] == ["high", "mid", "low"]


def test_rank_nodes_degree_with_no_edges():
    nodes = [_node("A"), _node("B")]
    ranked = _rank_nodes(nodes, [], seed_ids={"A"}, ranking="degree")
    assert ranked[0]["id"] == "A"  # seed first even with no edges
    assert ranked[1]["id"] == "B"
    assert all(n["_degree"] == 0 for n in ranked)


def test_rank_nodes_empty_input():
    assert _rank_nodes([], [], seed_ids=set(), ranking="degree") == []
    assert _rank_nodes([], [], seed_ids=set(), ranking="none") == []


def test_rank_nodes_multiple_seeds_preserved_order():
    nodes = [_node("s1"), _node("s2"), _node("other")]
    edges = [_edge("s1", "other"), _edge("s2", "other")]
    ranked = _rank_nodes(nodes, edges, seed_ids={"s1", "s2"}, ranking="degree")
    # Both seeds appear before non-seeds; order among seeds matches input order
    assert [n["id"] for n in ranked[:2]] == ["s1", "s2"]
    assert ranked[2]["id"] == "other"
