"""Tests for RRF fusion logic."""

from api.src.services.fusion import reciprocal_rank_fusion


def test_single_list():
    """Single retriever — RRF returns same order with scores."""
    results = [
        {"chunk_id": 1, "content": "a", "source": "semantic"},
        {"chunk_id": 2, "content": "b", "source": "semantic"},
    ]
    merged = reciprocal_rank_fusion(results)
    assert len(merged) == 2
    assert merged[0]["chunk_id"] == 1
    assert merged[1]["chunk_id"] == 2
    assert merged[0]["score"] > merged[1]["score"]


def test_two_lists_agreement():
    """Two retrievers agree on top result — it should score highest."""
    semantic = [
        {"chunk_id": 1, "content": "a", "source": "semantic"},
        {"chunk_id": 2, "content": "b", "source": "semantic"},
    ]
    keyword = [
        {"chunk_id": 1, "content": "a", "source": "keyword"},
        {"chunk_id": 3, "content": "c", "source": "keyword"},
    ]
    merged = reciprocal_rank_fusion(semantic, keyword)
    # chunk_id=1 appears in both lists at rank 1 → highest fused score
    assert merged[0]["chunk_id"] == 1
    assert merged[0]["score"] > merged[1]["score"]


def test_two_lists_disagreement():
    """Different top results — both should appear, fused correctly."""
    semantic = [
        {"chunk_id": 1, "content": "a", "source": "semantic"},
        {"chunk_id": 2, "content": "b", "source": "semantic"},
    ]
    keyword = [
        {"chunk_id": 3, "content": "c", "source": "keyword"},
        {"chunk_id": 2, "content": "b", "source": "keyword"},
    ]
    merged = reciprocal_rank_fusion(semantic, keyword)
    # chunk_id=2 appears in both → should rank higher than 1 or 3
    ids = [m["chunk_id"] for m in merged]
    assert 2 in ids
    assert merged[0]["chunk_id"] == 2  # appears in both at rank 2 each


def test_sources_tracked():
    """Sources from each retriever are tracked."""
    semantic = [{"chunk_id": 1, "content": "a", "source": "semantic"}]
    keyword = [{"chunk_id": 1, "content": "a", "source": "keyword"}]
    merged = reciprocal_rank_fusion(semantic, keyword)
    assert "semantic" in merged[0]["sources"]
    assert "keyword" in merged[0]["sources"]


def test_empty_lists():
    """Empty input returns empty output."""
    merged = reciprocal_rank_fusion()
    assert merged == []


def test_rrf_score_formula():
    """Verify the RRF score formula: 1/(k + rank)."""
    results = [{"chunk_id": 1, "content": "a", "source": "test"}]
    merged = reciprocal_rank_fusion(results, k=60)
    # rank=1, k=60 → score = 1/61
    expected = 1.0 / 61
    assert abs(merged[0]["score"] - expected) < 1e-9
