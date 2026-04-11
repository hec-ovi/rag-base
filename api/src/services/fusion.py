"""Reciprocal Rank Fusion (RRF) — merge results from multiple retrievers."""

# RRF constant (standard value from the original paper)
RRF_K = 60


def reciprocal_rank_fusion(
    *ranked_lists: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """Merge multiple ranked result lists using RRF.

    Each input list contains dicts with at least a "chunk_id" key.
    Returns a single merged list sorted by fused score, descending.

    Formula: score(d) = sum(1 / (k + rank_i)) for each retriever i
    where rank_i is the 1-based rank of document d in retriever i.
    """
    scores: dict[int, float] = {}
    items: dict[int, dict] = {}
    sources: dict[int, list[str]] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            chunk_id = item["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            items[chunk_id] = item
            if chunk_id not in sources:
                sources[chunk_id] = []
            if "source" in item:
                sources[chunk_id].append(item["source"])

    merged = []
    for chunk_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        result = {**items[chunk_id], "score": score, "sources": sources.get(chunk_id, [])}
        merged.append(result)

    return merged
