"""Integration: all 5 services are healthy."""

import httpx
import pytest


@pytest.mark.asyncio
async def test_health_all_services_up(api: httpx.AsyncClient):
    """The compose stack reports postgres + embedding + reranker + memgraph all healthy."""
    r = await api.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok", body
    assert body["postgres"] == "connected", body
    assert body["embedding"] == "healthy", body
    assert body["reranker"] == "healthy", body
    assert body["memgraph"] == "connected", body


@pytest.mark.asyncio
async def test_health_models_reports_active_models(api: httpx.AsyncClient):
    """/health/models returns the loaded embedding and reranker model identifiers."""
    r = await api.get("/health/models")
    assert r.status_code == 200
    body = r.json()
    assert "embedding" in body
    assert "reranker" in body
    assert body["embedding"]["model_id"] == "BAAI/bge-m3", body["embedding"]
    assert body["reranker"]["model_id"] == "BAAI/bge-reranker-v2-m3", body["reranker"]
