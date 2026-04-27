"""Integration test fixtures for rag-base.

These tests require the docker compose stack to be running locally.
They hit the live API at http://localhost:5050 and clean up after themselves.
"""

import os
from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio

API_URL = os.environ.get("RAGBASE_API_URL", "http://localhost:5050")


@pytest_asyncio.fixture
async def api() -> AsyncIterator[httpx.AsyncClient]:
    """httpx client pointed at the local rag-base API.

    Skips the test if the API is not reachable so the test suite still passes
    in environments without docker (e.g. unit-only CI).
    """
    # Long timeout: documents POST awaits LightRAG ingest inline, which can take
    # 5-10 minutes per doc on a local reasoning vLLM. 30 min ceiling gives us
    # a real "did the ingest finish" signal rather than a spurious test timeout.
    async with httpx.AsyncClient(base_url=API_URL, timeout=1800.0) as client:
        try:
            r = await client.get("/health")
            if r.status_code != 200:
                pytest.skip(f"rag-base API at {API_URL} not healthy ({r.status_code})")
        except httpx.ConnectError:
            pytest.skip(f"rag-base API at {API_URL} not reachable")
        yield client


@pytest_asyncio.fixture
async def created_docs(api: httpx.AsyncClient) -> AsyncIterator[list[int]]:
    """Tracks created document ids; deletes them at teardown."""
    ids: list[int] = []
    yield ids
    for doc_id in ids:
        try:
            await api.delete(f"/v1/documents/{doc_id}")
        except httpx.HTTPError:
            pass


async def ingest(
    api: httpx.AsyncClient,
    title: str,
    content: str,
    metadata: dict | None = None,
    *,
    with_graph: bool = False,
) -> dict:
    """Helper: ingest a doc, return the response dict including id.

    By default, sends `X-LightRAG-Ingest: false` so the API skips entity
    extraction. The full LightRAG path is ~9 minutes per doc on the local
    reasoning LLM and only the graph-specific tests need it. Tests that
    assert on graph behavior pass `with_graph=True` to opt in.
    """
    payload = {"title": title, "content": content, "metadata": metadata or {}}
    headers = {} if with_graph else {"X-LightRAG-Ingest": "false"}
    r = await api.post("/v1/documents", json=payload, headers=headers)
    r.raise_for_status()
    return r.json()
