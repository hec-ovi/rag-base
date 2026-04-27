"""Shared pytest fixtures for rag-base tests.

Phase 0: skeleton only. Real fixtures (Postgres, TEI, Memgraph clients) land in Phase 1
when the integration tests need them.
"""

import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "tests" / "golden"
FIXTURES_DIR = GOLDEN_DIR / "fixtures"


@pytest.fixture
def repo_root() -> pathlib.Path:
    return REPO_ROOT


@pytest.fixture
def golden_dir() -> pathlib.Path:
    return GOLDEN_DIR


@pytest.fixture
def fixtures_dir() -> pathlib.Path:
    return FIXTURES_DIR
