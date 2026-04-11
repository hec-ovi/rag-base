"""Tests for Pydantic model validation."""

import pytest
from pydantic import ValidationError

from api.src.models.document import DocumentCreate
from api.src.models.search import SearchRequest, EmbedRequest, RerankRequest
from api.src.models.concept import ConceptCreate
from api.src.models.relation import RelationCreate


def test_document_create_valid():
    doc = DocumentCreate(title="Test", content="Some content")
    assert doc.title == "Test"
    assert doc.metadata == {}


def test_document_create_empty_title():
    with pytest.raises(ValidationError):
        DocumentCreate(title="", content="Some content")


def test_document_create_empty_content():
    with pytest.raises(ValidationError):
        DocumentCreate(title="Test", content="")


def test_search_request_defaults():
    req = SearchRequest(query="test query")
    assert req.top_k == 20
    assert req.rerank is True
    assert req.include_graph is True
    assert req.min_score == 0.0


def test_search_request_top_k_bounds():
    with pytest.raises(ValidationError):
        SearchRequest(query="test", top_k=0)
    with pytest.raises(ValidationError):
        SearchRequest(query="test", top_k=101)


def test_embed_request_valid():
    req = EmbedRequest(inputs=["hello", "world"])
    assert len(req.inputs) == 2


def test_rerank_request_valid():
    req = RerankRequest(query="test", texts=["a", "b"])
    assert req.return_text is False


def test_concept_create_defaults():
    c = ConceptCreate(name="React")
    assert c.type == "Entity"
    assert c.description == ""


def test_relation_create_valid():
    r = RelationCreate(source_name="React", target_name="JavaScript", relation_type="DEPENDS_ON")
    assert r.relation_type == "DEPENDS_ON"


def test_relation_create_empty_type():
    with pytest.raises(ValidationError):
        RelationCreate(source_name="A", target_name="B", relation_type="")
