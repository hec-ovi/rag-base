"""Unit tests for the prompt-template loader (api/src/services/prompts.py).

The loader is intentionally tiny but it owns three contracts that are easy to
break by accident:

- {{name}} placeholders are substituted via str.replace, NOT str.format. So
  prompt bodies and substituted values can both contain stray `{` / `}` without
  crashing (e.g. when a document includes JSON or code).
- A leading `<!-- ... -->` comment block is stripped before the body reaches the
  LLM, so prompt files can carry inline editor docs without polluting the model.
- Missing template names raise FileNotFoundError (hard fail on typos at startup).

Tests run against the real prompt files shipped in api/prompts/ rather than
synthetic fixtures, so a future broken edit of those files will surface here.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "api"))

from src.services import prompts  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_cache():
    """Each test starts with an empty cache so order does not matter."""
    prompts._cache.clear()
    yield
    prompts._cache.clear()


def test_load_strips_header_comment_block():
    body = prompts.load("contextual_retrieval")
    # Both shipped prompts begin with <!-- ... -->; loader must strip it.
    assert "<!--" not in body
    assert "-->" not in body
    # Body still contains its substantive content.
    assert "Please give a short succinct context" in body


def test_load_caches_by_name():
    a = prompts.load("contextual_retrieval")
    b = prompts.load("contextual_retrieval")
    # Returned object identity is the cache signal.
    assert a is b


def test_load_missing_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        prompts.load("this_template_definitely_does_not_exist_xyz")


def test_render_substitutes_placeholders():
    out = prompts.render(
        "contextual_retrieval",
        document="DOC_TEXT_HERE",
        chunk="CHUNK_TEXT_HERE",
    )
    assert "DOC_TEXT_HERE" in out
    assert "CHUNK_TEXT_HERE" in out
    # Unsubstituted placeholders should not remain.
    assert "{{document}}" not in out
    assert "{{chunk}}" not in out


def test_render_handles_curly_braces_in_values():
    """str.format would crash on this; str.replace is safe."""
    out = prompts.render(
        "contextual_retrieval",
        document="payload: {x: 1, y: 2}",
        chunk="snippet: format string {0}",
    )
    assert "{x: 1, y: 2}" in out
    assert "{0}" in out


def test_render_does_not_recurse_through_substituted_values():
    """A value that contains `{{otherkey}}` should appear literally, not be
    re-expanded. Keeps prompt content from accidentally injecting new vars
    based on user input."""
    out = prompts.render(
        "contextual_retrieval",
        document="this contains {{chunk}} literally",
        chunk="REAL_CHUNK",
    )
    # The literal "{{chunk}}" inside `document` should remain (no second pass).
    assert "this contains {{chunk}} literally" in out
    # The actual chunk placeholder still got substituted with REAL_CHUNK once.
    # We verify by counting how many times REAL_CHUNK appears: should be exactly 1
    # (the genuine `{{chunk}}` slot, not a recursion through `document`).
    assert out.count("REAL_CHUNK") == 1


def test_query_entity_extraction_template_loads_and_substitutes():
    out = prompts.render("query_entity_extraction", query="who is Alice Chen?")
    assert "who is Alice Chen?" in out
    assert "{{query}}" not in out
    assert "<!--" not in out
    assert "JSON list" in out  # core instruction survives


def test_render_with_no_kwargs_returns_template_unchanged():
    """No substitution requested = template body returned with all placeholders intact."""
    body = prompts.load("contextual_retrieval")
    out = prompts.render("contextual_retrieval")
    assert out == body
