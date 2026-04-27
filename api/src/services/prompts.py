"""Tiny prompt-template loader.

Prompts live in `api/prompts/*.md`, decoupled from Python code so they can be
edited without touching the codebase. Loaded once at first access (cached for
the process lifetime); to reload after editing, restart the api container.

Each template file may begin with an HTML comment block (`<!-- ... -->`) used
as inline documentation for editors. The loader strips that header before the
template reaches the LLM, so the comment is invisible to the model.

Substitution: `{{name}}` placeholders are replaced in a SINGLE regex pass so
substituted values are not re-scanned for further placeholders. This keeps a
document that contains literal `{{chunk}}` text from accidentally triggering
substitution into the rendered prompt (defense-in-depth against prompt
injection via user-supplied content). We also avoid `str.format` so prompt
bodies and substituted values can both contain stray `{` / `}` (e.g. JSON or
code in a document) without crashing.
"""

import re
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

# Greedy DOTALL match strips a leading <!-- ... --> comment plus whitespace.
_HEADER_COMMENT_RE = re.compile(r"\A\s*<!--.*?-->\s*", re.DOTALL)

# Placeholder syntax: {{name}} where name is a python identifier.
_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")

_cache: dict[str, str] = {}


def _strip_header_comment(text: str) -> str:
    """Remove a leading HTML comment header if present. Idempotent."""
    return _HEADER_COMMENT_RE.sub("", text, count=1)


def load(name: str) -> str:
    """Load `api/prompts/<name>.md`, cache, return the body without the header comment.

    Raises FileNotFoundError if the template doesn't exist (let it surface so
    typos are caught at startup, not at the first request).
    """
    if name in _cache:
        return _cache[name]
    path = _PROMPTS_DIR / f"{name}.md"
    raw = path.read_text(encoding="utf-8")
    body = _strip_header_comment(raw).strip()
    _cache[name] = body
    return body


def render(name: str, **vars: str) -> str:
    """Load template `name` and substitute `{{key}}` -> str(value) for each kwarg.

    One regex pass, no recursion: a substituted value that itself contains
    `{{otherkey}}` is NOT re-scanned. Unknown placeholders (no matching kwarg)
    are left untouched in the output.
    """
    text = load(name)

    def _sub(match: re.Match) -> str:
        key = match.group(1)
        if key in vars:
            return str(vars[key])
        # Unknown placeholder: leave it intact so it's visible in the rendered output
        # rather than silently replaced with empty string.
        return match.group(0)

    return _PLACEHOLDER_RE.sub(_sub, text)
