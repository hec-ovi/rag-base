"""Adversarial corpus for Phase 4 integration validation.

Each doc is intentionally engineered to expose ONE specific channel's
behavior. The doc keys are stable so queries.py can reference them.

Scenarios:

- POLYSEMY:        "Python" the language vs "python" the snake.
                   Tests semantic disambiguation over a shared lexical token.
- PARAPHRASE:      query and doc share NO surface words.
                   Tests semantic over keyword.
- EXACT_PHRASE:    rare technical jargon (a regulatory citation).
                   Tests keyword over semantic.
- MULTI_HOP:       three docs forming a relation chain (researcher -> company -> city).
                   No single doc carries the full answer; graph should help.
- HEADER_PATH:     pair where the same body content lives under different markdown
                   headings vs no headings. Tests Phase 3c breadcrumb lift in isolation.
- DISTRACTOR:      noise docs unrelated to any target query.

The "filler" paragraphs in the header-path docs are deliberately worded to
avoid mentioning the disambiguating word (Python / Java / Setup) so the only
place that word appears for those docs is in the markdown header (and thus,
post-3c, in indexed_content via the breadcrumb).
"""

# Generic filler that does NOT contain "Python", "Java", "Setup", "Linux",
# "Windows", or any header word the test queries reference.
_GENERIC_FILLER_LONG = (
    "The widget process orchestrates synchronized stages reliably. "
    * 100
).strip()

_GENERIC_FILLER_SHORT_A = (
    "The blue gadget flows through aligned segments. " * 30
).strip()

_GENERIC_FILLER_SHORT_B = (
    "The red gadget flows through aligned segments. " * 30
).strip()


CORPUS: dict[str, dict] = {
    # ── POLYSEMY ────────────────────────────────────────────────
    "POLY_LANG": {
        "title": "Python: dynamic typing and the interpreter",
        "content": (
            "Python is a high level interpreted programming language designed by Guido "
            "van Rossum. It uses dynamic typing and reference counted garbage collection. "
            "The reference interpreter, CPython, parses source files into a stack based "
            "bytecode that runs on a virtual machine. Whitespace is significant: indentation "
            "delimits blocks instead of curly braces. Standard library batteries include "
            "modules for HTTP, JSON, sockets, and asynchronous IO. Common runtimes beyond "
            "CPython are PyPy (a tracing JIT) and MicroPython (microcontroller targets)."
        ),
        "metadata": {"scenario": "polysemy", "topic": "programming"},
    },
    "POLY_SNAKE": {
        "title": "Pythons: the constrictor snakes of Africa, Asia, and Australia",
        "content": (
            "Pythons are nonvenomous constricting reptiles native to tropical and subtropical "
            "regions of Africa, Asia, and Australia. They kill prey by coiling around it and "
            "tightening with each exhalation. Most species are oviparous; the female coils "
            "around the eggs and shivers to incubate them. Diet ranges from rodents to small "
            "ungulates depending on body size. The reticulated python is the longest extant "
            "snake. Pythons are distinguished from boas by skull morphology and the presence "
            "of premaxillary teeth."
        ),
        "metadata": {"scenario": "polysemy", "topic": "biology"},
    },

    # ── PARAPHRASE (semantic over keyword) ──────────────────────
    "PARA_VITE": {
        "title": "Vite: native ESM dev server",
        "content": (
            "Vite is a build tool that serves source modules as native ECMAScript modules "
            "to the browser during development. It avoids bundling at dev time, so file "
            "edits propagate to the running app in single digit milliseconds. Production "
            "builds use Rollup and tree shake aggressively. Plugins follow the Rollup "
            "convention so most ecosystem code transfers across. Vue, React, Svelte, and "
            "Solid all ship official Vite templates."
        ),
        "metadata": {"scenario": "paraphrase", "topic": "tooling"},
    },

    # ── EXACT PHRASE (keyword over semantic) ────────────────────
    "EXACT_HIPAA": {
        "title": "HIPAA Security Rule encryption requirements",
        "content": (
            "Under the HIPAA Security Rule §164.312(a)(2)(iv), covered entities must "
            "implement a mechanism to encrypt and decrypt electronic protected health "
            "information (ePHI) at rest. The implementation specification is addressable "
            "rather than required, meaning the entity must assess whether encryption is "
            "reasonable and appropriate; if not, an equivalent alternative measure must "
            "be documented. The Office for Civil Rights enforces this provision and may "
            "impose civil monetary penalties under 45 CFR Part 160 Subpart D."
        ),
        "metadata": {"scenario": "exact_phrase", "topic": "regulatory"},
    },

    # ── MULTI HOP CHAIN (graph wins) ────────────────────────────
    "HOP_RESEARCHER": {
        "title": "Alice Chen, senior researcher",
        "content": (
            "Alice Chen is a senior researcher specializing in computer vision at "
            "Acme Robotics. She joined the company in 2024 after completing her PhD "
            "at MIT. Her current work focuses on event-based vision sensors and their "
            "use in autonomous mobile platforms."
        ),
        "metadata": {"scenario": "multi_hop", "topic": "people"},
    },
    "HOP_COMPANY": {
        "title": "Acme Robotics company profile",
        "content": (
            "Acme Robotics is a robotics company founded in 2018. Its headquarters are "
            "located in Berkeley, California, with a satellite office in Tokyo. The "
            "company designs perception modules and motion planning stacks for autonomous "
            "ground vehicles. Acme has roughly 180 employees as of 2026."
        ),
        "metadata": {"scenario": "multi_hop", "topic": "company"},
    },
    "HOP_CITY": {
        "title": "Berkeley as an AI research hub",
        "content": (
            "Berkeley, California, is home to a dense cluster of AI research labs. "
            "The University of California, Berkeley hosts BAIR (the Berkeley AI Research "
            "lab), and several private labs and startups operate within walking distance "
            "of campus. The local talent pool draws on UC graduate programs in EECS and "
            "Statistics."
        ),
        "metadata": {"scenario": "multi_hop", "topic": "geography"},
    },

    # ── HEADER PATH ABLATION ────────────────────────────────────
    # WITH headers: the language name lives ONLY in the markdown headings.
    # The body filler is generic and contains no language word.
    "HDR_WITH": {
        "title": "Languages reference manual",
        "content": (
            "# Languages\n\n"
            "## Python\n\n"
            f"{_GENERIC_FILLER_LONG}\n\n"
            "## Java\n\n"
            f"{_GENERIC_FILLER_LONG}"
        ),
        "metadata": {"scenario": "header_path", "variant": "with_headers"},
    },
    # WITHOUT headers: same body filler, no markdown structure.
    # No chunk under this doc will contain "Python" or "Java" anywhere.
    "HDR_WITHOUT": {
        "title": "Languages reference manual",
        "content": (
            f"{_GENERIC_FILLER_LONG}\n\n"
            f"{_GENERIC_FILLER_LONG}"
        ),
        "metadata": {"scenario": "header_path", "variant": "no_headers"},
    },

    # ── DISTRACTORS ─────────────────────────────────────────────
    "DIST_COOKING": {
        "title": "Sourdough bread fundamentals",
        "content": (
            "Sourdough is leavened by a culture of wild yeasts and lactic acid bacteria "
            "rather than commercial yeast. The starter is a flour and water mix kept "
            "alive by daily feedings. Bulk fermentation, shaping, cold proof, and a hot "
            "dutch oven bake produce an open crumb and a crackling crust. Hydration "
            "between 70 and 85 percent is typical for country loaves."
        ),
        "metadata": {"scenario": "distractor", "topic": "cooking"},
    },
    "DIST_MARS": {
        "title": "Olympus Mons on Mars",
        "content": (
            "Olympus Mons is a shield volcano on the planet Mars and the tallest known "
            "volcano in the Solar System. Its peak rises about 22 kilometers above the "
            "Martian datum, and the base spans roughly 600 kilometers. The shape is the "
            "result of long-lived effusive eruptions on a stationary mantle plume."
        ),
        "metadata": {"scenario": "distractor", "topic": "astronomy"},
    },
}


def get_doc(key: str) -> dict:
    """Fetch a doc spec by stable key."""
    return CORPUS[key]


def all_docs(exclude: set[str] | None = None) -> list[tuple[str, dict]]:
    """Iterate (key, spec) pairs, optionally skipping some keys."""
    skip = exclude or set()
    return [(k, v) for k, v in CORPUS.items() if k not in skip]
