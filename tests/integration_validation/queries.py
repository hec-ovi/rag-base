"""Adversarial query set for Phase 4 integration validation.

Each query carries:
- text: the literal query string sent to /v1/search/*.
- target: the corpus key whose document should be retrieved.
- scenario: which scenario family the query belongs to.
- favors: which channel(s) we EXPECT to win on this query, before measurement.
          A channel can be "lexical", "semantic", "graph", "header_path", or
          "any" (all channels should handle it).
- note: short rationale.

The "favors" tag is a HYPOTHESIS, not a verdict. The Phase 4 report contrasts
the hypothesis against the observed ranks so we can flag surprises.
"""

QUERIES: list[dict] = [
    # ── POLYSEMY ────────────────────────────────────────────────
    {
        "text": "what programming language uses dynamic typing and an interpreter",
        "target": "POLY_LANG",
        "scenario": "polysemy",
        "favors": ["semantic"],
        "note": "Both POLY docs share the token 'Python', but only the language doc has the concept. Semantic should disambiguate.",
    },
    {
        "text": "constrictor reptiles native to africa and asia",
        "target": "POLY_SNAKE",
        "scenario": "polysemy",
        "favors": ["semantic", "lexical"],
        "note": "Distinguishing words 'constrictor' and 'reptiles' appear only in the snake doc.",
    },
    {
        "text": "Python",
        "target": None,
        "scenario": "polysemy",
        "favors": ["lexical"],
        "note": "Ambiguous bare token. Both polysemy docs match. Used to observe ranking, not a single target. Skipped from hit metrics.",
    },

    # ── PARAPHRASE ──────────────────────────────────────────────
    {
        "text": "tools for fast frontend development with quick reload",
        "target": "PARA_VITE",
        "scenario": "paraphrase",
        "favors": ["semantic"],
        "note": "Doc says 'native ESM' and 'single digit milliseconds'. Surface words do not overlap.",
    },
    {
        "text": "build system that uses native ECMAScript modules in the browser",
        "target": "PARA_VITE",
        "scenario": "paraphrase",
        "favors": ["lexical", "semantic"],
        "note": "Half-paraphrase: 'native ECMAScript modules' is verbatim, 'build system' is not.",
    },

    # ── EXACT PHRASE / RARE JARGON ──────────────────────────────
    {
        "text": "164.312(a)(2)(iv)",
        "target": "EXACT_HIPAA",
        "scenario": "exact_phrase",
        "favors": ["lexical"],
        "note": "Rare token. Embeddings often miss regulatory citations. BM25 should nail it.",
    },
    {
        "text": "encryption requirements for protected health information at rest",
        "target": "EXACT_HIPAA",
        "scenario": "exact_phrase",
        "favors": ["semantic", "lexical"],
        "note": "Natural language paraphrase of the same content, both channels should hit.",
    },

    # ── MULTI HOP (graph) ───────────────────────────────────────
    {
        "text": "Alice Chen",
        "target": "HOP_RESEARCHER",
        "scenario": "multi_hop",
        "favors": ["lexical", "semantic"],
        "note": "Direct lookup, name only. Retrieval is trivial; included as the entry point.",
    },
    {
        "text": "where is Acme Robotics headquartered",
        "target": "HOP_COMPANY",
        "scenario": "multi_hop",
        "favors": ["lexical", "semantic"],
        "note": "Single-doc question, no hops. Sanity check.",
    },
    {
        "text": "which researcher works at a company headquartered in Berkeley",
        "target": "HOP_RESEARCHER",
        "scenario": "multi_hop",
        "favors": ["graph"],
        "note": "Two-hop: researcher -> company -> city. No single doc carries the chain. Hybrid+graph should produce a multi-doc answer where the LightRAG channel surfaces HOP_RESEARCHER even though it never mentions Berkeley.",
    },

    # ── HEADER PATH ABLATION ────────────────────────────────────
    # These queries are designed so the answer can ONLY come from the
    # breadcrumb (the body has no language words). We expect HDR_WITH to be
    # findable for "Python" and HDR_WITHOUT to be NOT findable.
    {
        "text": "Python widget process synchronized stages",
        "target": "HDR_WITH",
        "scenario": "header_path",
        "favors": ["header_path"],
        "note": "The body has 'widget process synchronized stages' for both HDR_WITH and HDR_WITHOUT. Only HDR_WITH carries 'Python' in the breadcrumb of the relevant chunk.",
    },
    {
        "text": "Java widget process aligned segments",
        "target": "HDR_WITH",
        "scenario": "header_path",
        "favors": ["header_path"],
        "note": "Same idea, opposite section. The Java-section chunk of HDR_WITH should outrank any HDR_WITHOUT chunk.",
    },

    # ── DISTRACTOR PROBES (to make sure we are not always returning rank 1) ─
    {
        "text": "tallest volcano in the solar system",
        "target": "DIST_MARS",
        "scenario": "distractor",
        "favors": ["lexical", "semantic"],
        "note": "Sanity: distractor is findable, the system is not biased.",
    },
    {
        "text": "wild yeast leavened bread",
        "target": "DIST_COOKING",
        "scenario": "distractor",
        "favors": ["semantic"],
        "note": "Sanity: paraphrase still hits the cooking doc.",
    },
]


def queries_with_target() -> list[dict]:
    """Queries that have a concrete target_doc_id (used for hit/MRR metrics)."""
    return [q for q in QUERIES if q.get("target")]


def queries_for_scenario(scenario: str) -> list[dict]:
    return [q for q in QUERIES if q["scenario"] == scenario]
