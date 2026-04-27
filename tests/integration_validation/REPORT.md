# Phase 4: Integration validation findings

**Date:** 2026-04-27
**Engine:** rag-base at commit on `main`, post-Phase 3c.
**LLM:** Qwen3.6-27B-AWQ4 via vLLM, `enable_thinking=false` (Phase 4.0).
**Stack:** ParadeDB pg_search 0.23.1 (BM25), TEI BGE-M3 1024d (semantic, CPU), TEI BGE-reranker-v2-m3 (rerank, CPU), Memgraph MAGE 3.9 (graph), custom FastAPI on 5050.
**Corpus:** 11 adversarial docs (`tests/integration_validation/corpus.py`), 13 tagged queries (`queries.py`).

The corpus was authored by the LLM (me) to expose individual channels' behavior:
polysemy, paraphrase, exact-phrase rare jargon, three-doc multi-hop relation chain, header-path ablation pair, distractors. Each query is tagged with the channel it's _expected_ to favor; this report contrasts the hypothesis against measured ranks.

---

## Headline numbers (per channel, all 13 tagged queries)

| Channel | hit@1 | hit@5 | MRR | Latency p50 | Notes |
|---|---|---|---|---|---|
| `lexical` (BM25) | 0.92 | 1.00 | 0.962 | ~3 ms | one miss: paraphrase query "tools for fast frontend development" -> rank 2 |
| `semantic` (BGE-M3) | 0.92 | 1.00 | 0.949 | ~35 ms | one miss: multi-hop "researcher in Berkeley" -> rank 3 |
| `hybrid_norerank` (RRF) | 0.92 | 1.00 | 0.962 | ~40 ms | same multi-hop miss -> rank 2 (RRF promoted) |
| `hybrid_rerank` | 0.92 | 1.00 | 0.962 | **~17-29 s** | same multi-hop miss -> rank 2; latency dominated by reranker CPU inference |
| `hybrid_graph_rerank` | 0.92 | 1.00 | 0.942 | ~17-23 s | the multi-hop "researcher in Berkeley" query went to rank 4 (worse than hybrid_rerank's 2); see analysis below |

**Caveat: at this corpus size every channel saturates on hit@5.** With 11 docs and queries that point at unique-enough vocabulary, BM25 alone already finds the right doc inside top 5 for every tagged query. The discriminative signal is in `hit@1` and the per-query `rank`, not in headline metrics. The corpus needs to grow 10-100x before headline metrics separate the channels meaningfully.

---

## Per-scenario findings

### Polysemy ("Python" the language vs "python" the snake)

| Query | Lexical | Semantic | Hybrid | Hybrid+Rerank |
|---|---|---|---|---|
| "what programming language uses dynamic typing and an interpreter" -> POLY_LANG | 1 | 1 | 1 | 1 |
| "constrictor reptiles native to africa and asia" -> POLY_SNAKE | 1 | 1 | 1 | 1 |

All channels disambiguate cleanly. The query side carries enough non-Python tokens (`dynamic typing`, `interpreter`, `constrictor`, `reptiles`) that BM25 alone can pick the right doc; semantic has no extra job to do here. **The polysemy scenario does NOT separate channels at this corpus size.** A version where the query itself was just `Python` would test this better, but then there is no single ground-truth doc (both are valid).

### Paraphrase ("fast frontend dev" -> Vite doc with no surface overlap)

| Query | Lexical | Semantic | Hybrid | Hybrid+Rerank |
|---|---|---|---|---|
| "tools for fast frontend development with quick reload" -> PARA_VITE | **2** | 1 | 1 | 1 |
| "build system that uses native ECMAScript modules in the browser" -> PARA_VITE | 1 | 1 | 1 | 1 |

The first query is the only one in this corpus where lexical does NOT get rank 1 — semantic is required because the query and doc share zero surface tokens (`fast`/`rapid`, `quick reload`/`single digit milliseconds`). RRF and rerank both correctly absorb the semantic vote. **This is the cleanest evidence in the corpus that semantic earns its keep.**

### Exact-phrase rare jargon (HIPAA citation `164.312(a)(2)(iv)`)

| Query | Lexical | Semantic | Hybrid | Hybrid+Rerank |
|---|---|---|---|---|
| "164.312(a)(2)(iv)" -> EXACT_HIPAA | 1 | 1 | 1 | 1 |
| "encryption requirements for protected health information at rest" -> EXACT_HIPAA | 1 | 1 | 1 | 1 |

I expected semantic to fumble the literal citation token (BGE-M3 embeddings often blur regulatory codes). Instead it was rank 1. Reason: with only 11 docs in the index, "164.312" is the only chunk whose embedding has anything related to numbers + parenthetical structure, so cosine similarity still wins by elimination. **Bigger corpus would likely flip this.** This is the single biggest "the hypothesis is wrong at this corpus size" datapoint in the run.

### Multi-hop chain (researcher -> company -> city)

| Query | Lexical | Semantic | Hybrid | Hybrid+Rerank | Hybrid+Graph+Rerank |
|---|---|---|---|---|---|
| "Alice Chen" -> HOP_RESEARCHER | 1 | 1 | 1 | 1 | 1 |
| "where is Acme Robotics headquartered" -> HOP_COMPANY | 1 | 1 | 1 | 1 | 1 |
| **"which researcher works at a company headquartered in Berkeley" -> HOP_RESEARCHER** | **1** | **3** | **2** | **2** | **4** |

The interesting query is the third. **Surprising finding 1: lexical wins on the original phrasing.** The query contains the word `researcher` which appears literally only in `HOP_RESEARCHER`'s title and content; BM25 nails it on a single rare token. Semantic distributes weight across all three multi-hop docs (they all share concepts like "company / researcher / Berkeley") and ranks the answer at 3. RRF brings it back to 2 by averaging.

**Surprising finding 2: the graph channel made the multi-hop result WORSE on this query** (rank 4 vs rerank's 2). Decoding the top hits returned by `hybrid_graph_rerank`:

```
1. doc_id=110  Acme Robotics company profile           sources=['semantic','keyword']
2. doc_id=117  Acme Robotics company profile (graph)   sources=['semantic','keyword']
3. doc_id=116  Alice Chen, senior researcher (graph)   sources=['semantic','keyword','graph']
4. doc_id=109  Alice Chen, senior researcher           <- target
```

Two confounds:

(a) **The corpus carries duplicate variants** of each multi-hop doc (the "_GRAPH" variants from `ingest-graph`). With duplicates in the index the reranker spreads relevance across both copies, dropping the original below them.

(b) **The query phrase "company headquartered in Berkeley" matches HOP_COMPANY's content verbatim**, so the cross-encoder reranker scores the company docs above the researcher doc. The reranker is content-relevance based, not graph-aware: it does not know that the query's intent is to identify a person whose company satisfies a property.

To isolate the graph contribution from those confounds, I ran three cleaner ad-hoc queries against `/v1/search?include_graph=true`. Top-5 hits (with the `sources` field, which records which channels surfaced each chunk):

**Q "specializes in computer vision in the bay area":**
- Rank 1: HOP_RESEARCHER (no-graph), `sources=['semantic','keyword']`
- Rank 2: HOP_RESEARCHER_GRAPH, `sources=['semantic','keyword','graph']`
- Rank 3-5: Berkeley + Acme variants

**The graph channel fires** (the `graph` source tag is visible on rank 2). With "computer vision" appearing only in the researcher doc, semantic and keyword already pin rank 1 correctly; the graph contribution is observable but redundant.

**Q "find the AI researcher whose firm sits in northern California":**
- Rank 1: Berkeley (no-graph), `sources=['semantic','keyword']`
- Rank 2: Berkeley_GRAPH, `sources=['semantic','keyword']`
- Rank 5: HOP_RESEARCHER (the actual answer)

This is the **real multi-hop failure mode** and the most honest data point in the run. The query has zero token overlap with HOP_RESEARCHER ("AI", "researcher", "firm", "northern California" — none appear in Alice Chen's doc). Berkeley's content contains "AI research hub" and "Berkeley, California" verbatim. So the semantic + keyword channels hand the reranker a candidate set where Berkeley out-features Alice on every surface signal. The graph channel surfaced Berkeley correctly via the Acme -> Berkeley edge, but **the reranker arbitrates by content match, not by graph centrality**, so Berkeley wins.

**The graph channel cannot fix a reranker that scores literal relevance.** To answer "find the AI researcher" the system would need either (a) a graph-traversal post-processing step that promotes nodes one hop away from the strongest match (Acme/Berkeley -> Alice), or (b) an LLM synthesis step that reads the top docs and answers the question directly using the graph as scaffolding. Neither is a defect in any single Phase 3 component; it's a known limitation of "graph adds candidates, rerank picks one" architectures. This is exactly the gap that the parked Phase 5 (`POST /v1/search/graph` fast-mode endpoint, design at `.research/graph-only-retrieval-no-embedding-fast-mode/FINDINGS.md`) is meant to address: a structured `{matched_entities, subgraph, chunks}` response that lets the caller see the chain instead of one re-scored chunk.

**Graph value at this corpus size: marginal but observable.** With 11 docs every channel sees every candidate, so the graph adds tagging more than recall. In a 10k-doc corpus the graph channel's value would be bringing the 3 chained docs INTO the rerank's top-50 candidate set when they would not have made it via vector/keyword alone. That value is invisible here because scarcity is invisible here.

### Header-path ablation (Phase 3c isolated)

The same body content was ingested twice: once with markdown headings (`# Languages`, `## Python`, `## Java`), once with the headings stripped. The body filler is identical and contains no language word. Every chunk under HDR_WITHOUT therefore has zero occurrence of "Python" or "Java" anywhere in its text. Phase 3c stores the breadcrumb in `chunks.indexed_content` for HDR_WITH, so the query word lives only in the breadcrumb prefix.

| Query | Rank of HDR_WITH | Rank of HDR_WITHOUT |
|---|---|---|
| "Python widget process synchronized stages" | **1** | 2 |
| "Java widget process aligned segments" | **1** | 2 |

HDR_WITH outranks HDR_WITHOUT in both cases. The breadcrumb is the **only** signal that distinguishes them; the body content is identical. The lift is real but modest (one rank position) because BM25 still matches HDR_WITHOUT via the shared filler tokens. **In a production setting where dozens of unrelated docs share filler-class language, the lift would be larger** because HDR_WITHOUT would be drowned in noise while HDR_WITH still carries the breadcrumb signal.

This is the cleanest end-to-end demonstration that Phase 3c contributes measurable retrieval value, not just storage symmetry.

### Distractors (sanity checks)

Both distractor queries (volcano, sourdough) hit the right doc at rank 1 across all channels. The system is not biased toward any one topic.

---

## Latency observations

- **Lexical (BM25)** is sub-5 ms per query. ParadeDB pg_search delivers as advertised.
- **Semantic** is ~25-90 ms, dominated by the BGE-M3 forward pass on CPU TEI.
- **Hybrid no-rerank** is ~30-70 ms (semantic + BM25 + RRF fuse, no extra cost beyond the slowest channel).
- **Hybrid + rerank** is ~15-30 s. **This is much slower than the Phase 3 baseline of 1.2 s.** Cause: this corpus has longer chunks (HDR_WITH carries 600+600 word filler producing 3-4 chunks of ~512 tokens each), and the cross-encoder reranker on CPU TEI scales roughly linearly with `(query, chunk)` pairs and chunk length. TEI logs show `inference_time=3-5 s` per call plus `queue_time=3-12 s` of accumulated queueing under sequential load. Re-confirms the Phase 3b deferral logic: the BGE-reranker-v2-m3 on CPU is the latency-critical component, and a swap to a vLLM-served Gemma reranker on GPU would address this.

---

## Phase 4.0 (vLLM thinking-off): measured impact

**Smoke prompt**, "Reply with one word: yes.":
- Before (`reasoning.effort=minimal`): 684 output tokens, 33.7 s wall.
- After (`chat_template_kwargs.enable_thinking=false`): 3 output tokens, 0.7 s wall.
- **~50x latency reduction on short prompts.**

**End-to-end on the LightRAG ingest path (3 multi-hop docs):**
- HOP_RESEARCHER: 86.3 s
- HOP_COMPANY: 87.9 s
- HOP_CITY: 93.6 s
- Total: 4 min 28 s for 3 docs.
- Phase 2 baseline (same docs would have run with reasoning=minimal): ~5-10 min per doc, so 15-30 min total.
- **~5-7x faster on the realistic entity-extraction workload.**

The rule applies to BOTH ingest-time and query-time LLM calls. The same `make_llm_complete` closure is registered as LightRAG's `llm_model_func` and is also called directly by the hybrid graph channel's query-time NER (`extract_query_entities` in `lightrag_store.py`). Memory file `feedback_vllm_streaming_only.md` documents this with the wiring diagram.

---

## What does NOT make sense (questions for future investigation)

1. **Why does `semantic` rank `EXACT_HIPAA` at 1 for the literal citation `164.312(a)(2)(iv)`?** I expected BGE-M3 to blur this regulatory token; it should have been rank > 1. The likely answer is corpus-size luck (11 docs is not enough non-numeric distractors). Re-verify when the corpus grows.

2. **Why does `lexical` win on the Berkeley multi-hop query?** Because the query leaks the discriminative token `researcher`. The cleaner queries in the multi-hop section (no token leakage) confirm that without that leak, lexical fails too. The query was a worse multi-hop test than I designed it to be.

3. **Why is hybrid_rerank latency 14x worse than the Phase 3 smoke baseline?** Confirmed via TEI logs: reranker queue + inference time grow with chunk length, and HDR_WITH chunks are larger than the smoke-set average. Not a bug, but a usage constraint that should be noted in the README.

4. **Why doesn't the graph channel rescue the "AI researcher in northern California" query?** Because LightRAG's role is to surface candidates, not to arbitrate the answer. The reranker scores by content match. Even though Acme -> Berkeley -> Alice is a valid graph path, no component in the current pipeline traverses it as a graph; chunks are scored independently. This motivates the planned Phase 5 graph-only endpoint that returns structured `{entities, subgraph, chunks}` instead of one re-scored chunk.

---

## Architecture verification (rag-base vs knowledge-base)

The user asked whether knowledge-base sits OUTSIDE rag-base and whether the Karpathy LLM-Wiki + MemPalace insights belong above the engine. Cross-checking against the macro view memory + the `mempalace-karpathy-knowledge` research entry:

- `macro_view.md` line 19: "knowledge-base ... calls rag-base via REST. No retrieval mechanics in this repo."
- `mempalace-karpathy-knowledge/FINDINGS.md` lines 84-94 (the synthesis): "keep the write-time compilation Karpathy proposes, keep the spatial graph + supersession edge MemPalace proposes, but do use vectors and a reranker underneath because the wiki index will exceed context window for any non-toy corpus."

**Confirmed.** The right split is:

- `rag-base` (this repo): generic SOTA retrieval substrate (vectors + BM25 + LightRAG graph + rerank). No domain coupling. No write-time LLM compilation. Tested in isolation — that is what this Phase 4 report does.
- `knowledge-base` (sibling repo): write-time compilation layer. Anthropic Contextual Retrieval blurbs at ingest, episodic supersession edges (`valid_from`/`valid_to`), Karpathy `raw/` -> `wiki/` -> `index.md` compile pass at ingest, MemPalace-style contradiction-on-write graph updates, MCP tool surface for IDE agents. Calls rag-base over REST.

Phase 5 (graph-only `/v1/search/graph` endpoint) sits on the rag-base side because it is a retrieval shape, not a write-time concern. Phase 6 (knowledge-base wrapper) is everything above.

---

## Reproducing this report

```
# rag-base/ root, .env exported
.venv/bin/python tests/integration_validation/run.py ingest
.venv/bin/python tests/integration_validation/run.py step lexical
.venv/bin/python tests/integration_validation/run.py step semantic
.venv/bin/python tests/integration_validation/run.py step hybrid_norerank
.venv/bin/python tests/integration_validation/run.py step hybrid_rerank
.venv/bin/python tests/integration_validation/run.py step header_ablation

# Slow path (LightRAG entity extraction, ~3-5 min total with thinking off):
.venv/bin/python tests/integration_validation/run.py ingest-graph
.venv/bin/python tests/integration_validation/run.py step hybrid_graph_rerank

# Cleanup:
.venv/bin/python tests/integration_validation/run.py cleanup
```

JSONL rows live in `tests/integration_validation/results/step_*.jsonl`. Each step truncates its file on rerun, so the latest invocation is the canonical record.
