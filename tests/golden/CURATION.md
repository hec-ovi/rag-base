# Golden set curation plan

The golden set is the load-bearing artifact for rag-base SOTA work. Every phase gate from Phase 2 onward (NDCG@10, recall@K, faithfulness) is measured against it. Treat this file as a contract: changes to schema or slice allocation should land via PR with the same scrutiny as production code.

This is the **rag-base** retrieval-quality golden set; application-level slices (user-fact supersession, ingestor diversity) live in the knowledge-base wrapper's own golden set.

## File layout

```
tests/golden/
  CURATION.md                 (this file)
  golden_set.jsonl            (the dataset; one record per line)
  fixtures/                   (frozen source documents the queries reference)
    doc_<sha256>.json         (raw document snapshot at curation time)
  slices/                     (slice membership lists; query ids only)
    polysemous.txt
    identifier.txt
    contextual_disambiguation.txt
    multi_hop.txt
    auto_merge.txt
    long_context.txt
    time_travel.txt
    negative.txt
  eval_history.jsonl          (per-run metrics; append-only; one row per CI run)
  schema_version.txt          (currently 1)
```

`fixtures/` matters: source documents change as the user re-ingests or edits. We snapshot the exact bytes the golden record was labeled against so the eval is reproducible across the lifetime of the corpus. Tests load fixtures, not live DB rows, when computing expected_chunk_ids.

## Scope

This is the **rag-base** retrieval-quality golden set. It tests the engine: chunking, embedding, BM25, semantic, RRF, rerank, LightRAG dual-level, contextual retrieval, parent-child auto-merge, and the bitemporal `valid_at` filter. It does NOT cover application-level concerns (user-fact supersession decision logic, ingestor diversity, annotation context preservation), those live in the knowledge-base wrapper's own golden set.

## Target schema assumption

Records are labeled against the **post-Phase-1+ rag-base schema**:

- `documents (id, title, content, metadata, valid_from, valid_to, ...)` with bitemporal columns activated at retrieval via the `valid_at` query parameter
- `chunks (id, document_id, parent_id, chunk_index, content, indexed_content, token_count, embedding vector(1024), valid_from, valid_to, ...)` where `indexed_content` is the augmented `[title | meta] [header > path] <chunk>` form fed to embedding + BM25 (Phase 1 adds the table; Phase 3a moves keyword from tsvector + ts_rank to ParadeDB pg_search BM25 indexed on `indexed_content`; Phase 3c adds the header-path prefix; Phase 6 adds `parent_id`)
- LightRAG entity + relation collections in Memgraph (Phase 3)
- LightRAG vector collections in pgvector (Phase 3, distinct from `chunks.embedding`)

Curation does not begin until Phase 1 has the chunks table populated. The "best quality" choice (per project preference) is to label against the final unified schema rather than dump from earlier states and re-label later.

## Record schema (golden_set.jsonl)

Each line is one JSON object. Required fields are in **bold**.

| Field | Type | Notes |
|---|---|---|
| **`id`** | string | `gs_0001` style, monotonic, never reused |
| **`query`** | string | The user-style question / search input |
| **`intent`** | string | One of: `factual`, `definitional`, `comparative`, `procedural`, `multi_hop`, `time_travel`, `negative` |
| **`slices`** | string[] | Subset of slice tags listed below; can be empty if a "general" record |
| **`expected_doc_ids`** | string[] | sha256 of the fixture document(s) the answer must come from |
| **`expected_chunk_predicates`** | object[] | List of predicates that the chunks must satisfy (see below). Used because chunk_ids change when chunker config changes |
| `expected_answer` | string | Optional. Free-text reference answer for E2E faithfulness eval. Skip for retrieval-only records |
| `forbidden_doc_ids` | string[] | Optional. Documents that must NOT appear in top-K (used for negative slice) |
| `time_scope` | object | Optional. `{valid_at: "2026-04-26T..."}` for bitemporal time-travel queries (engine-level test of the `valid_at` filter) |
| `expected_via` | string[] | Optional. Subset of `["semantic", "bm25", "graph"]`. If present, the record fails unless EVERY listed channel contributed to at least one hit (read from the response's `sources` array per chunk). This is what makes a slice prove the SOTA mechanism fires; without it, a `multi_hop` record cannot distinguish "LightRAG worked" from "semantic happened to catch it". |
| `expected_blurb_visible` | bool | Optional. If true, the retrieved chunk's stored content must begin with a non-empty `contextual_blurb`. Validates Phase 4 contextual retrieval is end-to-end wired, not just sitting in a column. |
| `expected_parent_merged` | bool | Optional. If true, the response must contain the parent chunk's text rather than (or in addition to) the leaf chunk. Validates Phase 6 auto-merge fired. |
| `expected_entity_path` | string[] | Optional. For multi_hop. Format: `[entity, relation, entity, relation, ...]`. e.g. `["React", "DEPENDS_ON", "JavaScript", "USED_BY", "Node.js"]`. Validates LightRAG actually traversed the graph rather than the answer being inferable from a single chunk. |
| **`source`** | string | `synthetic_llm` / `hand_crafted` (rag-base is stateless, no real query log of its own; if/when wrappers feed back, add `wrapper_query_log`) |
| **`labeled_by`** | string | "hector" or other curator handle |
| **`labeled_at`** | string | ISO date |
| **`schema_version`** | int | Bumped when fields change; always equals top-of-file `schema_version.txt` |
| `notes` | string | Optional free-text rationale; encouraged for stress tests |

### Why `expected_chunk_predicates` instead of `expected_chunk_ids`

Chunk IDs are unstable: changing chunk size from 512 to 256 in Phase 6 would invalidate every record. A predicate is robust:

```json
{"predicate": "contains_all", "tokens": ["pgvector", "HNSW", "ef_construction"]}
{"predicate": "matches_regex", "pattern": "Snowflake/snowflake-arctic-embed-l-v2.0"}
{"predicate": "in_doc_with_title_substr", "value": "rag-base technical reference"}
```

Eval compares retrieved chunks against predicates, not opaque IDs. We accept any chunk that satisfies at least one of the listed predicates as a "hit" for that query.

## Slices and target counts

Initial target is 160 records, growing to 280 by Phase 6. Allocation. Each stress slice MUST set the relevant `expected_via` / `expected_blurb_visible` / `expected_parent_merged` / `expected_entity_path` fields so the test proves the SOTA mechanism fires (per the phase-gate rule, not just a "looked done" check).

| Slice | Initial count | What it tests | Required assertion fields |
|---|---|---|---|
| `factual` (general) | 50 | Plain "what is X" / "when did Y happen" lookups | none required (baseline) |
| `polysemous` | 15 | Same word, multiple meanings (e.g. "Python" lang vs snake) | `forbidden_doc_ids` (the wrong-sense doc) |
| `identifier` | 15 | Proper nouns, code symbols, product versions ("React.useEffect", "BGE-M3 1024d") | `expected_via: ["bm25"]` |
| `contextual_disambiguation` | 15 | Chunk text alone is ambiguous ("the first one"), only the contextual blurb resolves it | `expected_blurb_visible: true`, `expected_via: ["semantic"]` |
| `multi_hop` | 15 | Requires entity expansion across docs | `expected_via: ["graph"]`, `expected_entity_path: [...]` |
| `auto_merge` | 10 | Answer needs synthesis across 2+ leaf chunks of the same parent | `expected_parent_merged: true` |
| `long_context` | 15 | Answer requires combining content separated by >2000 tokens in source | overlaps with `auto_merge`; some records carry both slices |
| `time_travel` | 10 | Bitemporal `valid_at` filter actually changes results | `time_scope: {valid_at: ...}` set; pair with non-time-scoped variant where `expected_doc_ids` differs |
| `negative` | 15 | Queries that should return nothing | `expected_doc_ids: []`, `forbidden_doc_ids` if a near-miss exists |

Total initial: 160. Stress slices over-weighted (~69%) relative to "general" because these are the failure modes that distinguish SOTA from baseline. Application-level slices (`supersession` of user facts, `cross_source` of ingestor diversity) live in the knowledge-base golden set, not here, because the supersession decision logic and ingestor pipeline are knowledge-base concerns. Here we only test that the engine's `valid_at` filter works correctly when called.

## Quality bars (a record either meets all or is rejected)

1. **Reproducible**: pasting the query into the system today must yield the same expected chunks (allowing for chunker config changes covered by predicates).
2. **Discriminating**: the record must distinguish at least two of the retrieval channels. A query where semantic + BM25 + LightRAG all return the same top-10 in the same order teaches us nothing about the architecture; it goes in `factual` baseline only, not in any stress slice.
3. **Honest answer**: `expected_answer` (when present) must be supportable by the `expected_doc_ids` content alone. If it requires outside knowledge, the record is invalid.
4. **Predicate sufficiency**: the predicates must uniquely identify the target chunk(s) within the fixture document. If a predicate matches 5 chunks but only 1 is actually relevant, tighten the predicate.
5. **Time-travel pairs are atomic**: a `time_travel` record is curated as a pair sharing an id prefix (`gs_tt_001a`, `gs_tt_001b`). The "a" record queries without `time_scope` and returns the current doc; the "b" record queries the same string with `time_scope.valid_at` in the past and must return a different `expected_doc_ids`. Never commit one without the other.

## Curation pipeline

rag-base is stateless and has no real query log of its own; the corpus comes from synthetic generation grounded in fixture documents, plus hand-crafting for slices where synthesis is unsafe.

### Source 1: synthetic LLM-generated, grounded in real fixtures

For every slice except `time_travel` and `negative`:

1. Curate a small library of fixture documents under `fixtures/` (technical READMEs, language docs, library reference pages, polysemous-term docs). Domain-agnostic, chosen to exercise specific failure modes per slice.
2. Run Phase 1+ ingest on the fixtures so chunks, contextual blurbs, header paths, LightRAG entities/relations all exist in the rag-base test environment.
3. For each slice, prompt Claude Sonnet 4.6 with: slice definition, 3 hand-crafted exemplars from `_examples.jsonl`, the relevant fixture documents, AND the LightRAG graph dump from the test ingest (so `multi_hop` drafts cite real graph edges, not hallucinated ones).
4. LLM proposes 30 candidate queries per slice with rationale + predicate suggestions + assertion-field suggestions (e.g. for `multi_hop`, the candidate `expected_entity_path`).
5. Human (Hector) filters to the target count per slice via `tests/golden/curate.py`. Target throughput: ~30 records/hour after warmup. Reject anything that does not meet the discriminating bar.

Drafting model: Claude Sonnet 4.6 (best quality default; one-time cost is trivial against dataset lifetime).

The grounding rule for `multi_hop` is load-bearing: synthetic queries against hallucinated entities pass curation but fail at eval, producing phantom records. Always cite real edges from the graph dump.

### Source 2: hand-crafted (mandatory for `time_travel` and `negative`)

`time_travel` records are pairs (see Quality bars #5). Hand-crafted because the pair's correctness depends on careful fixture construction: ingest doc v1 with `valid_to` set, then doc v2 with `valid_from` matching, then verify both queries (with and without `valid_at`) return the right doc.

`negative` records are queries that should return nothing relevant. Hand-crafted because synthetic generation tends to invent plausibly-relevant queries; the whole point of this slice is hard-irrelevant.

Initial target: 10 time_travel pairs (20 records) + 15 negative records.

## Versioning + rotation policy

- `schema_version.txt`: bump when any field is added/renamed/removed. Migration script lives next to the file.
- `golden_set.jsonl`: append-only for new records; existing records edited only via PR with `Why:` line in the commit message.
- `fixtures/`: never overwrite. New fixture for new doc; old fixture stays so old records remain reproducible.
- `eval_history.jsonl`: append-only; one row per CI run with `{date, phase, commit, ndcg@10, recall@50, faithfulness, latency_p95}`. Phase 5 wires the regression gate against this file.
- Record retirement: when a fixture's source document is genuinely deleted by the user, the records targeting it are marked `status: archived` (not removed); they stop counting toward eval but stay in the file for historical reference.

## Eval workflow

Phase 5 wires this. For now, the contract:

1. Eval harness loads `golden_set.jsonl` and `fixtures/`.
2. For each record, fires the query through rag-base's `POST /v1/search` (the hybrid pipeline).
3. Computes per-record:
   - `hit@K` for K in {1, 5, 10}: did any retrieved chunk satisfy any expected predicate
   - `ndcg@10`: standard
   - `forbidden_violation`: did any retrieved chunk match `forbidden_doc_ids` (boolean; true is bad)
4. Computes per-slice metrics: averages over slice membership.
5. Computes E2E (records with `expected_answer` only). Note: rag-base does not generate answers, it returns ranked chunks. E2E faithfulness/answer_relevance metrics live in the knowledge-base golden set where the agent-mode synthesis happens. rag-base's golden set focuses on retrieval metrics (NDCG, recall, hit@K, forbidden_violation) only.
6. Appends one row to `eval_history.jsonl`.
7. CI gate (Phase 5): NDCG@10 may not drop more than 2 percentage points vs the previous main-branch row; per-slice NDCG may not drop more than 5 points on any individual slice.

## Initial concrete plan (work order)

Curation runs **after Phase 1 closes** (chunks table populated, contextual blurbs and header paths in place, LightRAG ingested) so every record references the unified post-SOTA schema directly. Curation work is 2 to 3 calendar days of human-in-the-loop, runs in parallel with Phase 2 implementation, gates only the Phase 2 eval (not Phase 2 implementation itself).

Step 1: write `tests/golden/curate.py` (CLI: shows LLM draft, accept/edit/reject keys, writes accepted records to `golden_set.jsonl`, rejected to `_drafts/rejected.jsonl` for analysis). ~2 hours.

Step 2: choose and commit ~10 to 20 fixture documents under `fixtures/`. Cover the failure modes per slice: polysemous (Python language doc + Python snake doc), identifier (a docs section with `useEffect`/`useState`/etc), multi_hop (a small interconnected set of ~5 docs about web frameworks with real `DEPENDS_ON` / `USED_BY` relations), contextual_disambiguation (multi-section docs where leaf chunks reference each other implicitly), auto_merge / long_context (one long structured doc such as the rag-base `llm.txt`), time_travel (a versioned product spec with v1 and v2). Run Phase 1+ ingest against this fixture set so chunks/blurbs/graph all exist. ~3 hours.

Step 3: run Source-1 drafting per slice (Claude Sonnet 4.6, grounded in fixtures + graph dump). ~1 hour automated + 5 to 7 hours human review for ~125 synthetic-generated records.

Step 4: hand-craft `time_travel` pairs (10 pairs = 20 records) + `negative` (15 records). ~3 hours.

Step 5: validate the whole file with `tests/golden/validate.py`. Checks: every record meets quality bars; predicates resolve to >=1 chunk against the fixture-ingested chunks; no orphaned fixture references; every record's `slices` are consistent with its assertion fields (e.g. `multi_hop` slice records must have `expected_via` containing `"graph"` and a non-empty `expected_entity_path`); time_travel pairs have matching prefix ids and the un-scoped record's `expected_doc_ids` differs from the scoped record's. ~1 hour automated + 1 hour fix-up.

Step 6: commit `golden_set.jsonl` v1 (~160 records) + all fixtures + `lightrag_graph.json` snapshot. Ready for Phase 2 eval gate.

Total: 2 to 3 calendar days of focused work after Phase 1 closes.

## Schema example records

Saved separately as `tests/golden/_examples.jsonl` so the schema is concrete and reviewable; not loaded by the eval harness (the leading `_` excludes it). Examples exercise every assertion field (`expected_via`, `expected_blurb_visible`, `expected_parent_merged`, `expected_entity_path`, `time_scope`, `forbidden_doc_ids`) so the curator has a working reference for each slice.
