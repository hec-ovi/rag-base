<!--
Anthropic Contextual Retrieval prompt.

Used by api/src/services/contextual_retrieval.py when a document is ingested
through POST /v1/documents with `contextual_retrieval: true`. The full document
is sent as a stable prefix on every per-chunk call so vLLM's automatic prefix
caching reuses the KV cache across calls. Output is the 50-100 token blurb
prepended to chunks.indexed_content (alongside the existing title + header path
prefixes); chunks.content stays untouched.

Edit the prompt body below freely; this file is loaded once at api startup.
After editing, restart the api container to pick up the change:
  docker compose restart api

Variables (substituted by the service via simple {{name}} replacement):
  {{document}}  the full source document text (long; the cache prefix)
  {{chunk}}     the chunk text we want a contextual blurb for (short; varies per call)
-->

<document>
{{document}}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{{chunk}}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
