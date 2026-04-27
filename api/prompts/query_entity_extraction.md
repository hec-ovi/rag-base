<!--
Query-time entity extraction prompt.

Used by api/src/services/lightrag_store.py:extract_query_entities when
POST /v1/search runs with include_graph=true. The LLM extracts entity names
from the user's query so we can match them against the LightRAG entity graph
in Memgraph. Concise-mode call (chat_template_kwargs.enable_thinking=false on
vLLM via the llm_complete closure), so the model emits the JSON directly
without reasoning preamble.

Edit the prompt body below freely; this file is loaded once at api startup.
After editing, restart the api container to pick up the change:
  docker compose restart api

Variables (substituted by the service via simple {{name}} replacement):
  {{query}}  the user's search query string
-->

Extract the named entities from the following question.
Output ONLY a JSON list of strings (entity names), nothing else.
If there are no clear entities, output [].

Question: {{query}}
