# Web Search & Crawling (Still in development)

* **Caching:** Crawled page content is cached in the `knowledge_base` table with a `content_hash` (change detection) and `expires_at`
* Executes web search/crawl based on model triggers or explicit user requests.
* Synthesizes and presnets only the finalized, context-aware answer in the main chat output.


## Security

Since the agent fetches and processes untrusted external content, there are several built in protections:

* **SSRF protection:**
* **Container hardening:**
* **API authentication:**
* **Prompt-injection defense:**
All Services (Postgres, pgAdmin, Crawl4AI) are bound to `127.0.0.1` only - never exposed beyond the local machine.

