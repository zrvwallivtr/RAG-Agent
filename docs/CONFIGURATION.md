# Configuration

## Models

Everything in this section is under `[models]`.

| Variable | Description |
| --- | --- |
| `ollama_host` | Set the URL for the Ollama docker service specified in `~/.agent_app/docker-compose.yaml`, default URL is `http://localhost:11434`. |
| `chat` | Set LLM for base agent model. |
| `memory` | Set LLM for all memory extraction and interpertation work flows. |
| `embedding` | Set embedding model for all embedding operations. |
| `fallback_tokenizer` | Set a fallback tokenizer in case any models does not have a tokenizer. |


### Specifying token limits for models that are not in the base code (Optional):

| Variable | Description |
| --- | --- |
| `chat_max_tokens` | Specified token limit for `chat` model. |
| `memory_max_tokens` | Specified token limit for `memory` model. |
| `embedding_max_tokens` | Specified token limit for `embedding` model. |

## Load system prompt

Everything in this section is under `[load_system_prompt]`.

| Variable | Description |
| --- | --- |
| `chat` |  |

## Memory

Everything in this section is under `[memory]`.

| Variable | Description |
| --- | --- |
| `retrieve_entry_limit` | Set the maximun memory entry the agent can retrieve with memory recalls. |
| `auto_memorhy_store_enable_at_model_tokens` | Set the minimum model max token count before enabling auto memory extraction at the end of every conversation, a simply saveguard to prevent smaller models from extracting irrelevant information into memory database. |
| `enable_auto_memory_retrieve` | Toggle on/off for auto memory recall from the database. |
| `enable_auto_memory_store` | Toggle on/off for auto memory extraction to the database. |

## File reader

Everything in this section is under `[file_reader]`.

| Variable | Description |
| --- | --- |
| `enable_auto_read_dropbox` | Toggle on/off for auto read embedding |
| `enable_docling` | Toggle on/off the enable docling for parsing attachments. When toggled off, none docling basic parsers will be used instead. |
