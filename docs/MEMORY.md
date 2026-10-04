# Memory

Long-term memory is **global**: memories are shared across all session. They are stored in PostgreSQL ad searched with pgvector cosine similarity.

## Memory entries

Each entry has:

| Field | Values |
| --- | --- |
| Category | `preference`, `stack`, `fact`, `project`, `instruction`, `correction` |
| Extraction method | `manual` (saved with `/memories`) or `auto` (extracted by the agent) |

## Saving memories

- **Manually:** `/memories <prompt>` extracts key facts and saves them.
- **Automatically:** When `enable_auto_memory_store` is no, memories are extracted at the end of a conversation. This only runs for models whose token limit is at leat `auto_memory_store_enable_at_model_tokens`.

## Retrieving memories

- **Manually:** `/recall <prompt` returns the memories most relevant to the prompt.
- **Automatically:** When `enable_auto_memory_store` is on, every prompt is embedded and up to `retrieve_entry_limit` similar memories are added to the context for that turn. They are not written to the chat history.

If the embedding model is unavailable, automatic memory and document retrieval are skipped for that turn.

## Related settings

See the `[memory]` section of teh [configuration reference](./CONFIGURATION.md).
