# Configuration

Settings live in `~/.agent_app/config.toml`. If the file is missing, the default is copied from the code base.

## `[models]`

| Option | Default | Description |
| --- | --- | --- |
| `ollama_host` | `http://localhost:11434` | URL for the Ollama docker service from `~/.agent_app/docker-compose.yaml`, default URL is `http://localhost:11434`. |
| `model` | `ministral-3:3b` | Ollama model used for conversation. |
| `embedding` | `nomic-embed-text` | Ollama model used for all embeddings (memory, documents). |
| `fallback_tokenizer` | `gpt2` | Hugging Face tokenizer for `model` model. |


**Optional configs:**

| Option | Default | Description |
| --- | --- | --- |
| `model_max_tokens` |  | Specified token limit for `model` model. |
| `embedding_max_tokens` |  | Specified token limit for `embedding` model. |

If the agent does not know the model's token limit (either form built in or specified in the configs), the token counter and auto compression feature are disabled.

## `[memory]`

| Option | Default | Description |
| --- | --- | --- |
| `retrieve_entry_limit` | `3` | Maximum number of memories retrieved per recall. |
| `auto_memorhy_store_enable_at_model_tokens` | `true` | Automatically retrieve relevant memories for every prompt. |
| `enable_auto_memory_retrieve` | `true` | Automatically extract memories at the end of a conversation. |
| `enable_auto_memory_store` | `128000` | Minimum model token limit required for automatic extraction. A safeguard that stops small models from saving irrelevant information. |

## [`file_reader`]

| Option | Default | Description |
| --- | --- | --- |
| `enable_auto_read_dropbox` | `true` | Parse attachments with Docling. When `false`, the basic parsers are used instead. |
| `enable_docling` | `true` | Automatically read files from the dropbox (`~/.agent/uploads`) directory. |
| `auto_read_dropbox_enable_at_model_tokens` | `128000` | Minimum model token limit required for automatic reading. |

## Full default file

```toml
[models]

ollama_host = "http://localhost:11434"

model     = "ministral_3b:latest"
embedding = "nomic-embed-text"

# Optional token limits
# model_max_tokens =
# pm_max_tokens    =

[memory]
retrieve_entry_limit                        = 3
auto_memory_store_enable_at_model_tokens    = 128000
enable_auto_memory_retrieve                 = true
enable_auto_memory_store                    = true

[file_reader]
auto_read_dropbox_enable_at_model_tokens    = 128000
enable_auto_read_dropbox                    = true

[search]
engine                                  = "http://localhost:8080/search"
max_results                             = 3
max_char_per_page                       = 3000
auto_web_search_enable_at_model_tokens  = 128000
enable_auto_web_search                  = true
```
