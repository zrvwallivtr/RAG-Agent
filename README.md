# RAG Agent

A local AI assistant featuring long-term memory, file context injection and automatic token management. It runs on [Ollama](https://ollama.com) and PostgreSQL (pgvector), keeping all user's data on host machine.

- **CLI and TUI support:** Interact with the agent in the terminal with `agent "MESSAGE"` or enter full-screen terminal UI with `agent`.
- **Long-term memory:** Global, cross-session memories retrieved by vector similarity.
- **File context:** Attach PDFs, documents, spreadsheets, code and more. Parsed content is injected into the prompt and stored to the database for later retrieval by vector similarity.
- **Sessions:** Named sessions with persistent chat logs and message token counts.
- **Token management:** Live token counter and auto compression when the context window reaches its limit.
- **Single local database:** Sessions, memories and embeddings all live in a local PostgreSQL database.

---

## Quick start

### 1. Requirements

- Docker with the compose plugin
- Python 3.11 or newer

Clone the repo by(No binary version is available yet.):
```
git clone https://github.com/zrvwallivtr/RAG-Agent.git
```

### 2. Start services

All the docker services must be started before running the application:

```sh
cd ~/.agent_app
docker-compose up -d
```
Compose reads credentials from `~/.agent_app/.env`.

### 3. Install models

Install a caht model and an embedding model before you can start any conversation. To enable token counts and automatic compression features, install the tokenizer for the LLM as well:

```sh
agent --install-model OLLAMA_LLM_NAME
agent --install-model OLLAMA_EMBEDDING_MODEL_NAME
agent --install-tokenizer   # Auto installs a fallback tokenizer (specified in `~/.agent_app/config.toml` and all tokenizers for your installed models

# Exmaple
agent --install-model ministral-3:3b
agent --install-model nomic-embed-text
```

### 4. Chat

```sh
agent "Write a message..."                  # CLI: Prompt model
agent --new-session SESSION_NAME            # CLI: Create a new session
agent -s SESSION_NAME "Write a message..."  # CLI: Continue from a named session
agent                                       # TUI: Open the terminal UI
```

Run `agent -h` for all flags. See the [CLI reference](./docs/CLI_REFERENCE.md) and [TUI reference](./docs/TUI_REFERENCE.md).

---

## Slash Commands

Activate by adding the command at the start of a prompt.

| Command | Description |
| --- | --- |
| `/memorise <prompt>` | Extract key facts from the prompt and save them to long-term memory. |
| `/recall <prompt>` | Retrieve relevant memories from the database. |
| `/compress [prompt]` | Summarise the conversation to free up tokens. An optional prompt steers the summary. |

---

## How it works

**Sessions:**
Every conversation turn (prompt, response, token counts, attachment metadata) is saved to the `chat_logs` table and linked to a row in `chat_sessions`. Allowing the creating, storing and switching between sessions.

**Memory:**
Memories are global and shared across sessions. Each prompt is embedded, and the most similar memories are injected into the context for that turn. For more details: [Memory](./docs/MEMORY.md).

**Documents:**
Attached files are converted to plain text or Markdown, injected into the current conversation turn and stored in the knowledge base for similarity retrieval in later turns. Details: [Document Processing](./docs/DOCUMENT_PROCESSING.md).

**Token management:**
Before each turn the agent estimates the size of the history plus the new prompt, with 1,000 tokens reserved for the reply. If that exceeds teh model's limit, older turns are summarised automatically. `/compress` does the same on demand. The auto compression requires a known token limit for the chat model, sess [Configuration](./docs/CONFIGURATION.md).

**Web search:**
Still in development. See [Web Search](./docs/WEB_SEARCH.md).

---

## Configuration

Behaviour, models and features toggles live in `~/.agent_app/config.toml`. A default file is copied there on first run if none exists. Secrets (database and pgAdmin credentials) live in `~/.agent_app/.env`, never in `config.toml`.

```toml
[models]
chat                = "ministral_3b:latest"
memory              = "nomic-embed-text"
project_manager     = "gemma3:1b"

# Optional token limits
# chat_max_tokens   =
# pm_max_tokens     =

[load_system_prompt]
chat            = "system"

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

Every option is described in the [configuration reference](./docs/CONFIGURATION.md).

---

## Document index
For in-depth guides and system specifications, refer to the project docs:

* [Memory Architecture](./docs/MEMORY.md)
* [Document Processing](./docs/DOCUMENT_PROCESSING.md)
* [Web Search Features](./docs/WEB_SEARCH.md)
* [Full Configuration References](./docs/CONFIGURATION.md)
* [Command Line Interface References](./docs/CLI_REFERENCE.md)
* [Terminal User Interface References](./docs/TUI_REFERENCE.md)
* [Docker Services](./docs/DOCKER.md)

---

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
