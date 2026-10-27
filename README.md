# RAG Agent

A local AI assistant featuring long-term memory, file context injection and automatic token management. It runs on [Ollama](https://ollama.com) and PostgreSQL (pgvector), keeping all user's data on host machine.

- **CLI and TUI support:** Interact with the agent in the terminal with `agent "MESSAGE"` or enter full-screen terminal UI with `agent`.
- **File context:** Attach PDFs, documents, spreadsheets, code and more. Parsed content is injected into the prompt and stored to the database for later retrieval by vector similarity.
- **Sessions:** Named sessions with persistent chat logs and message token counts.
- **Experts:** Named knowledge bases built from uploaded documents of choice. They exist independently of chat sessions and can be reused by any of them.
- **Token management:** Live token counter and auto compression when the context window reaches its limit.
- **Single local database:** Sessions, memories and embeddings all live in a local PostgreSQL database.

---

## Quick start

### 1. Requirements

- Docker with the compose plugin
- Python 3.11 or newer

Clone the repo by:
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

### 5. Expert

An expert is a named subject area. Its documents are chunked, embedded and searched by similarity, where the agent can answer from them.

```sh
agent --create-expert EXPERT_NAME                       # Create an expert
agent --upload-to-expert /path/to/document              # Chunk, embed and add a document
agent --list-experts                                    # List all created experts
agent --enable-session-expert SESSION_NAME EXPERT_NAME  # Enable the expert for the session in-order to use it
agent --delete-expert EXPERT_NAME                       # Delete the expert (No uploaded documents will be deleted)
```

Run `agent -h` for all flags. See the [CLI reference](./docs/CLI_REFERENCE.md) and [TUI reference](./docs/TUI_REFERENCE.md).

---

## Slash Commands

Activate by adding the command at the start of a prompt.

| Command | Description |
| --- | --- |
| `/compress [prompt]` | Summarise the conversation to free up tokens. An optional prompt steers the summary. |

---

## How it works

**Sessions:**
Every conversation turn (prompt, response, token counts, attachment metadata) is saved to the `chat_logs` table and linked to a row in `chat_sessions`. Allowing the creating, storing and switching between sessions.

**Documents:**
Attached files are converted to plain text or Markdown, injected into the current conversation turn and stored in the knowledge base for similarity retrieval in later turns. Details: [Document Processing](./docs/DOCUMENT_PROCESSING.md).

**Token management:**
Before each turn the agent estimates the size of the history plus the new prompt, with 1,000 tokens reserved for the reply. If that exceeds teh model's limit, older turns are summarised automatically. `/compress` does the same on demand. The auto compression requires a known token limit for the chat model, sess [Configuration](./docs/CONFIGURATION.md).

**Experts:**
An expert is a named set of documents. Each document is parsed with Docling, split into chunks (Docling's `HybridChunker`, sized with the embedding model's tokenizer), embedded and stored in pgvector.

---

## Configuration

Behaviour, models and features toggles live in `~/.agent_app/config.toml`. A default file is copied there on first run if none exists. Secrets (database and pgAdmin credentials) live in `~/.agent_app/.env`, never in `config.toml`.

```toml
[models]

ollama_host = "http://localhost:11434"

model     = "ministral-3:3b"
embedding = "nomic-embed-text"

fallback_tokenizer  = "gpt2"

[file_reader]

retrieve_entry_limit                        = 3
enable_auto_read_dropbox                    = true
enable_docling                              = true
```

Every option is described in the [configuration reference](./docs/CONFIGURATION.md).

---

## Document index

For in-depth guides and system specifications, refer to the project docs:

* [Document Processing](./docs/DOCUMENT_PROCESSING.md)
* [Full Configuration References](./docs/CONFIGURATION.md)
* [Command Line Interface References](./docs/CLI_REFERENCE.md)
* [Terminal User Interface References](./docs/TUI_REFERENCE.md)
* [Docker Services](./docs/DOCKER.md)

---

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
