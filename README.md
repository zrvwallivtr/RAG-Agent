# RAG Agent

A local Command-Line Interface (CLI) AI assistant featuring long-term memory, file context injection, (isolated web crawling / search and automated token management).

## Quick start

All the docker services must be started before running the application:

```sh
cd ~/.agent_app
docker-compose up -d
```

### CLI

Install a LLM and embedding model before you can start any conversation with the agent, to enable token count features install the tokenizer for the LLM as well:

```sh
agent --install-model OLLAMA_LLM_NAME
agent --install-model OLLAMA_EMBEDDING_MODEL_NAME
agent --install-tokenizer HUGGING_FACE_REPO

# Exmaples
agent --install-model ministral-3:3b    # Install ollama model 'ministral-3:3b'
agent --install-model nomic-embed-text  # Install ollama model 'nomic-embed-text'
agent --install-tokenizer gpt2         # Install tokenizer for model 'gpt2'
```

Chat with the agent in the terminal by typing:

```sh
agent "Write a message..."
```

Use the native help command `-h` for more flag options:

```sh
agent -h
```

### TUI

To enter the tui environment simply enter the application name:

```sh
agent
```

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

## Features

### Core Capabilities

* **CLI Flags Support:** Built-in command-line arguments.
* **Local Data Storage:** User data, session histories, chat logs, memories and document/web content embeddings are stored in a single **PostgreSQL** database (with the **pgvector** extension).

### Context & History Management

* **Logging:** Conversation turns (prompt, response, token counts, tool calling) are persisted per session to the `chat_logs` table, linked to a `chat_sessions` table via `session_id`.
* **Automated Session Compression:** (needs update) Dynamically summarizes older conversations history when context limits are reached, preventing context-window crashes.
* **Multi-session support:** Create, store and switch between chat sessions. Session names and IDs are managed in a dedicated `chat_sessions` table.

### Long-Term Memory

The agent automatically retrieves relevant memories using **pgvector** cosine-similarity search.
Memory is **global and cross-session**, saved automatically or manually by `/memorise`.
Each memory entry is tagged with a category (`preference`, `stack`, `fact`, `project`, `instruction`, `correction`), with an extraction method (`manual` or `auto`) to narrow down the entry context.

### Document Reading & Retrieval

Converts different file formats content into clean plaintext/markdown context strings, automatically injecting them into the active chat context and persisting them to the knowledge base for future retrieval.

* **Plain Text:** `.txt`
* **Data & Configuration Formats:** `.csv`, `.xlsx`, `.yaml`, `.yml`, `.toml`, `.xml`
* **Documents:** `.pdf`, `.docx`, `.epub`
* **Code & Scripts:** `.py`, `.js`, `.ts`, `.tsx`, `.json`, `.md`, `.sh`, `.html`, `.css`, `.rs`, `.go`
* **Fallback Behavior:** Any unlisted text-based format defaults to be read as plain-text.
* **Deduplication:** Documents are hashed (SHA-256) before embedding; re-uploading identical content is detected and skipped rather than re-embedded.
* **Path Safety:** All document reads are resolved and validated against a fixed uploads directory to prevent path traversal outside the allowed folder.

---

## Slash Commands

Activate by adding the commands at the start of every prompt.

| Command | Description |
| --- | --- |
| `/memorise <prompt>` | Instruct agent to extract and save key facts from the attached prompt to the embedding database. |
| `/recall <prompt>` | Retrieves relevant memories from the app embedding database. |
| `/compress <prompt (optional)>` | Manually calling model to summarise the conversations to free up session token usages, additional prompt could be added for model summarisation behavior. |

---

## Configuration

All agent behavior, models and feature toggles are managed through `~/.agent_app/config.toml`. Here is the default configuarion, which will be copied from the code base to the app directory if no custom config file is detected:

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

## Secrets

Secrets (database password, `CRAWL4AI_API_TOKEN`, pgAdmin credentials) are kept in a separate `.env` file, never in `config.toml`.

## Docker Services

Backing services are managed via `docker compose` and are all bound to `127.0.0.1` (never exposed to the network):

| Service | Purpose | Port | Notes |
| --- | --- | --- | --- |

The docker compose file is stored inside the app directory in `~/.agent_app/docker-compose.yaml`. The following is the default settings for the docker services:

```yaml
services:
  # Ollama
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    networks:
      - internal
    volumes:
      - ollama_models:/root/.ollama
    restart: unless-stopped

  # Postgres
  postgres:
    image: pgvector/pgvector:pg17
    container_name: pgcontainer
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${PGDB_USER:-pguser}
      POSTGRES_PASSWORD: ${PGDB_PASSWORD}
      POSTGRES_DB: ${PGDB_DBNAME:-pgdb}
    ports:
      - "127.0.0.1:5432:5432" # Localhost only
    volumes:
      - pgdata:/var/lib/postgresql/data
    networks:
      - internal # Isolated network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${PGDB_USER:-pguser} -d ${PGDB_DBNAME:-pgdb}"]
      interval: 5s
      timeout: 5s
      retries: 5

  # Pgadmin
  pgadmin:
    image: dpage/pgadmin4
    container_name: pgadmin
    restart: unless-stopped
    # profiles: ["admin"] # Opt-in via 'docker compose --profile admin up'
    environment:
      PGADMIN_DEFAULT_EMAIL: ${PGADMIN_EMAIL}
      PGADMIN_DEFAULT_PASSWORD: ${PGADMIN_PASSWORD}
    ports:
      - "127.0.0.1:5050:80" # Localhost only
    networks:
      - internal
    depends_on:
      - postgres

networks:
  internal:
    driver: bridge

volumes:
  ollama_models:
  pgdata:
```

### License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
