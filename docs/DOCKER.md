# Docker Services

Backing services run with Docker Compose. The compose file is at `~/.agent/docker-compose.yaml`.

```sh
cd ~/.agent
docker compose up -d    # Start
docker compose ps       # Check status
docker compose down     # Stop (Data is kept in volumes)
```

## Services

| Service | Image | Port | Purpose |
| --- | --- | --- | --- |
| `ollama` | `ollama/ollama` | `11434` | Runs the chat and embedding models. |
| `postgres` | `pgvector/pgvector:pg17` | `5432` | Sessions, chat logs, memories and embeddings. |
| `pgadmin` | `dpage/pgadmin4` | `5050` | Optional web UI for the database. |

All ports should be bound to `127.0.0.1` so nothing is exposed to the network.

## Environment variables

Create `~/.agent_app/.env`:

```sh
# PostgreSQL
PGDB_DBNAME      = "pgdb"
PGDB_USER        = "pguser"
PGDB_PASSWORD    = "postgres"

# pgAdmin (optional)
# PGADMIN_EMAIL       = "admin@email.com"
# PGADMIN_PASSWORD    = "admin"
```

## pgAdmin

Open <http://127.0.0.1:5050> in a browser of choice and login with `PGADMIN_EMAIL` and `PGADMIN_PASSWORD`. When registering the server, use host `postgres` and prot `5432` (pgAdmin reaches it over teh internal Docker network).

## Default compose file

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

  # Pgadmin (optional)
  # pgadmin:
  #   image: dpage/pgadmin4
  #   container_name: pgadmin
  #   restart: unless-stopped
  #   # profiles: ["admin"] # Opt-in via 'docker compose --profile admin up'
  #   environment:
  #     PGADMIN_DEFAULT_EMAIL: ${PGADMIN_EMAIL}
  #     PGADMIN_DEFAULT_PASSWORD: ${PGADMIN_PASSWORD}
  #   ports:
  #     - "127.0.0.1:5050:80" # Localhost only
  #   networks:
  #     - internal
  #   depends_on:
  #     - postgres

networks:
  internal:
    driver: bridge

volumes:
  ollama_models:
  pgdata:
```

