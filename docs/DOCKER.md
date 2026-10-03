

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

