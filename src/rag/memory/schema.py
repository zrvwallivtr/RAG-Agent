import threading

from src.libs.helpers.database_helpers import init_database
from src import logger

app_log = logger.app_logger(f"{__name__}.app")

_scheme_ready = False

_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS memory (
        id              BIGSERIAL PRIMARY KEY,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        embeddings      VECTOR({dimension}),
        prompt_tokens   INTEGER NOT NULL,
        content         TEXT NOT NULL,
        content_hash    TEXT,
        category        TEXT NOT NULL,
        extraction      VARCHAR(20) NOT NULL CHECK (extraction IN ('manual', 'auto'))
    );
    """,

    # HNSW index - must match the distance operator used in queries
    """
    CREATE INDEX IF NOT EXISTS idx_memory_embedding
    ON memory USING hnsw (embeddings vector_cosine_ops);
    """,

    # Index for category lookup
    """
    CREATE INDEX IF NOT EXISTS idx_memory_category
    ON memory (category);
    """,

    # Unique constraint on content_hash to enforce dedupe at database level
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_content_hash
    ON memory (content_hash);
    """,
)


def ensure_schema(conn) -> None:
    global _scheme_ready
    if _scheme_ready:
        return

    app_log.debug("Initialising memory table")
    init_database(conn=conn, stmts=_STATEMENTS)

    _scheme_ready = True
