import threading

from src.libs.helpers.database_helpers import init_database

from src.config.models import EMBED_MODEL

from src.libs.models_database import EMB_MODEL_DIMENSION

from src import logger

app_log = logger.app_logger(f"{__name__}.app")

_schema_ready = False

_STATEMENTS = (
    """CREATE EXTENSION IF NOT EXISTS VECTOR;""",

    """
    CREATE TABLE IF NOT EXISTS knowledge_base (
        id            BIGSERIAL PRIMARY KEY,
        session_id    VARCHAR(255) NOT NULL,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        expires_at    TIMESTAMPTZ,    -- Only applicable for web search contents
        type          VARCHAR(20) NOT NULL,
        embeddings    VECTOR({dimension}),
        embed_tokens  INTEGER NOT NULL,
        content       TEXT NOT NULL,
        prompt_tokens INTEGER NOT NULL,
        content_hash  TEXT,   -- Only applicable for web search contents
        metadata      JSONB NOT NULL DEFAULT '{{}}'::jsonb
    );
    """,

    # HNSW index - must match the distance operator used in queries
    """
    CREATE INDEX IF NOT EXISTS idx_kw_bs_embeddings
    ON knowledge_base USING hnsw (embeddings vector_cosine_ops);
    """,

    # Index for session id and knowledge type lookups
    """
    CREATE INDEX IF NOT EXISTS idx_kw_bs_session_type
    ON knowledge_base (session_id, type);
    """,

    # Index for expires at lookups
    """
    CREATE INDEX IF NOT EXISTS idx_kw_bs_expires
    ON knowledge_base (expires_at) WHERE expires_at IS NOT NULL;
    """,
)


def ensure_schema(conn) -> None:
    global _schema_ready
    if _schema_ready:
        return
    dim = EMB_MODEL_DIMENSION.get(EMBED_MODEL)
    if dim is None:
        raise ValueError(
            f"Unknown embedding dimension for model '{EMBED_MODEL}'. "
        )

    app_log.debug("Initialising knowledge base table")
    stmts = tuple(s.format(dimension=int(dim)) for s in _STATEMENTS)
    init_database(conn=conn, stmts=stmts)

    _schema_ready = True
