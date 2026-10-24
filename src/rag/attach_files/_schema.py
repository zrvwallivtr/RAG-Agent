"""
Attachments database for storing the uploaded files during a session, searchable
via vector search.
"""

from src.libs.custom_dataclasses import AgentHooks
from src.libs.helpers.database_helpers import fetch_one, init_database

from src.config.config_toml import EMBED_MODEL

from src.libs.models_database import EMB_MODEL_DIMENSION

from src import logger

app_log = logger.app_logger(f"{__name__}.app")

_schema_ready = False

_STATEMENTS = (
    # pgvector extension
    """CREATE EXTENSION IF NOT EXISTS VECTOR;""",
    # Attachments table:
    #
    # Stores the general data for the uploaded files.
    """
    CREATE TABLE IF NOT EXISTS attachments (
        id              BIGSERIAL PRIMARY KEY,
        title           TEXT NOT NULL,
        type            VARCHAR(20) NOT NULL,   -- e.g. pdf, markdown, code
        content_hash    TEXT NOT NULL UNIQUE,
        source_path     TEXT,                   -- Where it was read from
        content         TEXT NOT NULL,          -- Parsed Markdown
        embedding_model VARCHAR(255) NOT NULL,
        session_id      UUID NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
        uploaded_at     TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        metadata        JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        UNIQUE (session_id, content_hash)
    );
    """,
    # Attachment chunks table:
    #
    # Stores all teh chunks (including the text and embeddings) from the
    # file. One row per embedded chunk of a file, linked by attachment id
    # for the same file.
    """
    CREATE TABLE IF NOT EXISTS attachment_chunks (
        id            BIGSERIAL PRIMARY KEY,
        attachment_id BIGINT NOT NULL REFERENCES attachments(id) ON DELETE CASCADE,
        chunk_index   INTEGER NOT NULL,
        embeddings    VECTOR({dimension}) NOT NULL,
        embed_tokens  INTEGER NOT NULL,
        content       TEXT NOT NULL,
        prompt_tokens INTEGER NOT NULL,
        metadata      JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        UNIQUE (attachment_id, chunk_index)
    );
    """,
    # HNSW index - must match the distance operator used in queries
    """
    CREATE INDEX IF NOT EXISTS idx_att_chunks_embeddings
    ON attachment_chunks USING hnsw (embeddings vector_cosine_ops);
    """,
)


def _stored_dimension(conn) -> int | None:
    """
    Vector dimension of attachment_chunks.embeddings, or None if the table is
    missing.
    """
    row = fetch_one(
        conn=conn,
        sql=("""
            SELECT atttypmod FROM pg_attribute
            WHERE attrelid = to_regclass('attachment_chunks')
                AND attname = 'embeddings' AND NOT attisdropped;
            """),
    )
    return int(row[0]) if row and row[0] > 0 else None


def ensure_schema(
    conn,
    emb_model: str = EMBED_MODEL,
    emb_model_dim: dict[str, int] = EMB_MODEL_DIMENSION,
    hooks: AgentHooks | None = None,
) -> None:
    """Create attachments tables. Requires session table."""
    global _schema_ready
    if _schema_ready:
        return
    dim = emb_model_dim.get(emb_model)
    if dim is None:
        raise ValueError(f"Unknown embedding dimension for model '{EMBED_MODEL}'. ")

    from src.agent.chat_session_logs.schema import ensure_schema as ensure_chat_schema

    ensure_chat_schema(conn)

    if hooks and hooks.on_loading:
        hooks.on_loading("Initialising attachments table...")

    stmts = tuple(s.format(dimension=int(dim)) for s in _STATEMENTS)
    init_database(conn=conn, stmts=stmts)

    # The vector size is fixed when teh table is first created. Warn, rather
    # than raise, so experts can still be listed and deleted after a model
    # change.
    existing = _stored_dimension(conn)
    if existing is not None and existing != int(dim):
        app_log.warning(
            "Embedding model '%s' produces %d-dimension vectores but "
            "attachment_chunks stores %d. Documents must be re-embedded before "
            "adding or searching.",
            emb_model,
            int(dim),
            existing,
        )

    _schema_ready = True
