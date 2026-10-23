"""
Expert database for storing documents chunks, searchable via vector search.
Document are stored ONCE and linked to any number of experts.

Interaction with this database:
- Delete an expert, or unlinking a document, removes only the link row.
  Documents and their chunks are never deleted by it.
- documents.content keeps the parsed Markdown, so a document can be
  re-chunked and re-embedded (e.g. after the embedding model changes)
  without re-parsing.
- documents.embedding_model records which model embedded that document's
  chunks.

The core tables are independent of chat sessions;
session_experts is an optional link table created on demand.
"""

from src.config.config_toml import EMBED_MODEL

from src.libs.custom_dataclasses import AgentHooks
from src.libs.models_database import EMB_MODEL_DIMENSION
from src.libs.helpers.database_helpers import fetch_all, fetch_one, init_database

from src import logger

app_log = logger.app_logger(f"{__name__}.app")

_schema_ready = False
_link_schema_ready = False

# TODO:
#
# Add a directory column for docuemnt table for filtering
# documents. e.g. (DIR / doc_name), (DIR / TYPE / doc_name)
# Which allows quick search for documents stored under
# the directory: --all-documents DIR/doc_name

# Core tables: no dependency on chat sessions
_STATEMENTS = (
    # pgvector extension
    """CREATE EXTENSION IF NOT EXISTS VECTOR;""",
    # Experts table:
    #
    # Storing all created experts data, one row per expert
    """
    CREATE TABLE IF NOT EXISTS experts (
        id              BIGSERIAL PRIMARY KEY,
        name            VARCHAR(255) NOT NULL UNIQUE,
        description     TEXT,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """,
    # Expert document table:
    #
    # Stores the general data for the uploaded documents, that are applicable
    # to all of its chunks.
    """
    CREATE TABLE IF NOT EXISTS documents (
        id              BIGSERIAL PRIMARY KEY,
        title           TEXT NOT NULL,
        type            VARCHAR(20) NOT NULL,   -- e.g. pdf, markdown, code
        content_hash    TEXT NOT NULL UNIQUE,
        source_path     TEXT,                   -- Where it was read from
        content         TEXT NOT NULL,          -- Parsed Markdown
        description     TEXT,  -- Optional summary of the whole document
        embedding_model VARCHAR(255) NOT NULL,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        metadata        JSONB NOT NULL DEFAULT '{{}}'::jsonb
    );
    """,
    # Expert documents table:
    #
    # Link the expert to the embedded documents in the database.
    """
    CREATE TABLE IF NOT EXISTS expert_documents (
        expert_id   BIGINT NOT NULL REFERENCES experts(id) ON DELETE CASCADE,
        document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        linked_at   TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (expert_id, document_id)
    );
    """,
    # Document chunks table:
    #
    # Stores all the chunks (including the text and embeddings) from the
    # document. One row per embedded chunk of a document, linked by
    # document id for the same document.
    """
    CREATE TABLE IF NOT EXISTS document_chunks (
        id            BIGSERIAL PRIMARY KEY,
        document_id   BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        chunk_index   INTEGER NOT NULL,
        embeddings    VECTOR({dimension}) NOT NULL,
        embed_tokens  INTEGER NOT NULL,
        content       TEXT NOT NULL,
        prompt_tokens INTEGER NOT NULL,
        metadata      JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        UNIQUE (document_id, chunk_index)
    );
    """,
    # HNSW index - must match the distance operator used in queries
    """
    CREATE INDEX IF NOT EXISTS idx_doc_chunks_embeddings
    ON document_chunks USING hnsw (embeddings vector_cosine_ops);
    """,
    # Filter by enabled experts
    """
    CREATE INDEX IF NOT EXISTS idx_expert_documents_document
    ON expert_documents (document_id);
    """,
)

# Optional link table: only created when something first enables an expert
# in a session. Deleting a session or an expert only removes the link.
_LINK_STATEMENTS = (
    # Link enabled experts to sessions
    """
    CREATE TABLE IF NOT EXISTS session_experts (
        session_id UUID NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
        expert_id  BIGINT NOT NULL REFERENCES experts(id) ON DELETE CASCADE,
        enabled_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (session_id, expert_id)
    );
    """,
    # Find the sessions that use an expert (the primary key already covers
    # looking up a session's experts).
    """
    CREATE INDEX IF NOT EXISTS idx_session_experts_expert
    ON session_experts (expert_id);
    """,
)


def stored_dimension(conn) -> int | None:
    """
    Vector dimension of document_chunks.embeddings, or None if the table is
    missing.
    """
    row = fetch_one(
        conn=conn,
        sql=("""
            SELECT atttypmod FROM pg_attribute
            WHERE attrelid = to_regclass('document_chunks')
                AND attname = 'embeddings' AND NOT attisdropped;
            """),
    )
    return int(row[0]) if row and row[0] > 0 else None


def ensure_schema(conn, hooks: AgentHooks | None = None) -> None:
    """Create the core expert tables. Needs no chat tables."""
    global _schema_ready
    if _schema_ready:
        return
    dim = EMB_MODEL_DIMENSION.get(EMBED_MODEL)
    if dim is None:
        raise ValueError(f"Unknown embedding dimension for model '{EMBED_MODEL}'. ")

    if hooks and hooks.on_loading:
        hooks.on_loading("Initialising expert tables...")

    stmts = tuple(s.format(dimension=int(dim)) for s in _STATEMENTS)
    init_database(conn=conn, stmts=stmts)

    # The vector size is fixed when teh table is first created. Warn, rather
    # than raise, so experts can still be listed and deleted after a model
    # change.
    existing = stored_dimension(conn)
    if existing is not None and existing != int(dim):
        app_log.warning(
            "Embedding model '%s' produces %d-dimension vectores but "
            "document_chunks stores %d. Documents must be re-embedded before "
            "adding or searching.",
            EMBED_MODEL,
            int(dim),
            existing,
        )

    _schema_ready = True


def ensure_link_schema(conn, hooks: AgentHooks | None = None) -> None:
    """
    Create session_experts (and chat_sessions, which it references) on demand.
    """
    global _link_schema_ready
    if _link_schema_ready:
        return

    from src.agent.chat_session_logs.schema import ensure_schema as ensure_chat_schema

    ensure_schema(conn)
    ensure_chat_schema(conn)

    if hooks and hooks.on_loading:
        hooks.on_loading("Initialising session_experts link table...")

    init_database(conn=conn, stmts=_LINK_STATEMENTS)

    _link_schema_ready = True
