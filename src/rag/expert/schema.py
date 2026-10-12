"""
Expert database for storing accurate documents chunks that can be queried
via vector search. The core tables are independent of chat sessions; 
session_experts is an optional link table created on demand.
"""

import threading

from src.config.models import EMBED_MODEL

from src.libs.models_database import EMB_MODEL_DIMENSION
from src.libs.helpers.database_helpers import init_database

from src import logger

app_log = logger.app_logger(f"{__name__}.app")

_schema_ready = False
_link_schema_ready = False

# Core tables: no dependency on chat sessions
_STATEMENTS = (
    """CREATE EXTENSION IF NOT EXISTS VECTOR;""",

    # One row per expert
    """
    CREATE TABLE IF NOT EXISTS experts (
        id              BIGSERIAL PRIMARY KEY,
        name            VARCHAR(255) NOT NULL UNIQUE,
        description     TEXT,
        embedding_model VARCHAR(255) NOT NULL,  -- model used to embed every chunk of this expert
        created_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    );
    """,

    # One row per source document
    """
    CREATE TABLE IF NOT EXISTS expert_documents (
        id           BIGSERIAL PRIMARY KEY,
        expert_id    BIGINT NOT NULL REFERENCES expert(id) ON DELETE CASSCADE,
        source_path  TEXT,
        title        TEXT,
        type         VARCHAR(20) NOT NULL,  -- e.g. pdf, markdown, code, spreadsheet
        content_hash TEXT NOT NULL,
        created_at   TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        metadata     JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        UNIQUE (expert_id, content_hash)
    )
    """,

    # One row per embedded chunk. expert_id is denormalised so the
    # retrieval query can filter without joining expert_documents.
    """
    CREATE TABLE IF NOT EXISTS expert_chunks (
        id            BIGSERIAL PRIMARY KEY,
        expert_id     BIGINT NOT NULL REFERENCES expert(id) ON DELETE CASSCADE,
        document_id   BIGINT NOT NULL REFERENCES expert_documents(id) ON DELETE CASSCADE,
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
    CREATE INDEX IF NOT EXISTS idx_exp_chunks_embeddings
    ON expert_chunks USING hnsw (embeddings vector_cosine_ops);
    """,

    # Filter by enabled experts
    """
    CREATE INDEX IF NOT EXISTS idx_exp_chunks_expert
    ON expert_chunks (expert_id);
    """,

    # Document listing / deletion per expert
    """
    CREATE INDEX IF NOT EXISTS idx_exp_docs_expert
    ON expert_documents (expert_id);
    """,
)

# Optional link table: only created when something first enables an expert
# in a session. Deleting a session or an expert only removes the link.
_LINK_STATEMENTS = (
    # Link enabled experts to sessions
    """
    CREATE TABLE IF NOT EXISTS session_experts (
        session_id UUID NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASSCADE,
        expert_id  BIGINT NOT NULL REFERENCES experts(id) ON DELETE CASSCADE,
        enabled_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (session_id, expert_id)
    );
    """,

    # Look up a session's enabled experts
    """
    CREATE INDEX IF NOT EXISTS idx_exp_session_expert
    ON session_experts (expert_id);
    """,
)

def ensure_schema(conn) -> None:
    """Create the core expert tables. Needs no chat tables."""
    global _schema_ready
    if _schema_ready:
        return
    dim = EMB_MODEL_DIMENSION.get(EMBED_MODEL)
    if dim is None:
        raise ValueError(
            f"Unknown embedding dimension for model '{EMBED_MODEL}'. "
        )

    app_log.debug("Initialising expert tables")
    stmts = tuple(s.format(dimension=int(dim)) for s in _STATEMENTS)
    init_database(conn=conn, stmts=stmts)

    _schema_ready = True


def ensure_link_schema(conn) -> None:
    """Create session_experts (and chat_sessions, which it references) on demand."""
    global _link_schema_ready
    if _link_schema_ready:
        return

    from src.agent.chat_session_logs.schema import ensure_schema as ensure_chat_schema

    ensure_schema(conn)
    ensure_chat_schema(conn)

    app_log.debug("Initialising session_experts link table")
    init_database(conn=conn, stmts=_LINK_STATEMENTS)

    _link_schema_ready = True
