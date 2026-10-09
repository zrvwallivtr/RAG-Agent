import threading

from src.libs.helpers.database_helpers import init_database

from src import logger

app_log = logger.app_logger(f"{__name__}.app")

_schema_ready = False

_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS chat_sessions (
        session_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        session_name        VARCHAR(255) NOT NULL UNIQUE,
        created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        session_used_tokens INTEGER NOT NULL DEFAULT 0
    );
    """,

    # Schema migration: Add missing column if table already existed
    # """
    # ALTER TABLE chat_sessions
    # ADD COLUMN IF NOT EXISTS session_used_tokens INTEGER NOT NULL DEFAULT 0;
    # """,

    """
    CREATE TABLE IF NOT EXISTS chat_logs (
        id                  BIGSERIAL PRIMARY KEY,
        session_id          UUID NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
        created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        prompt              TEXT NOT NULL,
        response            TEXT NOT NULL,
        state               VARCHAR(20) NOT NULL CHECK (state IN ('external', 'internal')),
        total_prompt_tokens INTEGER NOT NULL,
        total_output_tokens INTEGER NOT NULL,
        is_compressed       BOOL NOT NULL DEFAULT FALSE,
        metadata            JSONB DEFAULT '{}'::jsonb
    );
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_chat_logs_session_id
    ON chat_logs (session_id);
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_chat_logs_session_created
    ON chat_logs (session_id, created_at);
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_chat_logs_created_at
    ON chat_logs (created_at DESC);
    """,
)


def ensure_schema(conn) -> None:
    """
    Create the chat log tables and indexes if missing.

    Runs the DDL ONCE per process. Later calls return immediately, so it is
    cheap to call from every ChatLogs.__init__. A failed run is not remembered,
    so the next call retries.
    """
    global _schema_ready
    if _schema_ready:
        return

    app_log.debug("Initialising chat tables and indexes")
    init_database(conn=conn, stmts=_STATEMENTS)

    _schema_ready = True
