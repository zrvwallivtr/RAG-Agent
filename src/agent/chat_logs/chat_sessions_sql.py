from typing import Callable

from src import logger
from src.libs.helpers.database_helpers import fetch_one, fetch_all, execute

app_log = logger.app_logger(f"{__name__}.app")


class ChatSessionsSQL:
    """All SQL for the 'chat_session' table."""

    def __init__(self, conn, sess_name: str, sys_prompt_tkns: Callable[[], int]):
        self._conn = conn
        self.sess_name = sess_name

        self._sys_prompt_tkns = sys_prompt_tkns  # Call only when a new session is created

    # Lookup chat sessions

    def get_id(self) -> str | None:
        """
        Look up this session's id. READ-ONLY: Never creates anything.
        Returns None if the session row does not exist.

        USE WHEN:
        - The session is certain to exist already.
        - Listing, searching or checking existence (None is a valid answer).
        - Deleting or clearing (Must NOT make up session that was not there).
        """
        app_log.debug("Fetching session id for session '%s'", self.sess_name)
        row = fetch_one(
            conn=self._conn,
            sql="SELECT session_id FROM chat_sessions WHERE LOWER(session_name) = LOWER(%s);",
            params=(self.sess_name,)
        )

        if not row:
            app_log.info("Session id for session '%s' not found", self.sess_name)
            return None
        return str(row[0])

    def get_or_create_id(self) -> str:
        """
        Return this session's id, CREATING the session row if it is missing. Never returns None,
        and writes to the database on a miss.

        USE WHEN:
        - Saving data that needs a session_id foreign key (add_conversation_turn).
        - The session may not exist yet.
        
        NEVER USE IN:
        - Read-only paths, listing or delete.
        """
        return self.get_id() or self.create()

    def get_name(self, sess_id: str) -> str | None:
        row = fetch_one(
            conn=self._conn,
            sql="SELECT session_name FROM chat_sessions WHERE session_id = %s;",
            params=(sess_id,),
        )

        if not row:
            app_log.warning("Session name for session id '%s' not found", sess_id)
            return None
        return str(row[0])

    def get_id_from_name(self, sess_name: str) -> str | None:
        row = fetch_one(
            conn=self._conn,
            sql="SELECT session_id FROM chat_sessions WHERE session_name = %s;",
            params=(id,),
        )

        if not row:
            app_log.warning("Session id for session name '%s' not found", sess_name)
            return None
        return str(row[0])

    def get_data(self, sess_name: str | None = None) -> tuple[str, str, int] | None:
        name = sess_name if sess_name else self.sess_name  # Specified session name (optional)
        row = fetch_one(
            conn=self._conn,
            sql=(
                """
                SELECT session_id, created_at, session_used_tokens
                FROM chat_sessions
                WHERE session_name = %s;
                """
            ),
            params=(name,),
        )

        if not row:
            app_log.warning("Session data for session name '%s' not found", name)
            return None
        return str(row[0]), str(row[1]), int(row[2])  # session id, created at, session used tokens

    def get_all_metadata(self) -> dict | None:
        rows = fetch_all(
            conn=self._conn,
            sql=(
                """
                SELECT session_id, session_name, created_at
                FROM chat_sessions;
                """
            ),
        )

        if not rows:
            app_log.debug("No existing session found in 'chat_sessions'")
            return None

        return {
            str(r[0]): {"session_name": str(r[1]), "created_at": str(r[2])}
            for r in rows
        }

    # Write to chat sessions

    def create(self) -> str:
        """Create session entry on the chat_sessions table and return its session id."""
        app_log.debug("Creating new session '%s'", self.sess_name)
        row = fetch_one(
            conn=self._conn,
            sql=(
                """
                INSERT INTO chat_sessions
                (session_name, session_used_tokens) VALUES (%s, %s)
                RETURNING session_id;
                """
            ),
            params=(self.sess_name, self._sys_prompt_tkns()),
        )

        if row is None:
            raise RuntimeError("Failed to create session: Database return no ID.")

        app_log.info("New session '%s' created", self.sess_name)
        return str(row[0])

    def update_used_tokens(self, sess_used_tkns: int) -> None:
        execute(
            conn=self._conn,
            sql=(
                """
                UPDATE chat_sessions
                SET session_used_tokens = %s
                WHERE LOWER(session_name) = LOWER(%s);
                """
            ),
            params=(sess_used_tkns, self.sess_name),
        )
