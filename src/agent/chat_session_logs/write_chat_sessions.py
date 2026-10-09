from typing import Callable

from src.libs.helpers.database_helpers import fetch_one, fetch_all, execute

from src import logger

app_log = logger.app_logger(f"{__name__}.app")


class WriteChatSessions:
    """All SQL for the 'chat_session' table."""

    def __init__(self, conn, sess_name: str, sys_prompt_tkns: Callable[[], int]):
        self._conn = conn

        self.sess_name = sess_name
        self._sys_prompt_tkns = sys_prompt_tkns  # Call only when a new session is created

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
