import json
from pathlib import Path
from typing import Any, Literal

from .metadata import ToolCallsMetadata
from .lookup_chat_sessions import LookupChatSessions

from src.agent.models import llm
from src import logger
from src.libs.helpers.database_helpers import fetch_one, fetch_all, execute
from src.libs.constants import COMPRESS_DB_ENTRY

app_log = logger.app_logger(f"{__name__}.app")

# Filter parring for SQL for lookup chat history
_HISTORY_FILTER_SQL = {
    "compressed":     True,
    "not_compressed": False,
    "all":            None,
}


class ReadChatLogs:
    """All SQL for the 'chat_logs' table."""

    def __init__(self, conn, lookup_chat_sess: LookupChatSessions, sys_prompt: str):
        self._conn = conn
        self._lookup_chat_sess = lookup_chat_sess
        self._sys_prompt = sys_prompt

    # Read chat logs

    def active_conversations(self) -> list[dict]:
        """
        Get all messages in a session with filter options.
        If session does not exists, return system prompt.
        """
        rows = fetch_all(
            conn=self._conn,
            sql=(
                """
                SELECT prompt, response
                FROM chat_logs
                WHERE session_id = %s AND state = 'external' AND is_compressed = FALSE
                ORDER BY created_at ASC;
                """
            ),
            params=(self._lookup_chat_sess.get_or_create_id(),),
        )

        msgs = [{"role": "system", "content": self._sys_prompt}]

        for prompt, response in rows:
            msgs.append(llm.user_message(prompt))
            msgs.append(llm.assistant_message(response))
        return msgs

    def conversation_history(
        self, filter: Literal["compressed", "not_compressed", "all"]
    ):
        if filter == "compressed":
            sql = (
                """
                SELECT prompt, response, total_prompt_tokens, total_output_tokens, metadata
                FROM chat_logs
                WHERE session_id = %s AND state = 'external' AND is_compressed = TRUE
                ORDER BY created_at ASC;
                """
            )
        elif filter == "not_compressed":
            sql = (
                """
                SELECT prompt, response, total_prompt_tokens, total_output_tokens, metadata
                FROM chat_logs
                WHERE session_id = %s AND state = 'external' AND is_compressed = FALSE
                ORDER BY created_at ASC;
                """
            )
        else:
            sql = (
                """
                SELECT prompt, response, total_prompt_tokens, total_output_tokens, metadata
                FROM chat_logs
                WHERE session_id = %s AND state = 'external'
                ORDER BY created_at ASC;
                """
            )
        rows = fetch_all(
            conn=self._conn,
            sql=sql,
            params=(self._lookup_chat_sess.get_id(),),
        )
        if not rows:
            return None

        convs = []
        for prompt, response, p_tkns, o_tkns, metadata in rows:
            metadata = metadata or {}
            user_msg = ({
                "role": "user",
                "content": prompt,
                "attachments": metadata.get("attachments")
            })

            if COMPRESS_DB_ENTRY in metadata:
                user_msg[COMPRESS_DB_ENTRY] = metadata[COMPRESS_DB_ENTRY]

            convs.append(user_msg)
            convs.append({
                "role": "assistant",
                "content":response,
                "prompt_tokens": p_tkns,
                "output_tokens": o_tkns,
            })
        return convs

    def latest_conversation_turn(self) -> list[dict] | None:
        row = fetch_one(
            conn=self._conn,
            sql=(
                """
                SELECT prompt, response
                FROM chat_logs
                WHERE session_id = %s AND state = 'external' AND is_compressed = FALSE
                ORDER BY created_at DESC
                LIMIT 1;
                """
            ),
            params=(self._lookup_chat_sess.get_or_create_id(),),
        )

        if not row:
            app_log.warning(
                "No conversation turns found in session '%s'",
                self._lookup_chat_sess.sess_name
            )
            return None
        return [
            {"role": "user", "content": row[0]},
            {"role": "assistant", "content": row[1]}
        ]

    def before_latest_conversations(self) -> list[dict] | None:
        """
        Get all previous user/assistant conversation turns right
        before the latest external one.
        """
        sess_id = self._lookup_chat_sess.get_or_create_id()
        rows = fetch_all(
            conn=self._conn,
            sql=(
                """
                SELECT prompt, response
                FROM chat_logs
                WHERE session_id = %s
                    AND id < (
                        SELECT MAX(id)
                        FROM chat_logs
                        WHERE session_id = %s AND state = 'external'
                    )
                ORDER BY id ASC;
                """
            ),
            params=(sess_id, sess_id),
        )

        if not rows:
            return None

        convs = []
        for prompt, response in rows:
            convs.append({"role": "user", "content": prompt})
            convs.append({"role": "assistant", "content": response})
        return convs

    def last_modified_time(self, sess_id: str | None = None) -> str | None:
        """Get the latest created time from session id in the chat logs."""
        row = fetch_one(
            conn=self._conn,
            sql=(
                """
                SELECT created_at
                FROM chat_logs
                WHERE session_id = %s
                ORDER BY created_at DESC
                LIMIT 1;
                """
            ),
            params=(sess_id,),
        )
        return str(row[0]) if row else None

    def latest_modified_session(self) ->  tuple[str, str] | None:
        """
        Get the chat 'session_name' and 'created_time' that has
        the latest 'created_time' of the chat logs.
        """
        row = fetch_one(
            conn=self._conn,
            sql=(
                """
                SELECT session_id, created_at
                FROM chat_logs
                ORDER BY created_at DESC
                LIMIT 1;
                """
            ),
        )

        if not row:
            return None

        sess_name = self._lookup_chat_sess.get_name(sess_id=str(row[0]))
        return (sess_name, str(row[1])) if sess_name else None  # Session name, created at
