import json
from pathlib import Path
from typing import Any, Literal

from src.libs.helpers.database_helpers import fetch_one, fetch_all, execute

from src.agent.models import llm
from src import logger
from .metadata import ToolCallsMetadata
from .lookup_chat_sessions import LookupChatSessions

app_log = logger.app_logger(f"{__name__}.app")

# Filter parring for SQL for lookup chat history
_HISTORY_FILTER_SQL = {
    "compressed": True,
    "not_compressed": False,
    "all": None,
}


class WriteChatLogs:
    """All SQL for the 'chat_logs' table."""

    def __init__(self, conn, lookup_chat_sess: LookupChatSessions):
        self._conn = conn
        self._lookup_chat_sess = lookup_chat_sess

    def add_turn(
        self,
        prompt: str,
        response: str,
        state: Literal["external", "internal"],
        # --- Tool calls ---
        attchmnts: list[Path] | None = None,
        unfmt_mem_prompt: str | None = None,
        unfmt_recall_prompt: str | None = None,
        unfmt_cmp_prompt: str | None = None,
        cmp_method: Literal["default", "instruct", "auto"] | None = None,
        # --- Token counts ---
        p_tkns: int = 0,
        o_tkns: int = 0,
    ) -> dict[str, dict[str, Any]]:
        """
        Insert new conversation turn including metadata
        into the specified session database table. Return
        the all tool calls metadata.
        """
        metadata = ToolCallsMetadata.all_tool_calls(
            attchmnts=attchmnts,
            unfmt_mem_prompt=unfmt_mem_prompt,
            unfmt_recall_prompt=unfmt_recall_prompt,
            unfmt_cmp_prompt=unfmt_cmp_prompt,
            cmp_method=cmp_method,
        )

        execute(
            conn=self._conn,
            sql=("""
                INSERT INTO chat_logs (session_id, prompt, response, state, total_prompt_tokens, total_output_tokens, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
                """),
            params=(
                self._lookup_chat_sess.get_or_create_id(),
                prompt,
                response,
                state,
                p_tkns,
                o_tkns,
                json.dumps(metadata or {}),
            ),
        )
        return metadata

    def mark_active_as_compressed(self):
        execute(
            conn=self._conn,
            sql=("""
                UPDATE chat_logs
                SET is_compressed = TRUE
                WHERE session_id = %s AND is_compressed = FALSE;
                """),
            params=(self._lookup_chat_sess.get_id(),),
        )

    def clear_session_logs(self) -> bool:
        """Delete all turns of this session. False if there was nothing to delete."""
        deleted = execute(
            conn=self._conn,
            sql="DELETE FROM chat_logs WHERE session_id = %s;",
            params=(self._lookup_chat_sess.get_id(),),
        )

        if deleted == 0:
            app_log.warning(
                "Failed to clear chat logs: Session '%s' does not exist or has no chat logs",
                self._lookup_chat_sess.sess_name,
            )
            return False
        return True
