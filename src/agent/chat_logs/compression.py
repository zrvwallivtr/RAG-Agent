from typing import Any, TYPE_CHECKING

from src.config import models, prompts

from src import format_context
from src.agent.models import llm
from .metadata import ToolCallsMetadata
from .chat_logs_sql import ChatLogsSQL
from src import logger


if TYPE_CHECKING:
    from .main_module import ChatSessionLogs

app_log = logger.app_logger(f"{__name__}.app")

MODEL           = models.MODEL
COMPRESS_PROMPT = prompts.COMPRESS_PROMPT

CompressionResult = tuple[str, int, int, dict[str, dict[str, Any]]]


class Compression:
    """Summarise a session's active conversation turns with model."""

    def __init__(self, chat_sess_logs: "ChatSessionLogs", chat_logs: ChatLogsSQL):
        self._chat_sess_logs = chat_sess_logs
        self._chat_logs = chat_logs

    def compress(
        self,
        prompt: str,
        contxt: list[dict] | None = None
    ) -> tuple[str, int, int, dict[str, dict[str, Any]]] | None:
        """
        Call model to summarise all conversations where 
        'is_compressed' = FALSE in the database.
        """
        app_log.info("Compressing session '%s' chat logs", self._chat_sess_logs.sess_name)

        cmbind_prompt = format_context.build_prompt(
            prompt=prompt, cmp_convs=self._chat_sess_logs.get_chat_history("not_compressed")
        )

        response = llm.response_with_new_sys_prompt_and_context(
            model=MODEL, sys_prompt=COMPRESS_PROMPT, contxt=contxt, prompt=cmbind_prompt,
        )
        if not response:
            return
        smry, p_tkns, o_tkns = response

        app_log.debug("Chat compression complete. Updating chat logs metadata")

        self._chat_logs.mark_active_as_compressed()

        metadata = self._chat_sess_logs.add_conversation_turn(
            prompt=prompt,
            response=smry,
            state="external",
            p_tkns=p_tkns,
            o_tkns=o_tkns
        )
        app_log.info(
            "Chat compression and chat logs metadata has been updated for session '%s'",
            self._chat_sess_logs.sess_name
        )
        return smry, p_tkns, o_tkns, metadata

    def auto(self) -> CompressionResult | None:
        """Auto compress session."""
        app_log.info("Chat compression was triggered for session '%s'", self._chat_sess_logs.sess_name)
        prompt = "Summarise all previous conversations."
        result = self.compress(prompt)
        if not result:
            return
        app_log.info("Auto compression complete. Continuing session")
        return result
