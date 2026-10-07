from typing import Any, TYPE_CHECKING, Literal
from pathlib import Path

from src.config.models import MODEL
from src.config.prompts import COMPRESS_PROMPT

from .metadata import ToolCallsMetadata
from .write_chat_logs import WriteChatLogs

from src import format_context
from src.agent.tokenizers import LocalTokenizer
from src.libs.slash_commands_dictionary import SLASH_CMDS_DICT
from src.agent.models import llm
from src import logger


if TYPE_CHECKING:
    from .main_module import ChatSessionLogs

app_log = logger.app_logger(f"{__name__}.app")

CompressionResult = tuple[str, int, int, dict[str, dict[str, Any]]]


class Compression:
    """Summarise a session's active conversation turns with model."""

    def __init__(self, chat_sess_logs: "ChatSessionLogs", write_chat_logs: WriteChatLogs):
        self._chat_sess_logs = chat_sess_logs
        self._write_chat_logs = write_chat_logs
        self.local_tknizr = LocalTokenizer(MODEL)

    def compress(
        self,
        prompt: str,
        unfmt_prompt: str,
        attchmnt_dict: dict[Path, dict[str, str]] | None = None,
        cmp_method: Literal["default", "instruct"] | None = None,
    ) -> tuple[str, int, int, dict[str, dict[str, Any]]] | None:
        """
        Call model to summarise all conversations where 
        'is_compressed' = FALSE in the database.
        """
        app_log.info("Compressing session '%s' chat logs", self._chat_sess_logs.sess_name)

        cmbind_prompt = format_context.build_prompt(
            prompt=prompt, attchmnt_dict=attchmnt_dict,
            cmp_convs=self._chat_sess_logs.get_chat_history("not_compressed")
        )

        response = llm.response_with_new_sys_prompt_and_context(
            model=MODEL, sys_prompt=COMPRESS_PROMPT, prompt=cmbind_prompt
        )

        if not response:
            return
        smry, p_tkns, o_tkns = response

        app_log.debug("Chat compression complete. Updating chat logs metadata")

        self._write_chat_logs.mark_active_as_compressed()

        metadata = self._chat_sess_logs.add_conversation_turn(
            prompt=prompt, response=smry, state="external", unfmt_cmp_prompt=unfmt_prompt,
            cmp_method=cmp_method, p_tkns=p_tkns, o_tkns=o_tkns
        )
        app_log.info(
            "Chat compression and chat logs metadata has been updated for session '%s'",
            self._chat_sess_logs.sess_name
        )

        # Update session
        self._chat_sess_logs.update_session_used_tokens()

        return smry, p_tkns, o_tkns, metadata

    def auto(self, unfmt_prompt: str) -> CompressionResult | None:
        """Auto compress session."""
        app_log.info("Chat compression was triggered for session '%s'", self._chat_sess_logs.sess_name)
        prompt = SLASH_CMDS_DICT["compress"].get("default_prompt") or ""
        result = self.compress(prompt=prompt, unfmt_prompt=unfmt_prompt, cmp_method="default")
        if not result:
            return
        app_log.info("Auto compression complete. Continuing session")
        return result
