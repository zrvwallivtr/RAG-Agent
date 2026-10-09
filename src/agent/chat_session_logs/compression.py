from typing import Any, TYPE_CHECKING, Literal, Callable
from pathlib import Path

from src.config.models import MODEL
from src.config.prompts import COMPRESS_PROMPT

from .metadata import ToolCallsMetadata
from .write_chat_logs import WriteChatLogs

from src.libs.slash_commands_dictionary import SLASH_CMDS_DICT
from src.libs.helpers.format_context_helper import build_prompt

from src.agent.tokenizers import LocalTokenizer
from src.agent.models import llm
from src.agent.models.llm import ModelResult, TokenCallback
from src import logger

if TYPE_CHECKING:
    from .main_module import ChatSessionLogs

app_log = logger.app_logger(f"{__name__}.app")

CompressionResult = tuple[str, int, int, dict[str, dict[str, Any]]]

def _default_compression_prompt() -> str | None:
    prompt = SLASH_CMDS_DICT["compress"].get("default_prompt")
    if not prompt:
        app_log.warning(
            "Pre-written compression instruction prompt not found "
            "in code base"
        )
    return prompt

class Compression:
    """Summarise a session's active conversation turns with model."""

    def __init__(
        self,
        chat_sess_logs: "ChatSessionLogs",
        write_chat_logs: WriteChatLogs
    ):
        self._chat_sess_logs = chat_sess_logs
        self._write_chat_logs = write_chat_logs
        self.local_tknizr = LocalTokenizer(MODEL)

    def _model_compress_with_given_context(
        self,
        prompt: str,
        attchmnt_dict: dict[Path, dict[str, str]] | None = None,
        callback: TokenCallback | None = None
    ) -> ModelResult | None:
        """
        Model reads chats from current session and generate a response with
        the compression system prompt.
        """
        cmbind_prompt = build_prompt(
            prompt=prompt,
            attchmnt_dict=attchmnt_dict,
            cmp_convs=self._chat_sess_logs.get_chat_history("not_compressed")
        )
        response = llm.model_response_with_new_system_prompt_and_context(
            model=MODEL,
            sys_prompt=COMPRESS_PROMPT,
            prompt=cmbind_prompt,
            callback=callback
        )
        return response

    def compress(
        self,
        clean_prompt: str,
        unfmt_prompt: str,
        cmp_method: Literal["default", "instruct", "auto"],
        attchmnt_dict: dict[Path, dict[str, str]] | None = None,
        callback: TokenCallback | None = None,
    ) -> tuple[str, int, int, dict[str, dict[str, Any]]] | None:
        """
        The main compression function. The model generates summary, update
        previous conversation metadata and add the new response to the
        database.
        """
        app_log.info(
            "Compressing session '%s' chat logs",
            self._chat_sess_logs.sess_name
        )
        response = self._model_compress_with_given_context(
            clean_prompt, attchmnt_dict, callback
        )
        if not response:
            return
        smry, p_tkns, o_tkns, is_stopped = response

        # Stop model response if requested
        if is_stopped is True:
            return None

        # Update database
        self._write_chat_logs.mark_active_as_compressed()
        metadata = self._chat_sess_logs.add_conversation_turn(
            prompt=clean_prompt,
            unfmt_cmp_prompt=unfmt_prompt,
            response=smry,
            state="external",
            cmp_method=cmp_method,
            p_tkns=p_tkns,
            o_tkns=o_tkns
        )
        self._chat_sess_logs.update_session_used_tokens()

        return smry, p_tkns, o_tkns, metadata

    def compress_with_pre_written(
        self,
        unfmt_prompt: str,
        cmp_method: Literal["default", "auto"],
        attchmnt_dict: dict[Path, dict[str, str]] | None = None,
        callback: TokenCallback | None = None
    ) -> CompressionResult | None:
        """Compress session using the pre-written prompt."""
        return self.compress(
            clean_prompt=_default_compression_prompt() or "",
            unfmt_prompt=unfmt_prompt,
            cmp_method=cmp_method,
            attchmnt_dict=attchmnt_dict,
            callback=callback
        )

    def auto_compress(
        self,
        unfmt_prompt: str,
    ) -> CompressionResult | None:
        """Auto compress session, triggered when session token count reached pre-set limit."""
        app_log.info(
            "Auto chat compression was triggered for session '%s'",
            self._chat_sess_logs.sess_name
        )
        result = self.compress(
            clean_prompt=_default_compression_prompt() or "",
            unfmt_prompt=unfmt_prompt,
            cmp_method="auto",
        )
        if not result:
            return
        app_log.info("Compression completed. Continuing session")
        return result
