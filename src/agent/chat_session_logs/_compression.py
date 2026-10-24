from typing import Any, TYPE_CHECKING, Literal, Callable
from pathlib import Path
from dataclasses import replace

from src.config.config_toml import MODEL
from src.config.prompts import COMPRESS_PROMPT, SYS_PROMPT

from ._metadata import ToolCallsMetadata
from ._chat_logs import ChatLogs

from src.libs.custom_dataclasses import AgentHooks
from src.libs.constants import (
    CompressionResult,
    ModelResult,
    DEFAULT_PROMPT,
    COMPRESS_METHODS,
)
from src.libs.helpers.format_context_helper import build_prompt

from src.agent.tokenizers import LocalTokenizer
from src.agent.models import llm
from src import logger

if TYPE_CHECKING:
    from .chat_session_logs import ChatSessionLogs

app_log = logger.app_logger(f"{__name__}.app")


class Compression:
    """Summarise a session's active conversation turns with model."""

    def __init__(
        self,
        chat_sess_logs: "ChatSessionLogs",
        chat_logs: ChatLogs,
        llm_tknizr: LocalTokenizer,
    ):
        self._chat_sess_logs = chat_sess_logs
        self._chat_logs = chat_logs
        self.llm_tknizr = llm_tknizr

    def _model_compress_with_given_context(
        self,
        prompt: str,
        attchmnt_dict: dict[Path, dict[str, str]] | None = None,
        model: str = MODEL,
        cmp_prompt: str = COMPRESS_PROMPT,
        hooks: AgentHooks | None = None,
    ) -> ModelResult | None:
        """
        Model reads chats from current session and generate a response with
        the compression system prompt.
        """
        cmbind_prompt = build_prompt(
            prompt=prompt,
            attchmnt_dict=attchmnt_dict,
            cmp_convs=self._chat_sess_logs.get_chat_history("not_compressed"),
        )
        response = llm.model_response_with_new_system_prompt_and_context(
            model=model,
            sys_prompt=cmp_prompt,
            prompt=cmbind_prompt,
            hooks=hooks,
        )
        return response

    def compress(
        self,
        clean_prompt: str,
        unfmt_prompt: str,
        cmp_method: COMPRESS_METHODS,
        attchmnt_dict: dict[Path, dict[str, str]] | None = None,
        hooks: AgentHooks | None = None,
    ) -> CompressionResult | None:
        """
        The main compression function. The model generates summary, update
        previous conversation metadata and add the new response to the
        database.
        """
        # Auto compression
        is_auto = cmp_method == "auto"
        cmp_hooks = hooks
        if is_auto and hooks and hooks.on_cmp_start and hooks.on_cmp_tkn:
            hooks.on_cmp_start()
            cmp_hooks = replace(hooks, on_tkn=hooks.on_cmp_tkn, on_first_tkn=None)

        # Message depends on if compression is triggered automatically
        if hooks and hooks.on_loading:
            hooks.on_loading(
                "Compressing due to token limit reached..."
                if is_auto
                else "Compressing session..."
            )

        response = self._model_compress_with_given_context(
            clean_prompt, attchmnt_dict, hooks=cmp_hooks
        )
        if not response:
            return
        smry, p_tkns, o_tkns, is_stopped = response

        # Stop model response if requested
        if is_stopped is True:
            return None

        # Update database
        self._chat_logs.mark_active_as_compressed()
        metadata = self._chat_sess_logs.add_conversation_turn(
            prompt=clean_prompt,
            unfmt_cmp_prompt=unfmt_prompt,
            response=smry,
            state="external",
            cmp_method=cmp_method,
            p_tkns=p_tkns,
            o_tkns=o_tkns,
        )
        self._chat_sess_logs.update_session_used_tokens()

        if is_auto and hooks and hooks.on_cmp_done:
            hooks.on_cmp_done(p_tkns + o_tkns)

        return smry, p_tkns, o_tkns, metadata

    def compress_with_pre_written_prompt(
        self,
        unfmt_prompt: str,
        cmp_method: Literal["default", "auto"],
        attchmnt_dict: dict[Path, dict[str, str]] | None = None,
        default_prompt: str = DEFAULT_PROMPT,
        hooks: AgentHooks | None = None,
    ) -> CompressionResult | None:
        """Compress session using the pre-written prompt."""
        return self.compress(
            clean_prompt=default_prompt,
            unfmt_prompt=unfmt_prompt,
            cmp_method=cmp_method,
            attchmnt_dict=attchmnt_dict,
            hooks=hooks,
        )

    def auto_compress_active_conversations(
        self,
        unfmt_prompt: str,
        default_prompt: str = DEFAULT_PROMPT,
        hooks: AgentHooks | None = None,
    ) -> CompressionResult | None:
        """Auto compress session, triggered when session token count reached pre-set limit."""
        app_log.debug(
            "Auto chat compression was triggered for session '%s'",
            self._chat_sess_logs.sess_name,
        )
        result = self.compress(
            clean_prompt=default_prompt,
            unfmt_prompt=unfmt_prompt,
            cmp_method="auto",
            hooks=hooks,
        )
        if not result:
            return

        if hooks and hooks.on_stat_change:
            info_msg = "Compression completed. Continuing session..."
            hooks.on_stat_change(info_msg)
        return result
