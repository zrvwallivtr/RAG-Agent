from pathlib import Path
from typing import Any, Callable

from src.config.models import MODEL

from src.libs.helpers.tui_helpers.agent_interface_helper import (
    show_loading_indicator, show_info
)
from src.libs.constants import COMPRESS_CMD, DEFAULT_PROMPT, NO_PROMPT_WARN, GenericEventHook
from src.libs.custom_dataclasses import AgentHooks

from src.agent.chat_session_logs import ChatSessionLogs
from src.agent.chat_session_logs.compression import CompressionResult
from src.rag.knowledge_base import DocumentKnowledgeBase

from src.logger import app_logger

app_log = app_logger(f"{__name__}.app")


# --- Frontend helpers ----------------------------------
# 
# def _notify_ignore_attachments(
#     paths: list[Path] | None,
#     in_tui: bool | None = None,
#     screen: Any | None = None
# ) -> None:
#     """
#     Handles notifications depend on where the TUI is running if file is attached.
#     Call this for default compression.
#     """
#     warn_str = NO_PROMPT_ATTCHMNT_WARN
# 
#     if in_tui and paths:
#         app_log.warning(warn_str)
#         if in_tui:
#             show_info(screen=screen, info=warn_str)
#         return None
# 
#     if paths:
#         app_log.warning(warn_str)
# 
# 
# def _show_compressing_indicator(
#     in_tui: bool | None = None,
#     screen: Any | None = None,
#     md_widget: Markdown | None = None,
#     chat_container: VerticalScroll | None = None,
# ):
#     if in_tui:
#         show_loading_indicator(
#             screen=screen,
#             md_widget=md_widget,
#             info=,
#             is_scroll_end=True,
#             chat_container=chat_container
#         )


class SlashCompress:
    """
    '/compress' calls model to summaries all active conversations (marked
    as is_compressed = FALSE).

    All conversation turns will be updated to compressed
    (is_compressed = TRUE). NOT INCLUDING the newly generated summary.
    """

    def __init__(self, conn, chat_logs: ChatSessionLogs):
        self.conn = conn
        self.chat_sess_logs = chat_logs
        self.doc_kw_bs = DocumentKnowledgeBase(
            conn=self.conn, chat_sess_logs=self.chat_sess_logs
        )

    def _default_compression(
        self,
        unfmt_prompt: str,
        is_attchmnt: bool,
        paths: list[Path] | None,
        hooks: AgentHooks | None = None,
    ) -> CompressionResult | None:
        """
        No instructions, ONLY '/compress' is entered in the chat, a simple
        pre-written user prompt will be sent to the model.

        This function does NOT support attachment uploads, a message will
        be displayed to notify user that the uploaded will be ignored.
        """
        if (is_attchmnt or paths) and hooks and hooks.on_stat_change:
            app_log.warning(NO_PROMPT_WARN)
            hooks.on_stat_change(NO_PROMPT_WARN)

        result = self.chat_sess_logs.default_compress_active_conversations(
            unfmt_prompt, "default", hooks=hooks
        )
        return result if result else None

    def _instructed_compression(
        self,
        clean_prompt: str,
        unfmt_prompt: str,
        is_attchmnt: bool,
        paths: list[Path] | None,
        hooks: AgentHooks | None = None
    ) -> CompressionResult | None:
        """'/compress' with added prompt to instructe to model what exactly to summaries."""
        attchmnt_dict = self.doc_kw_bs.get_attachments_content(
            is_attchmnt=is_attchmnt, attch_paths=paths
        )
        if hooks and hooks.on_loading:
            hooks.on_loading("Compressing session...")

        result = self.chat_sess_logs.instruct_compress_active_conversations(
            clean_prompt=clean_prompt,
            unfmt_prompt=unfmt_prompt,
            attchmnt_dict=attchmnt_dict,
            cmp_method="instruct",
            hooks=hooks
        )

        if not result:
            return None

        # Store attachments
        if attchmnt_dict:
            self.doc_kw_bs.store_attachments(attchmnt_dict)

        return result


    def run_compress(
        self,
        clean_prompt: str,
        unfmt_prompt: str,
        is_attchmnt: bool,
        paths: list[Path] | None,
        hooks: AgentHooks | None = None,
    ) -> CompressionResult | None:
        """
        Retrieve and print relevant entries according to user prompt.

        COMPRESS PROMPT INCLUDES:
        - Attachments (optional):
          -> Allows option to upload attachments that might influence chat compression.
        - All previous conversations that labelled as 'is_compressed = FALSE'
          -> Already included in 'compress_active_conv()'.
        - User prompt:
          -> Main instruction on compression focus.
        """
        if not clean_prompt:
            return self._default_compression(
                is_attchmnt=is_attchmnt,
                paths=paths,
                unfmt_prompt=unfmt_prompt,
                hooks=hooks,
            )

        return self._instructed_compression(
            clean_prompt=clean_prompt,
            unfmt_prompt=unfmt_prompt,
            is_attchmnt=is_attchmnt,
            paths=paths,
            hooks=hooks,
        )
