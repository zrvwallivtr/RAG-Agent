from pathlib import Path
from typing import Any

from textual.containers import VerticalScroll
from textual.widgets import Markdown

from src.config.models import MODEL

from src.libs.slash_commands_dictionary import SLASH_CMDS_DICT
from src.libs.helpers.tui_helpers import agent_interface_helper

from src.agent.chat_session_logs import ChatSessionLogs
from src.agent.chat_session_logs.compression import CompressionResult
from src.rag.knowledge_base import DocumentKnowledgeBase
from src import format_context

from src.logger import app_logger

app_log = app_logger(f"{__name__}.app")


class SlashCompress:
    """
    '/compress' calls model to summaries all active conversations (marked as is_compressed = FALSE).

    All conversation turns will be updated to compressed (is_compressed = TRUE)
    NOT INCLUDING the newly generated summary.
    """

    def __init__(self, conn, chat_logs: ChatSessionLogs):
        self.conn       = conn

        self.chat_sess_logs  = chat_logs
        self.doc_kw_bs = DocumentKnowledgeBase(
            conn=self.conn, chat_sess_logs=self.chat_sess_logs
        )

    def _notify_ignore_attachments(
        self, paths: list[Path] | None, in_tui: bool | None = None, screen: Any | None = None
    ):
        """Log and send notifications to user (if in TUI) if file is attached."""
        if paths:
            warn_str = SLASH_CMDS_DICT["compress"].get("no_prompt_attachments_warn") or ""
            app_log.warning(warn_str)
            if in_tui:
                agent_interface_helper.show_info(screen=screen, info=warn_str)
            return

    def _show_compressing_indicator(
        self,
        in_tui: bool | None = None,
        screen: Any | None = None,
        md_widget: Markdown | None = None,
        chat_container: VerticalScroll | None = None,
    ):
        if in_tui:
            info_str = SLASH_CMDS_DICT["compress"].get("loading_info") or ""
            agent_interface_helper.show_loading_indicator(
                screen=screen, md_widget=md_widget, info=info_str,
                is_scroll_end=True, chat_container=chat_container
            )

    def _run_default_compression(
        self,
        is_attchmnt: bool,
        paths: list[Path] | None,
        unfmt_prompt: str,

        # TUI related
        in_tui: bool | None = None,
        screen: Any | None = None,
        md_widget: Markdown | None = None,
        chat_container: VerticalScroll | None = None,
    ) -> CompressionResult | None:
        """
        No instructions, ONLY '/compress' is entered in the chat, a simple mock user prompt
        will be sent to the model.
        """
        self._notify_ignore_attachments(paths=paths, in_tui=in_tui, screen=screen)
        self._show_compressing_indicator(
            in_tui=in_tui, screen=screen, md_widget=md_widget, chat_container=chat_container
        )

        result = self.chat_sess_logs.auto_compress_active_conversations(unfmt_prompt)

        return result if result else None

    def _run_instructed_compression(
        self,
        prompt: str,
        unfmt_prompt: str,
        is_attchmnt: bool,
        paths: list[Path] | None,

        # TUI related
        in_tui: bool | None = None,
        screen: Any | None = None,
        md_widget: Markdown | None = None,
        chat_container: VerticalScroll | None = None,
    ) -> CompressionResult | None:
        """'/compress' with added prompt to instructe to model what exactly to summaries."""
        attchmnt_dict = self.doc_kw_bs.get_attachments_content(
            is_attchmnt=is_attchmnt, attch_paths=paths
        )
        self._show_compressing_indicator(
            in_tui=in_tui, screen=screen, md_widget=md_widget, chat_container=chat_container
        )

        result = self.chat_sess_logs.compress_active_conversations(
            prompt=prompt, unfmt_prompt=unfmt_prompt,
            attchmnt_dict=attchmnt_dict, cmp_method="instruct"
        )

        if not result:
            return None

        # Store attachments
        if attchmnt_dict:
            self.doc_kw_bs.store_attachments(attchmnt_dict)

        return result


    def cmd_compress(
        self,
        prompt: str,
        unfmt_prompt: str,
        is_attchmnt: bool,
        paths: list[Path] | None,

        # TUI related
        in_tui: bool | None = None,
        screen: Any | None = None,
        md_widget: Markdown | None = None,
        chat_container: VerticalScroll | None = None,
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
        if not prompt:
            result = self._run_default_compression(
                is_attchmnt=is_attchmnt, paths=paths, unfmt_prompt=unfmt_prompt, in_tui=in_tui,
                screen=screen, md_widget=md_widget, chat_container=chat_container
            )
            if not result:
                return

        else:
            result = self._run_instructed_compression(
                prompt=prompt, unfmt_prompt=unfmt_prompt, is_attchmnt=is_attchmnt, paths=paths,
                in_tui=in_tui, screen=screen, md_widget=md_widget, chat_container=chat_container
            )
            if not result:
                return

        return result
