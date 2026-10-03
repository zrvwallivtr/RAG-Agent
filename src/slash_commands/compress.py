from pathlib import Path
from typing import Any

from textual.containers import VerticalScroll
from textual.widgets import Markdown

from src.config import models
from src.config import prompts

from src import format_context
from src.slash_commands import slash_commands_dictionary
from src.agent import (
    chat_logs,
)
from src.rag import document_knowledge_base
from src.app.tui.screens.tui_helpers import agent_interface_helper

from src.logger import app_logger


app_log = app_logger(f"{__name__}.app")

MODEL                       = models.MODEL
SLASH_CMD_DICT = slash_commands_dictionary.slash_cmds_dict

ChatLogs                = chat_logs.ChatLogs
DocumentKnowledgeBase   = document_knowledge_base.DocumentKnowledgeBase


class SlashCompress:
    def __init__(self, conn, chat_logs: ChatLogs, sess_name: str | None = None):
        self.conn       = conn
        self.sess_name  = sess_name.strip() if sess_name else "default_session"
        self.chat_logs  = chat_logs

        self.doc_kw_bs = DocumentKnowledgeBase(
            conn=self.conn, chat_logs=self.chat_logs, sess_name=self.sess_name
        )


    def _get_full_context(self, prompt: str, is_attchmnt: bool, paths: list[Path] | None):
        attchmnt_dict = self.doc_kw_bs.get_attachments_content(
            is_attchmnt=is_attchmnt, attch_paths=paths
        )
        cmbind_prompt = format_context.build_prompt(
            prompt=prompt, attchmnt_dict=attchmnt_dict
        )
        return attchmnt_dict, cmbind_prompt


    def _run_default_compression(
        self,
        prompt: str,
        is_attchmnt: bool,
        paths: list[Path] | None,

        # TUI related
        in_tui: bool | None = None,
        screen: Any | None = None,
        md_widget: Markdown | None = None,
        chat_container: VerticalScroll | None = None,
    ) -> tuple[str, int, int, dict[str, dict[str, Any]]] | None:
        if not prompt:
            # Default compression (no instructions)
            if is_attchmnt and paths:
                warn_str = SLASH_CMD_DICT["compress"].get("no_prompt_attachments_warn") or ""
                app_log.warning(warn_str)
                if in_tui:
                    agent_interface_helper.show_info(screen=screen, info=warn_str)
                return

            if in_tui:
                info_str = SLASH_CMD_DICT["compress"].get("loading_info") or ""
                agent_interface_helper.show_loading_indicator(
                    screen=screen,
                    md_widget=md_widget,
                    info=info_str,
                    is_scroll_end=True,
                    chat_container=chat_container
                )

            result = self.chat_logs.auto_compresss_active_conv()
            return result if result else None


    def _run_instructed_compression(
        self,
        prompt: str,
        is_attchmnt: bool,
        paths: list[Path] | None,

        # TUI related
        in_tui: bool | None = None,
        screen: Any | None = None,
        md_widget: Markdown | None = None,
        chat_container: VerticalScroll | None = None,
    ) -> tuple[str, int, int, dict[str, dict[str, Any]]] | None:
        attchmnt_dict, cmbind_prompt = self._get_full_context(
            prompt=prompt, is_attchmnt=is_attchmnt, paths=paths
        )

        info_str = SLASH_CMD_DICT["compress"].get("loading_info") or ""

        app_log.info(info_str)

        if in_tui:
            agent_interface_helper.show_loading_indicator(
                screen=screen, md_widget=md_widget, info=info_str
            )

        result = self.chat_logs.compress_active_conv(prompt=cmbind_prompt)
        if not result:
            return

        # Store attachments
        if attchmnt_dict:
            self.doc_kw_bs.store_attachments(attchmnt_dict)

        return result


    def cmd_compress(
        self,
        prompt: str,
        is_attchmnt: bool,
        paths: list[Path] | None,

        # TUI related
        in_tui: bool | None = None,
        screen: Any | None = None,
        md_widget: Markdown | None = None,
        chat_container: VerticalScroll | None = None,
    ) -> tuple[str, int, int, dict[str, dict[str, Any]]] | None:
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
                prompt=prompt,
                is_attchmnt=is_attchmnt,
                paths=paths,
                in_tui=in_tui,
                screen=screen,
                md_widget=md_widget,
                chat_container=chat_container
            )
            if not result:
                return

        else:
            result = self._run_instructed_compression(
                prompt=prompt,
                is_attchmnt=is_attchmnt,
                paths=paths,
                in_tui=in_tui,
                screen=screen,
                md_widget=md_widget,
                chat_container=chat_container
            )
            if not result:
                return

        return result
