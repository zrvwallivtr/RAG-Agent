from pathlib import Path
from typing import Any, Callable

from agent import chat_session_logs
from rag import file_reader
from rag.docling import chunker
from src.config.config_toml import MODEL

from src.libs.constants import COMPRESS_CMD
from src.libs.custom_dataclasses import AgentHooks
from src.libs.constants import AgentResult

from src.agent.chat_session_logs import ChatSessionLogs

from src.rag.attach_files import Attachments, QueryAttachments
from src.rag.file_reader import FileReader
from src.rag.docling import DoclingChunker

from src.logger import app_logger

app_log = app_logger(f"{__name__}.app")


class SlashCompress:
    """
    '/compress' calls model to summaries all active conversations (marked
    as is_compressed = FALSE).

    All conversation turns will be updated in the database as
    is_compressed = TRUE. NOT INCLUDING the newly generated summary.
    """

    def __init__(
        self,
        conn,
        chat_sess_logs: ChatSessionLogs,
        file_reader: FileReader,
        chunker: DoclingChunker,
    ):
        self.conn = conn
        self.chat_sess_logs = chat_sess_logs
        self.attchmnts = Attachments(
            conn=self.conn,
            chat_sess_logs=self.chat_sess_logs,
            file_reader=file_reader,
            chunker=chunker,
        )
        self.qry_attchmnts = QueryAttachments(
            conn=self.conn,
            chat_sess_logs=self.chat_sess_logs,
        )
        self.file_reader = file_reader

    def _default_compression(
        self,
        unfmt_prompt: str,
        paths: list[Path] | None,
        cmp_cmd: str = COMPRESS_CMD,
        hooks: AgentHooks | None = None,
    ) -> AgentResult | None:
        """
        No instructions, ONLY '/compress' is entered in the chat, a simple
        pre-written user prompt will be sent to the model.

        This function does NOT support attachment uploads, a message will
        be displayed to notify user that the uploaded will be ignored.

        Parameters
        ----------
            unfmt_prompt: The original prompt including the slash command, used
                          for logging chat history.
            paths:        New attachments paths, output a warning message if
                          present to inform this slash command does not support
                          the feature.
            cmp_cmd:      The exact slash command to trigger compression.
            hooks:        Functions hooks for cli/tui integration.

        Returns
        -------
            - The response by the model.
            - Tokens used from the prompt.
            - Tokens used during the response generation.
            - Metadata of the new conversation turn that was saved to the database.
        """
        if paths and hooks and hooks.on_stat_change:
            warn_msg = (
                "Attachments ignored: '%s' does not support attachments"
                % cmp_cmd
            )
            app_log.warning(warn_msg)
            hooks.on_stat_change(warn_msg)

        result = self.chat_sess_logs.default_compress_active_conversations(
            unfmt_prompt, "default", hooks=hooks
        )
        return result if result else None

    def _instructed_compression(
        self,
        clean_prompt: str,
        unfmt_prompt: str,
        paths: list[Path] | None,
        cmp_cmd: str = COMPRESS_CMD,
        hooks: AgentHooks | None = None,
    ) -> AgentResult | None:
        """
        Compress session with added prompt to instructe the model
        what exactly to summaries.

        Parameters
        ----------
            clean_prompt: The user's prompt without the slash command, which will
                          be sent to the model for generating a response.
            unfmt_prompt: The original prompt including the slash command, used
                          for logging chat history.
            paths:        New attachments paths, output a warning message if
                          present to inform this slash command does not support
                          the feature.
            cmp_cmd:      The exact slash command to trigger compression.
            hooks:        Functions hooks for cli/tui integration.

        Returns
        -------
            - The response by the model.
            - Tokens used from the prompt.
            - Tokens used during the response generation.
            - Metadata of the new conversation turn that was saved to the database.
        """
        if paths and hooks and hooks.on_stat_change:
            warn_msg = (
                "Attachments ignored: '%s' does not support attachments"
                % cmp_cmd
            )
            app_log.warning(warn_msg)
            hooks.on_stat_change(warn_msg)

        result = self.chat_sess_logs.instruct_compress_active_conversations(
            clean_prompt=clean_prompt,
            unfmt_prompt=unfmt_prompt,
            cmp_method="instruct",
            hooks=hooks,
        )
        return result

    def run_compress(
        self,
        clean_prompt: str,
        unfmt_prompt: str,
        paths: list[Path] | None,
        hooks: AgentHooks | None = None,
    ) -> AgentResult | None:
        """
        Route to default compression process if only '/compress' is sent to the 
        agent.

        Parameters
        ----------
            clean_prompt: The user's prompt without the slash command, which will
                          be sent to the model for generating a response.
            unfmt_prompt: The original prompt including the slash command, used
                          for logging chat history.
            paths:        New attachments paths, output a warning message if
                          present to inform this slash command does not support
                          the feature.
            hooks:        Functions hooks for cli/tui integration.

        Returns
        -------
            - The response by the model.
            - Tokens used from the prompt.
            - Tokens used during the response generation.
            - Metadata of the new conversation turn that was saved to the database.
        """
        if not clean_prompt:
            return self._default_compression(
                paths=paths,
                unfmt_prompt=unfmt_prompt,
                hooks=hooks,
            )

        return self._instructed_compression(
            clean_prompt=clean_prompt,
            unfmt_prompt=unfmt_prompt,
            paths=paths,
            hooks=hooks,
        )
