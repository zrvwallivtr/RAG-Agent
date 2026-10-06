from re import search
from pathlib import Path
from typing import Any

from textual.containers import VerticalScroll
from textual.widgets import Markdown

from src.config.models import MODEL
from src.config.prompts import MEM_RECALL_INTERPRET_PROMPT

from .slash_commands_dictionary import SLASH_CMDS_DICT

from src.libs.helpers.tui_helpers import agent_interface_helper

from src.agent.chat_session_logs import ChatSessionLogs
from src.agent.models.llm import response_memory_recall_format
from src.agent.models.embed import embedding_content
from src.rag.memory import Memory
from src.rag.knowledge_base import KnowledgeBase, DocumentKnowledgeBase
from src import format_context

from src.logger import app_logger


app_log = app_logger(f"{__name__}.app")


class SlashRecall:

    def __init__(
        self,
        conn,
        chat_logs: ChatSessionLogs,
        sess_name: str | None = None,
        project: str | None = None
    ):
        self.conn       = conn
        self.sess_name  = sess_name.strip() if sess_name else "default_session"
        self.project    = project
        self.chat_sess_logs  = chat_logs

        self.mem = Memory(conn=self.conn, chat_sess_logs=self.chat_sess_logs)
        self.kw_bs = KnowledgeBase(conn=self.conn, chat_sess_logs=self.chat_sess_logs)
        self.doc_kw_bs = DocumentKnowledgeBase(conn=self.conn, chat_sess_logs=self.chat_sess_logs)

    def cmd_recall(
        self, prompt: str, is_attchmnt: bool, paths: list[Path] | None
    ) -> tuple[str, int, int, dict[str, dict[str, Any]]] | None:
        """
        Retrieve and print relevant entries according to user prompt.

        RECALL PROMPT INCLUDES:
        - User prompt:
          -> Main reference for what to recall
        """
        if not prompt:
            app_log.warning(SLASH_CMDS_DICT["recall"].get("no_prompt_warn"))
            return

        if is_attchmnt and paths:
            app_log.warning(SLASH_CMDS_DICT["recall"].get("attachments_warn"))

        msgs = self.chat_sess_logs.get_active_conversations()

        # Retrieve memory from database
        embed_response = embedding_content(prompt)
        if not embed_response:
            return
        prompt, prompt_embdings, emb_tkns = embed_response

        mem_list = self.mem.query_similar_content(qry=prompt, qry_embdings=prompt_embdings)

        # Interpret recalled memories
        ans, p_tkns, o_tkns = response_memory_recall_format(
            model=MODEL,
            sys_prompt=MEM_RECALL_INTERPRET_PROMPT,
            prompt=prompt,
            context=msgs
        )

        # Save messages
        metadata = self.chat_sess_logs.add_conversation_turn(
            prompt=prompt,
            response=ans,
            state="external",
            p_tkns=p_tkns,
            o_tkns=o_tkns
            # /// Add embed tokens log ///
        )
        return ans, p_tkns, o_tkns, metadata
