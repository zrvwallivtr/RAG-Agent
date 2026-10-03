from re import search
from pathlib import Path
from typing import Any

from textual.containers import VerticalScroll
from textual.widgets import Markdown

from src.config import models
from src.config import prompts

from src import format_context
from src.slash_commands import slash_commands_dictionary
from src.agent import (
    ollama,
    llm,
    embed,
    chat_logs,
)
from src.rag import (
    memory,
    knowledge_base,
    document_knowledge_base,
)
from src.app.tui.screens.tui_helpers import agent_interface_helper

from src.logger import app_logger


app_log = app_logger(f"{__name__}.app")

MODEL                       = models.MODEL
MEM_RECALL_INTERPRET_PROMPT = prompts.MEM_RECALL_INTERPRET_PROMPT

ChatLogs                = chat_logs.ChatLogs
Memory                  = memory.Memory
KnowledgeBase           = knowledge_base.KnowledgeBase
DocumentKnowledgeBase   = document_knowledge_base.DocumentKnowledgeBase

SLASH_CMD_DICT = slash_commands_dictionary.slash_cmds_dict


class SlashRecall:
    def __init__(
        self,
        conn,
        chat_logs: ChatLogs,
        sess_name: str | None = None,
        project: str | None = None
    ):
        self.conn       = conn
        self.sess_name  = sess_name.strip() if sess_name else "default_session"
        self.project    = project
        self.chat_logs  = chat_logs

        self.mem = Memory(
            conn=self.conn, chat_logs=self.chat_logs, project=self.project
        )
        self.kw_bs = KnowledgeBase(
            conn=self.conn, chat_logs=self.chat_logs, sess_name=self.sess_name
        )
        self.doc_kw_bs = DocumentKnowledgeBase(
            conn=self.conn, chat_logs=self.chat_logs, sess_name=self.sess_name
        )


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
            app_log.warning(SLASH_CMD_DICT["recall"].get("no_prompt_warn"))
            return

        if is_attchmnt and paths:
            app_log.warning(SLASH_CMD_DICT["recall"].get("attachments_warn"))

        msgs = self.chat_logs.get_active_conversations()

        # Retrieve memory from database
        embed_response = embed.embedding_content(prompt)
        if not embed_response:
            return
        prompt, prompt_embdings, emb_tkns = embed_response

        mem_list = self.mem.query_similar_content(qry=prompt, qry_embdings=prompt_embdings)

        # Interpret recalled memories
        ans, p_tkns, o_tkns = llm.response_memory_recall_format(
            model=self.model,
            sys_prompt=MEM_RECALL_INTERPRET_PROMPT,
            prompt=prompt,
            context=msgs
        )

        # Save messages
        metadata = self.chat_logs.add_conversation_turn(
            prompt=prompt,
            response=ans,
            state="external",
            p_tkns=p_tkns,
            o_tkns=o_tkns
            # /// Add embed tokens log ///
        )
        return ans, p_tkns, o_tkns, metadata
