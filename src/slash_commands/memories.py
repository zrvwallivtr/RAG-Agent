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
    search_agent,
    query_manager
)
from src.app.tui.screens.helpers import agent_interface_helper
from src.logger import app_logger


app_log = app_logger(f"{__name__}.app")

MODEL                       = models.MODEL
MEM_RECALL_INTERPRET_PROMPT = prompts.MEM_RECALL_INTERPRET_PROMPT

ChatLogs                = chat_logs.ChatLogs
Memory                  = memory.Memory
KnowledgeBase           = knowledge_base.KnowledgeBase
DocumentKnowledgeBase   = document_knowledge_base.DocumentKnowledgeBase
SearchAgent             = search_agent.SearchAgent

SLASH_CMD_DICT = slash_commands_dictionary.slash_cmds_dict


class SlashMemories:
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
        self.sear_agt = SearchAgent(
            conn=self.conn, sess_name=self.sess_name
        )


    def cmd_memorise(
        self, prompt: str, is_attchmnt: bool, paths: list[Path] | None
    ) -> tuple[str, int, int, dict[str, dict[str, Any]]] | None:
        """
        Extract key info from user prompt and attachments (optional),
        save extracted memory entries to database.

        MEMORISE PROMPT INCLUDES:
        - Attachments (optional):
          -> Allows option to memorise contents in uploaded attachments.
        - Previous messages
          -> Already included in 'extract_and_store_mem_from_conv()'
        - User prompt:
          -> Main instruction for memorise command

        NOTE: User's question will be saved directly, this function
              will then generate save a pre-written assistant message.
        """
        if not prompt:
            app_log.warning(
                SLASH_CMD_DICT["memorise"].get("no_prompt_warn")
            )
            return

        msgs = self.chat_logs.get_actv_convs()

        # Full context
        attchmnt_dict = self.doc_kw_bs.get_attachments_content(
            is_attchmnt=is_attchmnt, attch_paths=paths
        )
        cmbind_prompt = format_context.build_prompt(
            prompt=prompt, attchmnt_dict=attchmnt_dict
        )
        app_log.debug("Appended new message to current messages")

        # Extract and store memories
        app_log.info("Extracting content from user's prompt")
        response = self.mem.extract_and_store_mem_from_conv(
            extraction="manual", prompt=cmbind_prompt
        )

        if not response:
            app_log.warning("Failed to extract info from prompt: Model returns nothing")
            return
        created_ids, p_tkns, o_tkns, emb_tkns = response

        mem_dict = self.mem.get_mem_content_from_ids(created_ids)

        # Print to terminal
        print("CONTENT SAVED:")
        for cont, ctgry in mem_dict.items():
            print(f"[{ctgry}] {cont}\n\n")

        # User confirm options
        choice = input("Press [Enter] to continue or type [u] to undo:")
        if choice == "u":
            self.mem.delete_mem(created_ids)
            app_log.debug(SLASH_CMD_DICT["memorise"].get("removed_info"))
            return

        # Save messages
        # mock_resp = "Information has been extracted and added to database."
        metadata = self.chat_logs.add_conv_turn(
            prompt=prompt,
            response=SLASH_CMD_DICT["memorise"].get("mock_response") or "",
            state="external",
            attchmnts=paths,
            p_tkns=p_tkns,
            o_tkns=o_tkns
        )

        # Store attachments
        if attchmnt_dict:
            self.doc_kw_bs.store_attachments(attchmnt_dict)

        return "", p_tkns, o_tkns, metadata
