from textual.widgets import Markdown

from pathlib import Path
from typing import Callable, Any

from src.config.postgres import conn
from src.config.models import MODEL

from src.libs.helpers import chat_database_helpers, slash_commands_helpers

from src.agent.chat_logs import ChatSessionLogs
from src.rag.knowledge_base import KnowledgeBase
from src.agent.tokenizers import LocalTokenizer
from src.rag.memory import Memory
from src.slash_commands import SlashMemories, SlashRecall, SlashCompress, SLASH_CMDS_DICT
from src import format_context

from src.agent import (
    llm,
    embed,
    chat_logs,
    tokenizers
)
from src.rag import (
    memory,
    knowledge_base,
    DocumentKnowledgeBase
)

from src import logger

app_log = logger.app_logger(f"{__name__}.app")

TOKEN_RESERVE = 1000


class Agent:
    def __init__(
        self,
        sess_name: str | None = None,
    ):
        self.sess_name = sess_name
        self.conn      = conn

        # Main classes
        self.chat_logs = ChatSessionLogs(conn=self.conn, sess_name=self.sess_name)
        self.mem = Memory(conn=self.conn, chat_logs=self.chat_logs)
        self.kw_bs = KnowledgeBase(conn=self.conn, chat_logs=self.chat_logs)
        self.doc_kw_bs = DocumentKnowledgeBase(conn=self.conn, chat_logs=self.chat_logs)
        self.local_tknizr = LocalTokenizer(MODEL)

        # Slash command classes
        self.slash_memories = SlashMemories(
            conn=self.conn, chat_logs=self.chat_logs, sess_name=self.sess_name
        )
        self.slash_recall = SlashRecall(
            conn=self.conn, chat_logs=self.chat_logs, sess_name=self.sess_name
        )
        self.slash_compress = SlashCompress(
            conn=self.conn, chat_logs=self.chat_logs, sess_name=self.sess_name
        )


    def _is_over_budget(self, msgs: list[dict], prompt: str, reserve: int) -> bool:
        """
        True if the history plus the new prompt plus the reply reserve would
        exceed the model's token limit.
        """
        max_tkns = self.local_tknizr.model_max_tkns
        if not max_tkns:
            app_log.warning("Failed to load 'manage token budget feature': Model maximum token limit not set")
            return False

        hist_tkns = self.local_tknizr.count_history_tokens(msgs)
        if not hist_tkns:
            return False

        # Actual result
        est_next = len(prompt) // 4
        return hist_tkns + est_next + reserve > max_tkns


    def _compress_if_over_budget(self, prompt: str) -> list[dict]:
        """Compress the session if history + prompt + reserve exceeds the limit. Returns the history."""
        msgs = self.chat_logs.get_active_conversations()
        if self._is_over_budget(msgs=msgs, prompt=prompt, reserve=TOKEN_RESERVE):
            app_log.info("Current tokens exceeds threshold")
            self.chat_logs.auto_compress_active_conversations()
            msgs = self.chat_logs.get_active_conversations()
        return msgs


    def _route_to_slash_command(
        self,
        msgs: list[dict],
        cmd: str,
        user_prompt: str,
        is_attchmnt: bool,
        paths: list[Path] | None = None,

        # TUI related
        in_tui: bool | None = None,
        screen: Any | None = None,
        md_widget: Markdown | None = None
    ) -> tuple[str, int, int, dict[str, dict[str, Any]]] | None:
        """Route to slash command for '/memorise', '/recall' and '/compress'."""
        if cmd == SLASH_CMDS_DICT["memorise"].get("cmd"):
            app_log.debug("'%s' command triggered", cmd)
            result = self.slash_memories.cmd_memorise(
                prompt=user_prompt, is_attchmnt=is_attchmnt, paths=paths
            )
            return result if result else None

        if cmd == SLASH_CMDS_DICT["recall"].get("cmd"):
            app_log.debug("'%s' command triggered", cmd)
            result = self.slash_recall.cmd_recall(
                prompt=user_prompt, is_attchmnt=is_attchmnt, paths=paths
            )
            return result if result else None

        if cmd == SLASH_CMDS_DICT["compress"].get("cmd"):
            app_log.debug("'%s' command triggered", cmd)
            result = self.slash_compress.cmd_compress(
                prompt=user_prompt,
                is_attchmnt=is_attchmnt,
                paths=paths,
                in_tui=in_tui,
                screen=screen,
                md_widget=md_widget
            )
            return result if result else None


    def _embedding_content_retrieval_controller(
        self, prompt: str, is_auto_mem_rtve: bool, is_auto_doc_rtve: bool
    ) -> tuple[str, int, list[float], bool, bool]:
        """
        Return embeddings for user prompt for memory and document content retrieval
        features, if no embedding failed turn off related featurs.
        """
        embed_response = embed.embedding_content(prompt)

        if embed_response:
            prompt, prompt_embdings, prompt_tkns = embed_response

        else:
            app_log.warning(
                "Turning off features that requires embeddings: "
                "Auto memory retrieval, Auto document content retrieval"
            )
            prompt_embdings = [0.0]
            prompt_tkns = 0
            is_auto_mem_rtve = False # Turn off auto memory retrieve
            is_auto_doc_rtve = False # Turn off auto document chunk retrieve

        return prompt, prompt_tkns, prompt_embdings, is_auto_mem_rtve, is_auto_doc_rtve


    def _combine_context(
        self,
        prompt: str,
        prompt_embdings: list[float],
        is_auto_mem_rtve: bool,
        is_auto_doc_rtve: bool,
        is_attchmnt: bool,
        paths: list[Path] | None = None
    ) -> tuple[str, dict] | tuple[str, None]:
        # Auto retrieve relevant memories
        mem_list = self.mem.toggle_auto_retrive_memory_entries(
            is_auto_mem_rtve=is_auto_mem_rtve, prompt=prompt, prompt_embdings=prompt_embdings
        )

        # Auto retrieve relevant session documents
        doc_list = self.doc_kw_bs.toggle_auto_retrieve_sess_docs(
            is_auto_doc_rtve=is_auto_doc_rtve, prompt=prompt, prompt_embdings=prompt_embdings
        )

        # Uploaded attachments (optional)
        attchmnt_dict = self.doc_kw_bs.get_attachments_content(
            is_attchmnt=is_attchmnt, attch_paths=paths
        )

        # All context combined (won't be saved to chat history)
        cmbind_prompt = format_context.build_prompt(
            prompt=prompt, mem_list=mem_list, doc_list=doc_list, attchmnt_dict=attchmnt_dict
        )

        if attchmnt_dict:
            return cmbind_prompt, attchmnt_dict
        return cmbind_prompt, None


    def ask(
        self,
        prompt: str,
        is_auto_mem_rtve: bool = True,
        is_auto_mem_store: bool = False,
        is_auto_doc_rtve: bool = True,
        is_auto_web_sear: bool = False,
        is_attchmnt: bool = False,
        callback: Callable[[str], None] | None = None,
        paths: list[Path] | None = None,

        # TUI related
        in_tui: bool | None = None,
        screen: Any | None = None,
        md_widget: Markdown | None = None
    ) -> tuple[str, int, int, dict[str, dict[str, Any]]] | None:
        """
        Model decide what memories to read.
        Manage tokens, compress session if needed.

        Note:
        - Only the user question and LLM response will be
          stored into chat history.
        """
        msgs = self.chat_logs.get_active_conversations()

        # Slash commands
        cmd, user_prompt = slash_commands_helpers.detect_cmd(prompt)
        if cmd:
            # Run budget check and auto compress only when slash command is
            # not '/compress' as it is already a session compression tool call.
            if cmd != SLASH_CMDS_DICT["compress"].get("cmd"):
                msgs = self._compress_if_over_budget(user_prompt)

            return self._route_to_slash_command(
                msgs=msgs,
                cmd=cmd,
                user_prompt=user_prompt,
                is_attchmnt=is_attchmnt,
                paths=paths,
                in_tui=in_tui,
                screen=screen,
                md_widget=md_widget
            )

        prompt_tkns, prompt_tkns, prompt_embdings,is_auto_mem_rtve, is_auto_doc_rtve = (
            self._embedding_content_retrieval_controller(
                prompt=prompt,
                is_auto_mem_rtve=is_auto_mem_rtve,
                is_auto_doc_rtve=is_auto_doc_rtve
            )
        )

        # Combine all context, check token count and bundle them
        # into messages for model to answer.
        combine_prompt, attchmnt_dict = self._combine_context(
            prompt=prompt,
            prompt_embdings=prompt_embdings,
            is_auto_mem_rtve=is_auto_mem_rtve,
            is_auto_doc_rtve=is_auto_doc_rtve,
            is_attchmnt=is_attchmnt,
            paths=paths
        )
        msgs = self._compress_if_over_budget(combine_prompt)
        msgs.append(llm.user_message(combine_prompt))

        # Model answer
        response = llm.model_response(model=MODEL, msgs=msgs, callback=callback)
        if not response:
            return
        ans, p_tkns, o_tkns = response

        # Calculate total tokens
        total_p_tkns = prompt_tkns + p_tkns
        total_o_tkns = o_tkns

        # Save messages to database
        metadata = self.chat_logs.add_conversation_turn(
            prompt=prompt,
            response=ans,
            state="external",
            attchmnts=paths,
            p_tkns=total_p_tkns,
            o_tkns=total_o_tkns
        )

        # Update chat session table in database
        with self.conn.cursor() as cur:
            chat_database_helpers.update_session_used_tokens_on_prompt_submitted(
                chat_logs=self.chat_logs, local_tknizr=self.local_tknizr, cur=cur
            )

        # Store memory
        # self.memory.toggle_auto_store_memory_entries(
        #     is_auto_mem_store=is_auto_mem_store,
        #     model_max_tokens=self.get_model_max_tokens,
        #     context=self.chat.to_llm()
        # )

        # Store attachment(s)
        if attchmnt_dict:
            self.doc_kw_bs.store_attachments(attchmnt_dict)
        return ans, total_p_tkns, total_o_tkns, metadata
