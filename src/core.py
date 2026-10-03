from pathlib import Path
from typing import Callable, Any

from textual.widgets import Markdown

from src.config.postgres import conn
from src.config import models

from src import format_context
from src.main_helpers import chat_database_helpers
from src.slash_commands import (
    slash_commands_helpers,
    slash_commands_dictionary,
    memories,
    recall,
    compress
)
from src.agent import (
    ollama,
    llm,
    embed,
    chat_logs,
    tokenizers
)
from src.rag import (
    memory,
    knowledge_base,
    document_knowledge_base
)

from src import logger


app_log = logger.app_logger(f"{__name__}.app")

MODEL            = models.MODEL
MODEL_MAX_TOKENS = models.MODEL_MAX_TOKENS
EMBED_MODEL      = models.EMBED_MODEL
EMBED_MAX_TOKENS = models.EMBED_MAX_TOKENS

Tknizr                = tokenizers.Tknizr
ChatLogs              = chat_logs.ChatLogs
Memory                = memory.Memory
KnowledgeBase         = knowledge_base.KnowledgeBase
DocumentKnowledgeBase = document_knowledge_base.DocumentKnowledgeBase

SLASH_CMD_DICT = slash_commands_dictionary.slash_cmds_dict


class Agent:
    def __init__(
        self,
        sess_name: str | None = None,
    ):
        self.sess_name = sess_name
        self.conn      = conn

        # Main classes
        self.chat_logs = ChatLogs(conn=self.conn, sess_name=self.sess_name)
        self.mem = Memory(conn=self.conn, chat_logs=self.chat_logs)
        self.kw_bs = KnowledgeBase(
            conn=self.conn, chat_logs=self.chat_logs, sess_name=self.sess_name
        )
        self.doc_kw_bs = DocumentKnowledgeBase(
            conn=self.conn, chat_logs=self.chat_logs, sess_name=self.sess_name
        )
        self.tknizr = Tknizr(MODEL)

        # Slash command classes
        self.slash_memories = memories.SlashMemories(
            conn=self.conn, chat_logs=self.chat_logs, sess_name=self.sess_name
        )
        self.slash_recall = recall.SlashRecall(
            conn=self.conn, chat_logs=self.chat_logs, sess_name=self.sess_name
        )
        self.slash_compress = compress.SlashCompress(
            conn=self.conn, chat_logs=self.chat_logs, sess_name=self.sess_name
        )


    def _manage_token_budget(self, prompt: str, reserve: int) -> None:
        """
        Reserves extra tokens for model response. If exceeds
        maximum tokens, the model summarise previous messages
        to free up token space.
        """
        app_log.debug("Estimating token usage for session '%s'", self.sess_name)
        curr_hist_tkns = self.tknizr.count_history_tokens(self.chat_logs.get_active_conversations())
        if not curr_hist_tkns:
            return

        est_next = curr_hist_tkns + (len(prompt) // 4)

        if not self.tknizr.model_max_tkns:
            app_log.warning("Failed to load 'manage token budget feature': Model maximum token limit not set")
            return

        if self.tknizr.model_max_tkns - est_next - reserve < 0:
            app_log.info("Current tokens exceeds threshold")
            self.chat_logs.auto_compress_active_conversations()
            return


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
        if cmd == SLASH_CMD_DICT["memorise"].get("cmd"):
            app_log.debug("'%s' command triggered", cmd)
            msgs.append(llm.user_message(user_prompt))
            result = self.slash_memories.cmd_memorise(
                prompt=user_prompt, is_attchmnt=is_attchmnt, paths=paths
            )
            return result if result else None

        if cmd == SLASH_CMD_DICT["recall"].get("cmd"):
            app_log.debug("'%s' command triggered", cmd)
            msgs.append(llm.user_message(user_prompt))
            result = self.slash_recall.cmd_recall(
                prompt=user_prompt, is_attchmnt=is_attchmnt, paths=paths
            )
            return result if result else None

        if cmd == SLASH_CMD_DICT["compress"].get("cmd"):
            app_log.debug("'%s' command triggered", cmd)
            msgs.append(llm.user_message(user_prompt))
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
    ):
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

            # Turn off auto memory retrieve
            is_auto_mem_rtve = False

            # Turn off auto document chunk retrieve
            is_auto_doc_rtve = False

        return prompt_tkns, is_auto_mem_rtve, prompt_embdings, is_auto_doc_rtve


    def _combind_context(
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
        self._manage_token_budget(prompt=prompt, reserve=1000)

        msgs = self.chat_logs.get_active_conversations()

        # Slash commands
        cmd, user_prompt = slash_commands_helpers.detect_cmd(prompt)
        if cmd:
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

        prompt_tkns, is_auto_mem_rtve, prompt_embdings, is_auto_doc_rtve = self._embedding_content_retrieval_controller(
            prompt=prompt, is_auto_mem_rtve=is_auto_mem_rtve, is_auto_doc_rtve=is_auto_doc_rtve
        )

        # Combind all context for sending to the model
        cmbind_prompt, attchmnt_dict = self._combind_context(
            prompt=prompt,
            prompt_embdings=prompt_embdings,
            is_auto_mem_rtve=is_auto_mem_rtve,
            is_auto_doc_rtve=is_auto_doc_rtve,
            is_attchmnt=is_attchmnt,
            paths=paths
        )
        msgs.append(llm.user_message(cmbind_prompt))

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

        chat_database_helpers.update_session_used_tokens_on_prompt_submitted(
            chat_logs=self.chat_logs, tknizr=self.tknizr, cur=conn.cursor()
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
