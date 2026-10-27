from pathlib import Path
from typing import Any

from src.config.postgres import CONN
from src.config.config_toml import MODEL, EMBED_MODEL, TOKEN_RESERVE

from src.libs.helpers.format_context_helper import build_prompt
from src.libs.helpers.slash_commands_helpers import detect_cmd
from src.libs.custom_dataclasses import AgentHooks
from src.libs.constants import CompressionResult

from src.agent.chat_session_logs import ChatSessionLogs
from src.agent.tokenizers import LocalTokenizer
from src.rag.expert import Expert, SessionExperts
from src.slash_commands import SlashCompress
from src.slash_commands.compress import COMPRESS_CMD
from src.agent.models.llm import user_message, model_response
from src.agent.models.embed import embedding_content
from src.rag.file_reader import FileReader
from src.rag.attach_files import Attachments, QueryAttachments
from src.rag.docling import DoclingChunker
from src.rag.basic_parser import BasicParser
from src.rag.docling import DoclingParser

from src import logger

app_log = logger.app_logger(f"{__name__}.app")


class Agent:
    """Contains functions for how the agent response."""

    def __init__(
        self,
        sess_name: str | None = None,
        model: str = MODEL,
        emb_model: str = EMBED_MODEL,
    ):
        self.conn = CONN
        self.llm_tknizr = LocalTokenizer(model)
        self.emb_tknizr = LocalTokenizer(emb_model)
        self.file_reader = FileReader(
            basic_parser=BasicParser(), docling_parser=DoclingParser()
        )
        self.chunker = DoclingChunker()

        self.sess_name = sess_name

        self.chat_sess_logs = ChatSessionLogs(
            conn=self.conn, sess_name=self.sess_name, llm_tknizr=self.llm_tknizr
        )
        self.slash_compress = SlashCompress(
            conn=self.conn,
            chat_sess_logs=self.chat_sess_logs,
            file_reader=self.file_reader,
            chunker=self.chunker,
        )
        self.attchmnts = Attachments(
            conn=self.conn,
            chat_sess_logs=self.chat_sess_logs,
            file_reader=self.file_reader,
            chunker=self.chunker,
        )
        self.qry_attchmnts = QueryAttachments(
            conn=self.conn, chat_sess_logs=self.chat_sess_logs
        )

    def ask(
        self,
        prompt: str,
        paths: list[Path] | None = None,
        enabled_exp: list[str] | None = None,
        is_auto_doc_rtve: bool = True,
        hooks: AgentHooks | None = None,
    ) -> tuple[str, int, int, dict[str, dict[str, Any]]] | None:
        """
        Note:
        - Only the user question and LLM response will be
          stored into chat history.
        """
        msgs = self.chat_sess_logs.get_active_conversations()

        # Slash commands
        cmd, clean_prompt = detect_cmd(prompt)
        if cmd:
            # Run budget check and auto compress only when slash command is
            # not '/compress' as it is already a session compression tool call.
            if cmd != COMPRESS_CMD:
                msgs = self._compress_if_over_budget(clean_prompt, hooks=hooks)

            return self._route_to_slash_command(
                unfmt_prompt=prompt,
                clean_prompt=clean_prompt,
                msgs=msgs,
                cmd=cmd,
                paths=paths,
                hooks=hooks,
            )

        prompt, prompt_tkns, prompt_embdings, is_auto_doc_rtve = (
            self._embedding_content_retrieval_controller(
                prompt=prompt, is_auto_doc_rtve=is_auto_doc_rtve
            )
        )

        # Combine all context, check token count and bundle them
        # into messages for model to answer.
        cmbine_prompt, exp_chunks, retrieved_attchmnts, attchmnt_dict = (
            self._combine_context(
                prompt=prompt,
                prompt_embdings=prompt_embdings,
                is_auto_doc_rtve=is_auto_doc_rtve,
                paths=paths,
                hooks=hooks,
            )
        )
        cmbine_prompt, attchmnt_dict, is_stored = self._prompt_tokens_budget_handler(
            prompt=prompt,
            prompt_embdings=prompt_embdings,
            msgs=msgs,
            cmbind_prompt=cmbine_prompt,
            exp_chunks=exp_chunks,
            retrieved_attchmnts=retrieved_attchmnts,
            attchmnt_dict=attchmnt_dict,
            hooks=hooks,
        )
        msgs = self._compress_if_over_budget(cmbine_prompt, hooks=hooks)
        msgs.append(user_message(cmbine_prompt))

        # Model answer
        response = model_response(model=MODEL, msgs=msgs, hooks=hooks)
        if not response:
            return
        ans, p_tkns, o_tkns, is_stopped = response

        if is_stopped is True:
            return None

        # Calculate total tokens
        total_p_tkns = prompt_tkns + p_tkns
        total_o_tkns = o_tkns

        # Save messages to database
        metadata = self.chat_sess_logs.add_conversation_turn(
            prompt=prompt,
            response=ans,
            state="external",
            attchmnts=paths,
            p_tkns=total_p_tkns,
            o_tkns=total_o_tkns,
        )

        self.chat_sess_logs.update_session_used_tokens()

        # Store attachment(s)
        if attchmnt_dict and not is_stored:
            self.attchmnts.store_already_read_attachments(attchmnt_dict, hooks=hooks)
        return ans, total_p_tkns, total_o_tkns, metadata

    @property
    def exp_list(self) -> list[str]:
        """Names of the expert enabled in this session (read on each use)."""
        return list(
            SessionExperts.all_expert_data_from_session(self.conn, self.sess_name)
        )

    def _is_over_token_budget(
        self,
        msgs: list[dict],
        prompt: str,
        reserve: int,
        hooks: AgentHooks | None = None,
    ) -> bool:
        """
        True if the history plus the new prompt plus the reply reserve would
        exceed the model's token limit.
        """
        if not self.llm_tknizr.model_max_tkns:
            app_log.warning(
                "Failed to load 'manage token budget feature': "
                "Model maximum token limit not set"
            )
            return False

        hist_tkns = self.llm_tknizr.count_history_tokens(msgs)
        if not hist_tkns:
            return False

        # Actual result
        est_next = len(prompt) // 4
        return hist_tkns + est_next + reserve > self.llm_tknizr.model_max_tkns

    def _compress_if_over_budget(
        self,
        prompt: str,
        hooks: AgentHooks | None = None,
    ) -> list[dict]:
        """
        Compress the session if history + prompt + reserve exceeds the limit.
        Returns the history.
        """
        msgs = self.chat_sess_logs.get_active_conversations()
        if self._is_over_token_budget(msgs=msgs, prompt=prompt, reserve=TOKEN_RESERVE):
            self.chat_sess_logs.auto_compress_active_conversations(
                unfmt_prompt="Auto compress", hooks=hooks
            )
            msgs = self.chat_sess_logs.get_active_conversations()
        return msgs

    def _route_to_slash_command(
        self,
        unfmt_prompt: str,
        clean_prompt: str,
        msgs: list[dict],
        cmd: str,
        paths: list[Path] | None,
        hooks: AgentHooks | None = None,
    ) -> CompressionResult | None:
        """Route to slash command for '/compress'."""
        if cmd == COMPRESS_CMD:
            app_log.debug("'%s' command triggered", cmd)
            result = self.slash_compress.run_compress(
                clean_prompt=clean_prompt,
                unfmt_prompt=unfmt_prompt,
                paths=paths,
                hooks=hooks,
            )
            return result if result else None

    def _embedding_content_retrieval_controller(
        self, prompt: str, is_auto_doc_rtve: bool
    ) -> tuple[str, int, list[float], bool]:
        """
        Return embeddings for user prompt for document and
        previously attaached content retrieval features,
        if no embedding failed turn off related featurs.
        """
        embed_response = embedding_content(prompt)

        if embed_response:
            prompt, prompt_embdings, prompt_tkns = embed_response

        else:
            app_log.warning(
                "Turning off features that requires embeddings: "
                "Auto memory retrieval, Auto document content retrieval"
            )
            prompt_embdings = [0.0]
            prompt_tkns = 0
            is_auto_doc_rtve = False  # Turn off auto document chunk retrieve

        return prompt, prompt_tkns, prompt_embdings, is_auto_doc_rtve

    def _query_experts(
        self, prompt: str, hooks: AgentHooks | None = None
    ) -> list[dict] | None:
        """
        Query every expert enabled in this session. Returns the closest
        chunks across them, best first, or None if none are enabled.
        """
        if not self.exp_list:
            return None
        return Expert.query_multiple_experts(
            conn=CONN, prompt=prompt, exp_list=self.exp_list
        )

    def _combine_context(
        self,
        prompt: str,
        prompt_embdings: list[float],
        is_auto_doc_rtve: bool,
        paths: list[Path] | None = None,
        hooks: AgentHooks | None = None,
    ) -> tuple[str, list[dict] | None, list[dict] | None, dict | None]:
        # Expert document chunks (only if experts are enabled in this session)
        exp_chunks = self._query_experts(prompt, hooks=hooks)

        # Auto retrieve relevant session documents
        retrieved_attchmnts = None
        if is_auto_doc_rtve:
            retrieved_attchmnts = self.qry_attchmnts.query_session_attachments(
                qry_embeddings=prompt_embdings,
            )

        # Uploaded attachments (optional)
        attchmnt_dict = None
        if paths:
            attchmnt_dict = self.file_reader.read_content(paths)

        # All context combined (won't be saved to chat history)
        cmbind_prompt = build_prompt(
            prompt=prompt,
            exp_chunks=exp_chunks,
            doc_list=retrieved_attchmnts,
            attchmnt_dict=attchmnt_dict,
        )

        return cmbind_prompt, exp_chunks, retrieved_attchmnts, attchmnt_dict

    def _prompt_tokens_budget_handler(
        self,
        prompt: str,
        prompt_embdings: list[float],
        msgs: list[dict],
        cmbind_prompt: str,
        exp_chunks: list[dict] | None,
        retrieved_attchmnts: list[dict] | None,
        attchmnt_dict: dict | None,
        tkn_reserve: int = TOKEN_RESERVE,
        hooks: AgentHooks | None = None,
    ) -> tuple[str, dict, bool] | tuple[str, None, bool]:
        """
        If the attachments don't fit in the context window, store them into the
        database first, and retrieve the relevant chunks from the database.

        Return (prompt, attachments, is_stored). is_stored is True only if the
        attachments were stored here, so they aren't stored again downstream.
        """
        if not attchmnt_dict:
            return cmbind_prompt, None, False

        if not self.llm_tknizr.model_max_tkns:
            return cmbind_prompt, attchmnt_dict, False

        hist_tkns = self.llm_tknizr.count_history_tokens(msgs) or 0
        added_tkns = self.llm_tknizr.count_string_tokens(cmbind_prompt) or 0

        if hist_tkns + added_tkns + tkn_reserve <= self.llm_tknizr.model_max_tkns:
            return cmbind_prompt, attchmnt_dict, False

        stored = self.attchmnts.store_already_read_attachments(attchmnt_dict, hooks=hooks)
        if not stored:
            app_log.warning("Failed to store attachments: Sending in full")
            return cmbind_prompt, attchmnt_dict, False

        base_prompt = build_prompt(
            prompt=prompt, exp_chunks=exp_chunks, doc_list=retrieved_attchmnts
        )
        tkn_budget = (
            self.llm_tknizr.model_max_tkns
            - tkn_reserve
            - hist_tkns
            - (self.llm_tknizr.count_string_tokens(base_prompt) or 0)
        )

        hits = self.qry_attchmnts.query_specified_session_attachments(
            qry_embeddings=prompt_embdings,
            attchmnt_dict=attchmnt_dict,
            tkn_budget=tkn_budget,
            hooks=hooks,
        )
        if not hits:
            app_log.warning("No attachment chunks fit the token budget")

        cmbind_prompt = build_prompt(
            prompt=prompt,
            exp_chunks=exp_chunks,
            doc_list=[*(retrieved_attchmnts or []), *(hits or [])],
        )
        return cmbind_prompt, attchmnt_dict, True
