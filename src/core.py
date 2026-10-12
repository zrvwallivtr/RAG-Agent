from pathlib import Path
from typing import Callable, Any

from src.config.postgres import CONN
from src.config.models import MODEL, TOKEN_RESERVE

from src.libs.helpers.format_context_helper import build_prompt
from src.libs.helpers.slash_commands_helpers import detect_cmd
from src.libs.custom_dataclasses import AgentHooks

from src.agent.chat_session_logs import ChatSessionLogs
from src.agent.tokenizers import LocalTokenizer
from src.slash_commands import SlashCompress
from src.slash_commands.compress import COMPRESS_CMD
from src.agent.models.llm import user_message, model_response
from src.agent.models.embed import embedding_content

from src.rag import KnowledgeBase, DocumentKnowledgeBase

from src import logger

app_log = logger.app_logger(f"{__name__}.app")


class Agent:
    def __init__(
        self,
        sess_name: str | None = None,
    ):
        self.sess_name = sess_name
        self.conn      = CONN

        self.chat_sess_logs = ChatSessionLogs(
            conn=self.conn, sess_name=self.sess_name
        )
        self.kw_bs = KnowledgeBase(
            conn=self.conn, chat_sess_logs=self.chat_sess_logs
        )
        self.doc_kw_bs = DocumentKnowledgeBase(
            conn=self.conn, chat_sess_logs=self.chat_sess_logs
        )
        self.local_tknizr = LocalTokenizer(MODEL)

        self.slash_compress = SlashCompress(
            conn=self.conn, chat_logs=self.chat_sess_logs
        )

    def _is_over_budget(
        self, msgs: list[dict], prompt: str, reserve: int
    ) -> bool:
        """
        True if the history plus the new prompt plus the reply reserve would
        exceed the model's token limit.
        """
        max_tkns = self.local_tknizr.model_max_tkns
        if not max_tkns:
            app_log.warning(
                "Failed to load 'manage token budget feature': "
                "Model maximum token limit not set"
            )
            return False

        hist_tkns = self.local_tknizr.count_history_tokens(msgs)
        if not hist_tkns:
            return False

        # Actual result
        est_next = len(prompt) // 4
        return hist_tkns + est_next + reserve > max_tkns

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
        if self._is_over_budget(msgs=msgs, prompt=prompt, reserve=TOKEN_RESERVE):
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
        is_attchmnt: bool,
        paths: list[Path] | None,
        hooks: AgentHooks | None = None,
    ) -> tuple[str, int, int, dict[str, dict[str, Any]]] | None:
        """Route to slash command for '/memorise', '/recall' and '/compress'."""
        if cmd == COMPRESS_CMD:
            app_log.debug("'%s' command triggered", cmd)
            result = self.slash_compress.run_compress(
                clean_prompt=clean_prompt,
                unfmt_prompt=unfmt_prompt,
                is_attchmnt=is_attchmnt,
                paths=paths,
                hooks=hooks,
            )
            return result if result else None

    def _embedding_content_retrieval_controller(
        self, prompt: str, is_auto_doc_rtve: bool
    ) -> tuple[str, int, list[float], bool]:
        """
        Return embeddings for user prompt for memory and document content retrieval
        features, if no embedding failed turn off related featurs.
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
            is_auto_doc_rtve = False # Turn off auto document chunk retrieve

        return prompt, prompt_tkns, prompt_embdings, is_auto_doc_rtve

    def _combine_context(
        self,
        prompt: str,
        prompt_embdings: list[float],
        is_auto_doc_rtve: bool,
        is_attchmnt: bool,
        paths: list[Path] | None = None
    ) -> tuple[str, dict] | tuple[str, None]:
        # Auto retrieve relevant session documents
        doc_list = self.doc_kw_bs.toggle_auto_retrieve_sess_docs(
            is_auto_doc_rtve=is_auto_doc_rtve, prompt=prompt, prompt_embdings=prompt_embdings
        )

        # Uploaded attachments (optional)
        attchmnt_dict = self.doc_kw_bs.get_attachments_content(
            is_attchmnt=is_attchmnt, attch_paths=paths
        )

        # All context combined (won't be saved to chat history)
        cmbind_prompt = build_prompt(
            prompt=prompt, doc_list=doc_list, attchmnt_dict=attchmnt_dict
        )

        if attchmnt_dict:
            return cmbind_prompt, attchmnt_dict
        return cmbind_prompt, None

    def ask(
        self,
        prompt: str,
        paths: list[Path] | None = None,
        is_auto_doc_rtve: bool = True,
        is_attchmnt: bool = False,
        hooks: AgentHooks | None = None,
    ) -> tuple[str, int, int, dict[str, dict[str, Any]]] | None:
        """
        Model decide what memories to read.
        Manage tokens, compress session if needed.

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
                msgs = self._compress_if_over_budget(
                    clean_prompt, hooks=hooks
                )

            return self._route_to_slash_command(
                unfmt_prompt=prompt,
                clean_prompt=clean_prompt,
                msgs=msgs,
                cmd=cmd,
                is_attchmnt=is_attchmnt,
                paths=paths,
                hooks=hooks,
            )

        prompt_tkns, prompt_tkns, prompt_embdings, is_auto_doc_rtve = (
            self._embedding_content_retrieval_controller(
                prompt=prompt,
                is_auto_doc_rtve=is_auto_doc_rtve
            )
        )

        # Combine all context, check token count and bundle them
        # into messages for model to answer.
        combine_prompt, attchmnt_dict = self._combine_context(
            prompt=prompt,
            prompt_embdings=prompt_embdings,
            is_auto_doc_rtve=is_auto_doc_rtve,
            is_attchmnt=is_attchmnt,
            paths=paths,
        )
        msgs = self._compress_if_over_budget(combine_prompt, hooks=hooks)
        msgs.append(user_message(combine_prompt))

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
            o_tkns=total_o_tkns
        )

        self.chat_sess_logs.update_session_used_tokens()

        # Store attachment(s)
        if attchmnt_dict:
            self.doc_kw_bs.store_attachments(attchmnt_dict)
        return ans, total_p_tkns, total_o_tkns, metadata
