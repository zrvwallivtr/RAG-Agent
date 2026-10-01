from re import search
from pathlib import Path
from typing import Any

from src.config import models
from src.config import prompts

from src import format_context
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
from src.logger import app_logger


app_log = app_logger(f"{__name__}.app")

MODEL                       = models.MODEL
MEM_RECALL_INTERPRET_PROMPT = prompts.MEM_RECALL_INTERPRET_PROMPT

ChatLogs                = chat_logs.ChatLogs
Memory                  = memory.Memory
KnowledgeBase           = knowledge_base.KnowledgeBase
DocumentKnowledgeBase   = document_knowledge_base.DocumentKnowledgeBase
SearchAgent             = search_agent.SearchAgent


class SlashCmds:
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
                "Command '/memorise' aborted: No prompt was provided. Please specify instructions"
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
            app_log.debug("Removing saved memory(s)")
            self.mem.delete_mem(created_ids)
            return

        # Save messages
        mock_resp = "Important information(s) has been extracted added to database."
        metadata = self.chat_logs.add_conv_turn(
            prompt=prompt,
            response=mock_resp,
            state="external",
            attchmnts=paths,
            p_tkns=p_tkns,
            o_tkns=o_tkns
        )

        # Store attachments
        if attchmnt_dict:
            self.doc_kw_bs.store_attachments(attchmnt_dict)

        return "", p_tkns, o_tkns, metadata


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
            app_log.warning("Command '/recall' aborted: No prompt was provided")
            return

        if is_attchmnt and paths:
            app_log.warning(
                "Command '/recall' does not support attachment uploads. "
                "Ignoring uploaded content(s)"
            )

        msgs = self.chat_logs.get_actv_convs()

        # Retrieve memory from database
        embed_response = embed.embedding_content(prompt)
        if not embed_response:
            return
        prompt, prompt_embdings, emb_tkns = embed_response

        mem_list = self.mem.query_similar_content(qry=prompt, qry_embdings=prompt_embdings)

        # Interpret recalled memories
        ans, p_tkns, o_tkns = llm.response_memory_recall_format(
            model=self.mem.model,
            sys_prompt=MEM_RECALL_INTERPRET_PROMPT,
            prompt=prompt,
            context=msgs
        )

        # Save messages
        metadata = self.chat_logs.add_conv_turn(
            prompt=prompt,
            response=ans,
            state="external",
            p_tkns=p_tkns,
            o_tkns=o_tkns
            # /// Add embed tokens log ///
        )
        return ans, p_tkns, o_tkns, metadata


    def cmd_compress(
        self, prompt: str, is_attchmnt: bool, paths: list[Path] | None
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
        # Full context
        attchmnt_dict = self.doc_kw_bs.get_attachments_content(
            is_attchmnt=is_attchmnt, attch_paths=paths
        )
        cmbind_prompt = format_context.build_prompt(
            prompt=prompt, attchmnt_dict=attchmnt_dict
        )

        # Default compression (no instructions)
        if not prompt:
            if is_attchmnt and paths:
                app_log.warning(
                    "Command '/compress' aborted: Prompt must be provided if attachment is uploaded"
                )
                return

            result = self.chat_logs.auto_compresss_active_conv()
            if not result:
                return
            smry, p_tkns, o_tkns, metadata = result

        # Instructed compression
        else:
            app_log.info("Compression session '%s' conversations...", self.sess_name)
            result = self.chat_logs.compress_active_conv(prompt=cmbind_prompt)
            if not result:
                return
            smry, p_tkns, o_tkns, metadata = result

            # Store attachments
            if attchmnt_dict:
                self.doc_kw_bs.store_attachments(attchmnt_dict)

        return smry, p_tkns, o_tkns, metadata


    def cmd_search(
        self,
        prompt: str,
        is_attchmnt: bool,
        paths: list[Path] | None
    ) -> str | None:
        """
        Generates, search and answer query based on user prompt.

        WEB SEARCH PROMPT INCLUDES:
        - Attachments (optional):
          -> Allows option to upload attachments for more specific searches.
        - User prompt:
          -> Main query that influences model's searches.

        INTERPRET SEARCH RESULTS PROMPT INCLUDES:
        - Attachments (optional):
          -> Allows extra context from attachments.
        - Web search results:
          -> From web search results interpret/answer user prompt.
        - User prompt:
          -> Uses the same prompt as the previous step, this time for model
             to answer from the retrieved search results.
        """
        if not prompt:
            app_log.error("Command '/search' aborted: No prompt was provided")
            return "Please specify what to search."

        msgs = self.chat_logs.get_actv_convs()

        # Full context for web search
        attchmnt_dict = self.doc_kw_bs.get_attachments_content(
            is_attchmnt=is_attchmnt, attch_paths=paths
        )

        cmbind_sear_prompt = format_context.build_prompt(
            prompt=prompt, attchmnt_dict=attchmnt_dict
        )

        # Full context for model answer
        response = self.sear_agt.query_surface_content(
            contxt=msgs, prompt=cmbind_sear_prompt
        )
        if not response:
            return "No results found"
        sear_results, gen_qry_p_tkns, gen_qry_o_tkns = response

        cmbind_prompt = format_context.build_prompt(
            prompt=prompt, attchmnt_dict=attchmnt_dict, sear_results=sear_results
        )
        msgs.append(llm.user_message(cmbind_prompt))

        # Model answer
        ans_response = llm.model_response(
            model=MODEL,
            msgs=msgs
        )
        if not ans_response:
            return
        answer, ans_p_tkns, ans_o_tkns = ans_response

        # Save messages
        self.chat_logs.add_conv_turn(
            prompt=prompt,
            response=answer,
            state="external",
            p_tkns=gen_qry_p_tkns + ans_p_tkns,
            o_tkns=gen_qry_o_tkns + ans_o_tkns
        )

        # Store attachments
        if attchmnt_dict:
            self.doc_kw_bs.store_attachments(attchmnt_dict)
        return
