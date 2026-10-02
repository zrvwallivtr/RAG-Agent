from src.logger import app_logger


app_log = app_logger(f"{__name__}.app")


def detect_cmd(prompt: str) -> tuple[str | None, str]:
    """Extracts shortcut if detected."""
    question_trimmed = prompt.strip()

    if question_trimmed.startswith(f"/"):
        parts           = question_trimmed.split(" ", 1)
        cmd             = parts[0]
        cleaned_text    = parts[1].strip() if len(parts) > 1 else ""
        app_log.debug("Command detected: %s", cmd)
        return cmd, cleaned_text

    return None, prompt


# class SlashCmds:
#     def __init__(
#         self,
#         conn,
#         chat_logs: ChatLogs,
#         sess_name: str | None = None,
#         project: str | None = None
#     ):
#         self.conn       = conn
#         self.sess_name  = sess_name.strip() if sess_name else "default_session"
#         self.project    = project
#         self.chat_logs  = chat_logs
# 
#         self.mem = Memory(
#             conn=self.conn, chat_logs=self.chat_logs, project=self.project
#         )
#         self.kw_bs = KnowledgeBase(
#             conn=self.conn, chat_logs=self.chat_logs, sess_name=self.sess_name
#         )
#         self.doc_kw_bs = DocumentKnowledgeBase(
#             conn=self.conn, chat_logs=self.chat_logs, sess_name=self.sess_name
#         )
#         self.sear_agt = SearchAgent(
#             conn=self.conn, sess_name=self.sess_name
#         )
# 
# 
#     def cmd_search(
#         self,
#         prompt: str,
#         is_attchmnt: bool,
#         paths: list[Path] | None
#     ) -> str | None:
#         """
#         Generates, search and answer query based on user prompt.
# 
#         WEB SEARCH PROMPT INCLUDES:
#         - Attachments (optional):
#           -> Allows option to upload attachments for more specific searches.
#         - User prompt:
#           -> Main query that influences model's searches.
# 
#         INTERPRET SEARCH RESULTS PROMPT INCLUDES:
#         - Attachments (optional):
#           -> Allows extra context from attachments.
#         - Web search results:
#           -> From web search results interpret/answer user prompt.
#         - User prompt:
#           -> Uses the same prompt as the previous step, this time for model
#              to answer from the retrieved search results.
#         """
#         if not prompt:
#             app_log.error("Command '/search' aborted: No prompt was provided")
#             return "Please specify what to search."
# 
#         msgs = self.chat_logs.get_actv_convs()
# 
#         # Full context for web search
#         attchmnt_dict = self.doc_kw_bs.get_attachments_content(
#             is_attchmnt=is_attchmnt, attch_paths=paths
#         )
# 
#         cmbind_sear_prompt = format_context.build_prompt(
#             prompt=prompt, attchmnt_dict=attchmnt_dict
#         )
# 
#         # Full context for model answer
#         response = self.sear_agt.query_surface_content(
#             contxt=msgs, prompt=cmbind_sear_prompt
#         )
#         if not response:
#             return "No results found"
#         sear_results, gen_qry_p_tkns, gen_qry_o_tkns = response
# 
#         cmbind_prompt = format_context.build_prompt(
#             prompt=prompt, attchmnt_dict=attchmnt_dict, sear_results=sear_results
#         )
#         msgs.append(llm.user_message(cmbind_prompt))
# 
#         # Model answer
#         ans_response = llm.model_response(
#             model=MODEL,
#             msgs=msgs
#         )
#         if not ans_response:
#             return
#         answer, ans_p_tkns, ans_o_tkns = ans_response
# 
#         # Save messages
#         self.chat_logs.add_conv_turn(
#             prompt=prompt,
#             response=answer,
#             state="external",
#             p_tkns=gen_qry_p_tkns + ans_p_tkns,
#             o_tkns=gen_qry_o_tkns + ans_o_tkns
#         )
# 
#         # Store attachments
#         if attchmnt_dict:
#             self.doc_kw_bs.store_attachments(attchmnt_dict)
#         return
