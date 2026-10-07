from typing import Literal, Any, get_args

from src.config.memory import AUTO_MEMORY_STORE_TOKENS

from .schema import ensure_schema
from .store_entries import StoreEntries
from .query_entries import QueryEntries

from src.libs.helpers.database_helpers import fetch_one, fetch_all, execute

from src.agent.chat_session_logs import ChatSessionLogs
from src.models_database import EMB_MODEL_DIMENSION
from src.logger import app_logger
from src import format_context

app_log     = app_logger(f"{__name__}.app")


class Memory:
    def __init__(
        self,
        conn,
        chat_sess_logs: ChatSessionLogs,
    ):
        self.conn = conn

        self.chat_sess_logs = chat_sess_logs
        self.query_entries = QueryEntries(conn=self.conn, chat_sess_logs=self.chat_sess_logs)
        self.store_entries = StoreEntries(conn=self.conn, chat_sess_logs=self.chat_sess_logs)

        ensure_schema(self.conn)

    def get_mem_content_from_ids(self, ids):
        return self.query_entries.query_from_ids(ids)

    def query_similar_content(self, qry, qry_embdings):
        return self.query_entries.cosine_similarity_search(qry, qry_embdings)

    def extract_and_store_mem_from_conv(self, extraction, prompt: str | None = None):
        return self.store_entries.store_extracted_memory(extraction, prompt)

    def toggle_auto_retrive_memory_entries(
        self,
        is_auto_mem_rtve: bool,
        prompt: str,
        prompt_embdings: list[float]
    ) -> list[dict[str, Any]] | None:
        if not is_auto_mem_rtve:
            return
        return self.query_similar_content(prompt, prompt_embdings)

    # This should ONLY read the current conversation turn as context
    # (Auto store memory runs automatically for every conversation turns.)
    def toggle_auto_store_memory_entries(
        self,
        enable_auto_memory_store: bool,
        model_max_tokens: int,
        contxt: list[dict],
    ):
        """
        Auto store memory no prompts needed. Only calls at end of every conversations.
        """
        if enable_auto_memory_store == False:
            return

        if model_max_tokens < AUTO_MEMORY_STORE_TOKENS:
            return

        response = self.extract_and_store_mem_from_conv(extraction= "auto")
        if response:
            created_ids, p_tkns, o_tkns, total_tkn_used = response

    def delete_memory(self, id):
        return self.store_entries.delete_entry_by_id(id)

    def delete_mem(self, ids):
        return self.store_entries.delete_entries_by_ids(ids)
