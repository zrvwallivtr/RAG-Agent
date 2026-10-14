import json
from pathlib import Path
from typing import Any
from datetime import datetime, timedelta, timezone

from .query_documents import QueryDocuments
from .store_documents import StoreDocuments

from src.libs.helpers.database_helpers import fetch_one, fetch_all, execute
from src.libs.helpers.string_helpers import hash_string
from src.libs.helpers.file_metadata_helpers import unpack_file_metadata
from src.libs.custom_dataclasses import AgentHooks

from src.agent.models import embed
from src.agent.chat_session_logs import ChatSessionLogs
from src.rag.file_parser import FileParser
from src.logger import app_logger

app_log = app_logger(f"{__name__}.app")


class DocumentKnowledgeBase:
    def __init__(
        self,
        conn,
        chat_sess_logs: ChatSessionLogs,
    ):
        self.conn = conn

        self.chat_sess_logs = chat_sess_logs
        self.file_parser = FileParser()
        self.query_docs = QueryDocuments(
            conn=self.conn, chat_sess_logs=self.chat_sess_logs
        )
        self.store_docs = StoreDocuments(
            conn=self.conn,
            chat_sess_logs=self.chat_sess_logs,
            file_parser=self.file_parser,
            query_docs=self.query_docs
        )
        self.sess_name = self.chat_sess_logs.sess_name

    # --- Query documents -----------------------------------------------------

    def get_document_metadata_from_path(self, path):
        return unpack_file_metadata(path)

    def get_attachments_content(
        self, is_attchmnt, attch_paths: list[Path] | None
    ):
        return self.file_parser.read_content(is_attchmnt, attch_paths)

    def _get_all_docs_metadata(self):
        return self.query_docs.all_documents_metadata()

    def get_all_docs_names(self):
        return self.query_docs.all_documents_names()

    def query_similar_knowledge(self, qry: str, qry_embdings: list[float]):
        return self.query_docs.query_uploads(qry, qry_embdings)

    def toggle_auto_retrieve_sess_docs(
        self,
        is_auto_doc_rtve: bool,
        prompt: str,
        prompt_embdings: list[float]
    ) -> list[dict[str, Any]] | None:
        """
        Auto fetches previous documents contents ability, return relevant
        contents if its toggled on.
        """
        if not is_auto_doc_rtve:
            return

        return self.query_similar_knowledge(prompt, prompt_embdings)

    # --- Store documents -----------------------------------------------------

    def store_attachments(
        self,
        attchmnt_dict,
        hooks: AgentHooks | None = None,
    ):
        return self.store_docs.store_documents(attchmnt_dict, hooks=hooks)

    def embed_and_add_to_kw_bs(
        self,
        path,
        cont,
        format,
        hooks: AgentHooks | None = None,
    ):
        return self.store_docs.add_embed_to_knowledge_base(
            path, cont, format, hooks=hooks
        )
