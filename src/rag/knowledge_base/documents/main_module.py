import json
from pathlib import Path
from typing import Any
from datetime import datetime, timedelta, timezone

from .document_reader import DocumentReader
from .query_documents import QueryDocuments
from .store_documents import StoreDocuments

from src.libs.helpers.database_helpers import fetch_one, fetch_all, execute
from src.libs.helpers.string_helpers import hash_string

from src.agent import chat_logs, embed
from src.agent.chat_logs import ChatSessionLogs
from src.logger import app_logger

app_log = app_logger(f"{__name__}.app")


class DocumentKnowledgeBase:
    def __init__(
        self,
        conn,
        chat_logs: ChatSessionLogs,
    ):
        self.conn = conn

        self.chat_logs = chat_logs
        self.doc_reader = DocumentReader()
        self.query_docs = QueryDocuments(conn=self.conn, chat_logs=self.chat_logs)
        self.store_docs = StoreDocuments(
            conn=self.conn, chat_logs=self.chat_logs,
            doc_reader=self.doc_reader, query_docs=self.query_docs
        )
        self.sess_name = self.chat_logs.sess_name

    # File metadata

    def get_document_metadata_from_path(self, path):
        return self.query_docs.get_document_metadata_from_path(path)

    # Attachments

    def store_attachments(self, attchmnt_dict):
        return self.store_docs.store_attachments(attchmnt_dict)

    def get_attachments_content(self, is_attchmnt, attch_paths: list[Path] | None):
        return self.doc_reader.read_content(is_attchmnt, attch_paths)

    # EMBEDDING

    def embed_and_add_to_kw_bs(self, path, cont, format):
        return self.store_docs.add_embed_to_knowledge_base(path, cont, format)

    # FROM DOCUMENTS IN KNOWLEDGE BASE

    def _get_all_docs_metadata(self):
        return self.query_docs.all_documents_metadata()

    def get_all_docs_names(self):
        return self.query_docs.all_documents_names()

    # LIST CONTENTS

    def list_all_uploaded_documents(self) -> str | None:
        """Return a list of all document(s) in the database."""
        rows = self.query_docs.all_documents_metadata()
        if not rows:
            return

        # Unpack data in metadata
        lines = [
            f"UPLOADED DOCUMENT(S)\n"
            f"====================\n"
            f"UPLOADED AT\t\t\t\tNAME"
        ]
        for time, metadata in rows:
            data_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
            doc_name = data_dict.get("document_name", "Unknown")
            lines.append(f"{str(time)}\t{doc_name}")
        return "\n".join(lines)

    # QUERY KNOWNLEDGE BASE

    def query_similar_knowledge(
        self, qry: str, qry_embdings: list[float], min_sim: float = 0.65
    ):
        return

    # AUTO FUNCTIONS

    def toggle_auto_retrieve_sess_docs(
        self,
        is_auto_doc_rtve: bool,
        prompt: str,
        prompt_embdings: list[float]
    ) -> list[dict[str, Any]] | None:
        """Auto fetches previous documents contents ability, return relevant contents if its toggled on."""
        if not is_auto_doc_rtve:
            return

        app_log.debug("Auto document retrieve is currently on")
        return self.query_similar_knowledge(prompt, prompt_embdings)
