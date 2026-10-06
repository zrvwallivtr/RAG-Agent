from logging import debug
import json
import mimetypes
from psycopg2 import sql
from pathlib import Path
from typing import Any, Literal

from src.config import models

from .schema import ensure_schema

from src.libs.helpers.database_helpers import fetch_one

from src.agent import chat_logs
from src.models_database import EMB_MODEL_DIMENSION
from src.rag.knowledge_base.documents.main_module import DocumentKnowledgeBase
from src.logger import app_logger


app_log = app_logger(f"{__name__}.app")

EMBED_MODEL = models.EMBED_MODEL

ChatSessionLogs = chat_logs.ChatSessionLogs


class KnowledgeBase:
    def __init__(
        self,
        conn,
        chat_logs: ChatSessionLogs,
    ):
        self.conn   = conn
        self.cur    = self.conn.cursor()

        self.chat_logs  = chat_logs
        self.doc_kw_bs  = DocumentKnowledgeBase(conn=self.conn, chat_logs=self.chat_logs)
        self.sess_name  = self.chat_logs.sess_name

        ensure_schema(self.conn)
        self.doc_names = self.doc_kw_bs.get_all_docs_names

    def clear_sess_kw_bs(self, cont_type: Literal["document"]) -> None:
        """Clear all contents related to specified type and current session."""
        app_log.info("Clearing session '%s' knowledge base", self.sess_name)
        row = fetch_one(
            conn=self.conn,
            sql="DELETE FROM knowledge_base WHERE session_id = %s AND type = %s",
            params=(self.chat_logs.get_session_id(), cont_type)
        )

        if row == 0:
            app_log.warning(
                "Failed to clear session '%s' knowledge base: '%s' contents not found in database",
                self.sess_name, cont_type
            )
