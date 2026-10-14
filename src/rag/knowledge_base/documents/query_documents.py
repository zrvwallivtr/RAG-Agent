import mimetypes
from pathlib import Path
from typing import Any

from src.config.config_toml import (
    RETRIEVE_DOC_ENTRY_LIMIT,
    MIN_SIMILARITY_SCORE
)

from src.libs.helpers.database_helpers import fetch_one, fetch_all, execute

from src.agent.chat_session_logs import ChatSessionLogs
from src.logger import app_logger

app_log = app_logger(f"{__name__}.app")

# Cosine distance
_SEAR_SQL = (
    """
    SELECT content, metadata, 1 - (embeddings <=> %s) AS cosine_similarity
    FROM knowledge_base
    WHERE 1 - (embeddings <=> %s) >= %s
    ORDER BY embeddings <=> %s ASC
    LIMIT %s;
    """
)


class QueryDocuments:
    """SQL query for document knowledge base."""

    def __init__(self, conn, chat_sess_logs: ChatSessionLogs):
        self._conn = conn
        self.chat_sess_logs = chat_sess_logs

    def all_documents_metadata(self) -> list[tuple[Any, Any]] | None:
        """Return a list of all uploaded documents data in the database."""
        rows = fetch_all(
            conn=self._conn,
            sql=(
                """
                SELECT created_at, metadata
                FROM knowledge_base
                WHERE session_id = %s AND type = %s
                    AND (metadata->>'document_chunk_index')::int = 0
                """
            ),
            params=(self.chat_sess_logs.get_session_id(), "document")
        )

        if not rows:
            app_log.debug(
                "Failed to retrieve any documents metadata: "
                "Session '%s' knowledge base is empty",
                self.chat_sess_logs.sess_name
            )
        return rows

    def all_documents_names(self) -> list[str] | None:
        """Return a list of all uploaded documents names in the database."""
        rows = self.all_documents_metadata()
        if not rows:
            return

        # Unpack data in metadata
        doc_names = []
        for _, metadata in rows:
            doc_names.append(metadata["document_name"])
        return doc_names

    # --- Vector search -------------------------------------------------------

    def query_uploads(
        self,
        qry: str,
        qry_embdings: list[float],
        sear_sql: str = _SEAR_SQL,
        min_sim_score: float = MIN_SIMILARITY_SCORE,
    ) -> list[dict[str, Any]] | None:
        """Queries knowledge base for similar content."""
        kw_dict = []

        rows = fetch_all(
            conn=self._conn,
            sql=sear_sql,
            params=(
                str(qry_embdings),
                str(qry_embdings),
                min_sim_score,
                str(qry_embdings),
                RETRIEVE_DOC_ENTRY_LIMIT,
            )
        )

        if rows:
            for cont, metadata, score in rows:
                kw_dict.append({
                    "document_name": metadata.get("document_name", "Unknown"),
                    "content": cont,
                    "similarity": score
                })
            return kw_dict
        else:
            app_log.debug(
                "Failed to retrieve relevant document chunks: "
                "Session knowledge base is empty or no content reached"
                " minimum similarity score"
            )
            return
