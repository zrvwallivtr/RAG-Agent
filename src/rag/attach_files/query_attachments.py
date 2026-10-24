from typing import Any

from src.config.config_toml import (
    RETRIEVE_DOC_ENTRY_LIMIT,
    MIN_SIMILARITY_SCORE,
    EXPLORATION_FACTOR,
    EMBED_MODEL,
)

from ._schema import ensure_schema

from src.libs.helpers.database_helpers import fetch_one, fetch_all, execute

from src.agent.chat_session_logs import ChatSessionLogs
from src.logger import app_logger

app_log = app_logger(f"{__name__}.app")

# Cosine distance, nearest first. Only this session's attachments, and only
# ones embedded with the CURRENT embedding model (vectors from different
# models can't be compared).
# Placeholders: query vector, session id, embedding model, query vector, limit.
_SEARCH_SQL = """
    SELECT a.title, c.content, c.metadata, c.prompt_tokens,
        c.embeddings <=> %s::vector AS distance
    FROM attachment_chunks c
    JOIN attachments a ON a.id = c.attachment_id
    WHERE a.session_id = %s AND a.embedding_model = %s
    ORDER BY c.embeddings <=> %s::vector
    LIMIT %s;
    """


class QueryAttachments:
    """Search for the attachments of the current session."""

    def __init__(
        self, conn, chat_sess_logs: ChatSessionLogs, ef_search: int = EXPLORATION_FACTOR
    ):
        self.conn = conn
        self.chat_sess_logs = chat_sess_logs
        self.sess_id = self.chat_sess_logs.get_session_id()
        if not self.sess_id:
            raise ValueError("Attachments need an existing chat session.")
        self._ef_search = ef_search
        ensure_schema(conn)

    def all_attachments_titles(self) -> list[str]:
        """Titles of all attachments in the current session."""
        rows = fetch_all(
            conn=self.conn,
            sql="SELECT title FROM attachments WHERE session_id = %s ORDER BY title, id;",
            params=(self.sess_id,),
        )
        return [row[0] for row in rows]

    def all_attachments_metadata(self) -> list[tuple[Any, Any]] | None:
        """Return a list of all attachments data in the current session."""
        rows = fetch_all(
            conn=self.conn,
            sql=("""
                SELECT created_at, metadata
                FROM knowledge_base
                WHERE session_id = %s AND type = %s
                    AND (metadata->>'document_chunk_index')::int = 0
                """),
            params=(self.chat_sess_logs.get_session_id(), "document"),
        )

        if not rows:
            app_log.debug(
                "Failed to retrieve any documents metadata: "
                "Session '%s' knowledge base is empty",
                self.chat_sess_logs.sess_name,
            )
        return rows

    def query_session_attachments(
        self,
        qry_embeddings: list[float],
        retrieve_lim: int = RETRIEVE_DOC_ENTRY_LIMIT,
        min_sim_score: float | None = MIN_SIMILARITY_SCORE,
    ) -> list[dict[str, Any]]:
        """
        Closest chunks of the current session's attachments, best first.
        Returns [] if there are none or none reach min_sim_score.
        """
        fetch_one(
            conn=self.conn,
            sql="SELECT set_config('hnsw.ef_search', %s, false);",
            params=(str(self._ef_search),),
        )
        rows = fetch_all(
            conn=self.conn,
            sql=_SEARCH_SQL,
            params=(
                qry_embeddings,
                self.sess_id,
                EMBED_MODEL,
                qry_embeddings,
                retrieve_lim,
            ),
        )

        hits = []
        for title, cont, meta, tkns, dist in rows:
            score = 1 - float(dist)
            if min_sim_score is not None and score < min_sim_score:
                break
            hits.append(
                {
                    "title": title,
                    "content": cont,
                    "similarity": score,
                    "tokens": tkns,
                    "metadata": meta or {},
                }
            )

        if not hits:
            app_log.debug(
                "No attachment chunks retrieved: nothing attached in this"
                " session, or nothing reached the minimum similarity score"
            )
        return hits
