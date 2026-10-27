from typing import Any
from pathlib import Path

from src.config.config_toml import (
    RETRIEVE_DOC_ENTRY_LIMIT,
    MIN_SIMILARITY_SCORE,
    EXPLORATION_FACTOR,
    EMBED_MODEL,
)

from ._schema import ensure_schema

from src.libs.helpers.database_helpers import fetch_one, fetch_all, execute
from src.libs.custom_dataclasses import AgentHooks
from src.libs.helpers.string_helpers import hash_string

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

_SEARCH_SPECIFIED_SQL = """
    SELECT a.title, c.content, c.metadata, c.prompt_tokens,
        c.embeddings <=> %s::vector AS distance
    FROM attachment_chunks c
    JOIN attachments a ON a.id = c.attachment_id
    WHERE a.session_id = %s AND a.embedding_model = %s
        AND a.content_hash = ANY(%s)
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
        self._ef_search = ef_search
        ensure_schema(conn)

    @property
    def sess_id(self):
        return self.chat_sess_logs.get_session_id()

    def all_attachments_titles(self) -> list[str]:
        """Titles of all attachments in the current session."""
        if not self.sess_id:
            return []
        rows = fetch_all(
            conn=self.conn,
            sql="SELECT title FROM attachments WHERE session_id = %s ORDER BY title, id;",
            params=(self.sess_id,),
        )
        return [row[0] for row in rows]

    def query_session_attachments(
        self,
        qry_embeddings: list[float],
        retrieve_lim: int = RETRIEVE_DOC_ENTRY_LIMIT,
        min_sim_score: float | None = MIN_SIMILARITY_SCORE,
        hooks: AgentHooks | None = None,
    ) -> list[dict[str, Any]]:
        """
        Closest chunks of the current session's attachments, best first.
        Returns [] if there are none or none reach min_sim_score.
        """
        if not self.sess_id:
            return []

        if hooks and hooks.on_loading:
            info_msg = "Querying session attachments..."
            hooks.on_loading(info_msg)

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

    def query_specified_session_attachments(
        self,
        qry_embeddings: list[float],
        attchmnt_dict: dict[Path, dict[str, str]] | None,
        attchmnt_chunks_lim: int = 10,
        tkn_budget: int | None = None,
        hooks: AgentHooks | None = None,
    ) -> list[dict[str, Any]] | None:
        """
        Closest chunks of the given (already stored) attachments, best first,
        limited to tkn_budget tokens. Returns None if none are found.
        """
        if not attchmnt_dict or not self.sess_id:
            return None

        if hooks and hooks.on_loading:
            hooks.on_loading("Retrieving relevant attachment content...")

        hashes = [hash_string(data["content"]) for data in attchmnt_dict.values()]

        paths = attchmnt_dict.keys()
        fetch_one(
            conn=self.conn,
            sql="SELECT set_config('hnsw.ef_search', %s, false);",
            params=(str(self._ef_search),),
        )
        rows = fetch_all(
            conn=self.conn,
            sql=_SEARCH_SPECIFIED_SQL,
            params=(
                qry_embeddings,
                self.sess_id,
                EMBED_MODEL,
                hashes,
                qry_embeddings,
                attchmnt_chunks_lim,
            ),
        )

        hits, used = [], 0
        for title, cont, meta, tkns, dist in rows:
            # Skip chunks that don't fit
            if tkn_budget is not None and used + tkns > tkn_budget:
                continue

            used += tkns
            hits.append(
                {
                    "title": title,
                    "content": cont,
                    "similarity": 1 - float(dist),
                    "tokens": tkns,
                    "metadata": meta or {},
                }
            )
        if not hits:
            app_log.debug("No chunks retrieved from the specified attachments")
            return None
        return hits
