from typing import Any

from src.config.memory import RETRIEVE_MEM_ENTRY_LIMIT

from src.agent.chat_logs import ChatSessionLogs
from src.libs.helpers.database_helpers import fetch_one, fetch_all, execute
from src.logger import app_logger

app_log     = app_logger(f"{__name__}.app")

class QueryEntries:

    def __init__(self, conn, chat_logs: ChatSessionLogs):
        self.conn = conn

        self.chat_logs = chat_logs

    def cosine_similarity_search(
        self,
        qry: str,
        qry_embdings: list[float],
        min_sim: float = 0.65
    ) -> list[dict[str, Any]] | None:
        """Queries database for similar content with set minimum similarity score."""
        rows = fetch_all(
            conn=self.conn,
            sql=(
                """
                SELECT content, 1 - (embeddings <=> %s) AS cosine_similarity
                FROM memory
                WHERE 1 - (embeddings <=> %s) >= %s
                ORDER BY embeddings <=> %s ASC
                LIMIT %s;
                """
            ),
            params=(
                str(qry_embdings),
                str(qry_embdings),
                min_sim,
                str(qry_embdings),
                RETRIEVE_MEM_ENTRY_LIMIT
            )
        )

        if not rows:
            app_log.debug(
                "Failed to retrieve relevant memory: "
                "No entry exists in memory or Entry does not reached minimum similarity score"
            )
            return

        return [
            {
                "content": row[0],
                "similarity": float(row[1])
            } for row in rows
        ]

    def query_from_ids(self, ids: list[str]) -> dict[str, str]:
        """Return memory dictionary from a list of ids."""
        mem_dict = {}

        for id in ids:
            row = fetch_one(
                conn=self.conn,
                sql="SELECT content, category FROM memory WHERE id = %s",
                params=(id,)
            )

            if not row:
                app_log.warning(
                    "Failed to retrieve memory entry '%s': Memory does not exist. Skipping",
                    id
                )
                continue

            mem_dict[str(row[0])] = str(row[1])
        return mem_dict
