"""
Expert class that contains chunked documents and its embeddings, where they
can be searched by vector similarity.

Document handling:
    - Handles storing documents only once.
    - A document can be linked to muiltiple experts.

Relation with chat sessions:
    - Exist independently of chat sessions.
    - Can be enabled in more than one session.
"""

from typing import Sequence, Any
from pathlib import Path

from src.config.config_toml import (
    MODEL,
    EMBED_MODEL,
    EXPERT_RETRIEVE_LIMIT,
    PER_EXPERT_RETRIEVAL_LIMIT,
    ENABLE_DOCLING,
    MIN_SIMILARITY_SCORE,
)

from src.libs.helpers.database_helpers import fetch_one, fetch_all, execute
from src.libs.custom_dataclasses import AgentHooks
from src.libs.helpers.string_helpers import hash_string
from src.libs.helpers.file_metadata_helpers import unpack_file_metadata

from ._schema import ensure_schema, ensure_link_schema
from ._get_or_create import get_or_create_expert
from .query_expert import QueryExpert, EXPERT_RESERVE_TKNS
from .documents import ExpertDocuments, StoreStatus

from src.agent.tokenizers import LocalTokenizer
from src.agent.models.llm import model_response_with_new_system_prompt_and_context
from src import logger

app_log = logger.app_logger(f"{__name__}.app")


class Expert:
    """
    Created on first use (one expert per class), loaded on every use after
    that.

    Parameters
    ----------
        conn:     Connection to the database.
        exp_name: Name of this expert.
        des:      Description for this expert (optional).
        hooks:    Function hooks for cli/tui integration.
    """

    def __init__(
        self,
        conn,
        exp_name: str,
        des: str | None = None,
        hooks: AgentHooks | None = None,
    ):
        self.conn = conn
        self.exp_name = exp_name

        ensure_schema(self.conn)
        self.exp_id = get_or_create_expert(self.conn, self.exp_name, des, hooks=hooks)

        self.qry_exp = QueryExpert(conn=self.conn)
        self.exp_doc = ExpertDocuments(
            conn=self.conn, exp_name=self.exp_name, exp_id=self.exp_id
        )

    @staticmethod
    def exists(conn, exp_name: str) -> bool:
        """
        Check by name whether an expert exists, without creating it.

        Parameters
        ----------
            conn:     Connection to the database.
            exp_name: Expert's name to query.

        Returns
        -------
            - If the expert exists.
        """
        ensure_schema(conn)
        return (
            fetch_one(
                conn=conn,
                sql="SELECT 1 FROM experts WHERE name = %s;",
                params=(exp_name,),
            )
            is not None
        )

    def delete_expert(self) -> bool:
        """
        Delete the entry that matches self.exp_id from the experts table
        in the database.

        The foreign keys remove only this expert's link rows (expert_documents
        and session_experts). Documents and chunks are not touched.

        NO DOCUMENTS ARE DELETED HERE.

        Returns
        -------
            - If the delete action is done.
        """
        return (
            execute(
                conn=self.conn,
                sql=("""
                DELETE FROM experts
                WHERE id = %s;
                """),
                params=(self.exp_id,),
            )
            > 0
        )

    @staticmethod
    def all_experts_data(conn) -> dict[str, dict[str, Any]]:
        """
        Return relevant data about all the created experts, keyed by expert
        name, including how many documents each one is linked to.

        Parameters
        ----------
            conn: Connection to the database.

        Returns
        -------
            - Structured dictionary about all existing experts.
        """
        ensure_schema(conn)
        rows = fetch_all(
            conn=conn,
            sql=("""
                SELECT e.name, e.description, e.created_at, e.updated_at,
                    COUNT(ed.document_id)
                FROM experts e
                LEFT JOIN expert_documents ed ON ed.expert_id = e.id
                GROUP BY e.id
                ORDER BY e.name;
                """),
        )

        return {
            row[0]: {
                "description": row[1],
                "created_at": row[2],
                "updated_at": row[3],
                "document_count": row[4],
            }
            for row in rows
        }

    @staticmethod
    def query_multiple_experts(
        conn,
        prompt: str,
        exp_list: list[str],
        tkn_budget: int | None = None,
    ) -> list[dict]:
        """
        Closest chunks across the experts, best first, as one flat list.

        The retrieval limit depends on if more than one expert is enabled,
        so a large expert can't crowd out a smaller one.

        Parameters
        ----------
            conn:       Connection to the database.
            prompt:     The string that will be used to perform vector
                        search on the expert's database.
            exp_list:   A list of experts to query from.
            tkn_budget: Token limit to avoid agent context overflow.

        Returns
        -------
            - 
        """
        retrieve_lim = (
            PER_EXPERT_RETRIEVAL_LIMIT if len(exp_list) > 1 else EXPERT_RETRIEVE_LIMIT
        )

        hits = []
        for exp_name in exp_list:
            exp = Expert(conn=conn, exp_name=exp_name)
            hits.extend(exp._query_expert(prompt=prompt, retrieve_lim=retrieve_lim))

        hits.sort(key=lambda h: h["distance"])

        # Apply the token budget across all experts
        if tkn_budget is not None:
            kept, used = [], 0
            for hit in hits:
                if used + hit["tokens"] <= tkn_budget:
                    kept.append(hit)
                    used += hit["tokens"]
            hits = kept
        return hits

    def _query_expert(
        self,
        prompt: str,
        retrieve_lim: int,
        tkn_budget: int | None = None,
    ) -> list[dict]:
        """
        Search this expert's documents for the chunks closest to the
        prompt.

        Parameters
        ----------
            prompt:       The string that will be used to perform vector
                          search on the expert's database.
            retrieve_lim: Maximum entries returned from teh query.
            tkn_budget:   Token limit to avoid agent context overflow.

        Returns
        -------
            - A list of dictionary about the matched entries.
        """
        return self.qry_exp.query_from_prompt(
            prompt=prompt, exp_ids=[self.exp_id], tkn_budget=tkn_budget
        )


def _document_entry(row: list) -> dict[str, Any]:
    """
    Format a document row into a labelled dictionary.

    The row columns must follow:

    Parameters
    ----------
        row: Row columns fetched from the database (id, title, type,
             description, created_at, updated_at, metadata).

    Returns
    -------
        - Structured dictionary from the row columns.
    """
    return {
        "title": row[1],
        "type": row[2],
        "description": row[3],
        "created_at": row[4],
        "updated_at": row[5],
        "metadata": row[6],
    }
