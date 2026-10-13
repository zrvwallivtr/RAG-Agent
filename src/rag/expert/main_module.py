from typing import Sequence, Any

from src.config.models import EMBED_MODEL, MODEL_MAX_TOKENS
from src.config.expert import PER_EXPERT_RETRIEVAL_LIMIT

from src.libs.helpers.database_helpers import fetch_one, execute
from src.libs.custom_dataclasses import AgentHooks

from .get_or_create import get_or_create_expert
from .query_experts import QueryExperts
from .store_documents import StoreDocuments

from src.rag.expert.schema import ensure_link_schema, ensure_schema
from src import logger

app_log = logger.app_logger(f"{__name__}.app")


def query_multiple_experts(
    conn,
    prompt: str,
    exp_ids: Sequence[int],
    per_exp_lim = PER_EXPERT_RETRIEVAL_LIMIT,
):
    QueryExperts(conn).query_from_prompt(
        prompt=prompt,
        exp_ids=exp_ids,
        per_exp_lim=per_exp_lim
    )


class Expert:
    """
    Created on first use, loaded on every use after that, so building the
    same Expert twice is safe.
    """

    def __init__(
        self,
        conn,
        exp_name: str,
        des: str | None = None,
    ):
        self.conn = conn
        self.exp_name = exp_name

        ensure_schema(self.conn)
        ensure_link_schema(self.conn)
        self.exp_id = get_or_create_expert(self, des)

        self.qry_exp = QueryExperts(conn=self.conn)
        self.store_doc = StoreDocuments(
            conn=self.conn, exp_name=self.exp_name, exp_id=self.exp_id
        )

    # --- Query ---------------------------------------------------------------

    def query_expert(
        self,
        prompt: str,
        exp_ids: Sequence[int],
    ):
        if len(exp_ids) > 1:
            per_exp_lim = PER_EXPERT_RETRIEVE_LIMIT
        else:
            per_exp_lim = None

        self.qry_exp.query_from_prompt(
            prompt=prompt,
            exp_ids=exp_ids,
            per_exp_lim=per_exp_lim
        )

    # --- Store ----------------------------------------------------------------

    def store_documents(
        self,
        docs: dict[str, dict[Any, Any]],
        hooks: AgentHooks | None = None,
    ):
        """
        docs contant dictionary for document.

        Per document dictionary:
        "title": {
            "doc_type": ,
            "cont_hash": ,
            "chunks": [],
            "path": ,
            "metadata": {},
        }
        """
        for title, data in docs.items():
            doc_type = data.get("doc_type")
            cont_hash = data.get("cont_hash")
            chunks = data.get("chunks")
            path = data.get("path")
            metadata = data.get("metadata")

            doc_id = self.store_doc.add_entire_document_to_expert(
                title=title,
                doc_type=doc_type,
                cont_hash=cont_hash,
                chunks=chunks,
                path=path,
                metadata=metadata,
                hooks=hooks
            )
