from typing import Sequence, Any
from pathlib import Path

from src.config.config_toml import (
    MODEL,
    EMBED_MODEL,
    MODEL_MAX_TOKENS,
    PER_EXPERT_RETRIEVAL_LIMIT,
    ENABLE_DOCLING,
)

from src.libs.helpers.database_helpers import fetch_one, execute
from src.libs.custom_dataclasses import AgentHooks
from src.libs.helpers.string_helpers import hash_string
from src.libs.helpers.file_metadata_helpers import unpack_file_metadata

from .get_or_create import get_or_create_expert
from .query_experts import QueryExperts
from .store_documents import StoreDocuments

from src.rag.file_parser import FileParser
from src.rag.docling import DoclingParser
from src.rag.docling import DoclingChunker
from src.agent.tokenizers import LocalTokenizer
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


def _format_to_document_dictionary(
    path: Path, chunks: list[str], cont_hash: str
) -> dict[str, dict[Any, Any]] | None:
    """Format into the expected dictionary structure for store_documents()."""
    file_data = unpack_file_metadata(path)
    if not file_data:
        return
    name, mime, size = file_data

    return {
        name: {
            "format": format,
            "content_hash": cont_hash,
            "chunks": chunks,
            "path": str(path.resolve()),
            "metadata": {
                "mime_type": mime,
                "size_bytes": size,
                "page_count": "",
            }
        }
    }


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
        self.local_tknizr = LocalTokenizer(MODEL)
        self.file_parser = FileParser()
        self.chunker = DoclingChunker(self.local_tknizr)

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
            per_exp_lim = PER_EXPERT_RETRIEVAL_LIMIT
        else:
            per_exp_lim = None

        self.qry_exp.query_from_prompt(
            prompt=prompt,
            exp_ids=exp_ids,
            per_exp_lim=per_exp_lim
        )

    # --- Store ----------------------------------------------------------------

    def _store_documents(
        self,
        doc_dict: dict[str, dict[Any, Any]],
        hooks: AgentHooks | None = None,
    ):
        """
        docs contant dictionary for document.

        Per document dictionary:
        "title": {
            "format": ,
            "cont_hash": ,
            "chunks": [],
            "path": ,
            "metadata": {},
        }
        """
        for title, data in doc_dict.items():
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

    def chunk_document_to_database(
        self,
        path: Path,
        is_docling: bool = ENABLE_DOCLING,
        hooks: AgentHooks | None = None
    ):
        """
        Parse the document for the docling chunker, using the docling
        chunker to chunk the document and store to the database.
        """
        out = self.file_parser.read_document(path)
        if not out:
            return
        cont, format = out

        chunks = self.chunker.chunking(cont)
        cont_hash = hash_string(cont)
        doc_dict = _format_to_document_dictionary(path, chunks, cont_hash)
        if not doc_dict:
            return

        doc_id = self._store_documents(doc_dict, hooks=hooks)
