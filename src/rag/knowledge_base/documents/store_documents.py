import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .document_reader import DocumentReader
from .query_documents import QueryDocuments

from src.libs.helpers.database_helpers import fetch_one, fetch_all, execute
from src.libs.helpers.string_helpers import hash_string

from src.agent.chat_logs import ChatSessionLogs
from src.agent.models.embed import embedding_content
from src.logger import app_logger

app_log = app_logger(f"{__name__}.app")


class StoreDocuments:
    def __init__(
        self, conn, chat_logs: ChatSessionLogs, doc_reader: DocumentReader, query_docs: QueryDocuments
    ):
        self._conn = conn

        self.chat_logs = chat_logs
        self.doc_reader = doc_reader
        self.query_docs = query_docs

    def _is_doc_cont_chunk_exist(self, new_hash: str) -> tuple[str, list[float], int] | None:
        """
        Check if the same document content chunk was uploaded before (same
        hash, accross sessions). Return embeddings if a duplicate exist.
        """
        row = fetch_one(
            conn=self._conn,
            sql=(
                """
                SELECT content, embeddings, prompt_tokens
                FROM knowledge_base
                WHERE content_hash = %s
                LIMIT 1
                """
            ),
            params=(new_hash,)
        )
        if row is None:
            return None

        app_log.debug("Document content chunk already exist")
        return row

    def _upload_to_knowledge_base(
        self,
        doc_name: str,
        chnk_idx: int,
        embdings: list[float],
        chnk_tkns: int,
        cont: str,
        mime: str,
        size: int,
        cont_hash: str,
        exprs_at: datetime | None = None,
    ) -> None:
        """Upload a document to knowledge base."""
        metadata = {
            "document_name": doc_name,
            "document_chunk_index": chnk_idx,
            "mime_type": mime,
            "size_bytes": size
        }

        row = fetch_one(
            conn=self._conn,
            sql=(
                """
                INSERT INTO knowledge_base (session_id, type, embeddings, prompt_tokens, content, content_hash, expires_at, metadata)
                VALUES (%s, %s, %s::vector, %s, %s, %s, %s, %s)
                """
            ),
            params=(
                self.chat_logs.get_session_id(), "document", str(embdings),
                chnk_tkns, cont, cont_hash, exprs_at, json.dumps(metadata),
            )
        )

        if row:
            app_log.info(
                "Added document '%s' to session '%s' knowledge base",
                doc_name, self.chat_logs.sess_name
            )
            return

    def add_embed_to_knowledge_base(
        self, path: Path, cont: str | None, format: str | None
    ) -> int | None:
        """
        Uses langchain text splitters for file format accordingly,
        embed each chunks and save to knowledge base.
        """
        doc_data = self.query_docs.get_document_metadata_from_path(path)
        if not doc_data:
            return
        name, mime, size = doc_data

        chnks = []

        if not cont and not format:
            out = self.doc_reader.read_document(path)
            if not out:
                return
            cont, format = out

        # Langchain text splitters
        from src.rag.splitters import txt_spltr, md_spltr
        if format == "txt":
            chnks = txt_spltr.split_text(cont)
        if format == "md":
            chnks = md_spltr.split_text(cont)

        if not chnks:
            return

        count = 0
        for idx, chnk in enumerate(chnks):
            chnk_hash = hash_string(chnk)
            doc_chnk = self._is_doc_cont_chunk_exist(chnk_hash)

            if doc_chnk:
                # Skip embedding if already exists
                app_log.debug(
                    "Document already exists in session '%s' knowledge base. Skipping re-embed",
                    self.chat_logs.sess_name
                )
                chnk_cont, embdings, chnk_tkns = doc_chnk

            else:
                response = embedding_content(chnk)
                if not response:
                    continue
                chnk_cont, embdings, chnk_tkns = response

            hash = hash_string(chnk_cont)

            # Skip chunks that failed
            if not embdings:
                app_log.warning(
                    "Error occur in embedding chunk in '%s'. Skipping chunk", path
                )
                continue

            # Skip upload if already exists
            if not self._is_doc_cont_chunk_exist(hash):
                result = self._upload_to_knowledge_base(
                    doc_name=name,
                    chnk_idx=idx,
                    embdings=embdings,
                    chnk_tkns=chnk_tkns,
                    cont=chnk_cont,
                    mime=mime,
                    size=size,
                    cont_hash=chnk_hash
                )
            count += 1

        return count

    def store_attachments(self, attchmnt_dict: dict[Path, dict[str, str]]) -> None:
        for doc_path, data in attchmnt_dict.items():
            cont = data["content"]
            format = data["format"]
            count = self.add_embed_to_knowledge_base(
                path=doc_path, cont=cont, format=format
            )
            if not count:
                continue
        return
