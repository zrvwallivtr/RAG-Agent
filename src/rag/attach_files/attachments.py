import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.config.config_toml import MODEL, EMBED_MODEL

from ._schema import ensure_schema

from src.libs.helpers.database_helpers import (
    fetch_one,
    fetch_all,
    execute,
    execute_many,
)
from src.libs.helpers.string_helpers import hash_string
from src.libs.helpers.file_metadata_helpers import unpack_file_metadata
from src.libs.custom_dataclasses import AgentHooks

from src.agent.chat_session_logs import ChatSessionLogs
from src.agent.models.embed import embedding_content
from src.agent.tokenizers import LocalTokenizer
from src.rag.file_reader import FileReader
from src.logger import app_logger

app_log = app_logger(f"{__name__}.app")


def _attachment_entry(row: list) -> dict[str, Any]:
    """
    Format a attachment row into a labelled dictionary.

    The row columns must follow:
    id, title, type, description, created_at, updated_at, metadata.
    """
    return {
        "title": row[1],
        "type": row[2],
        "description": row[3],
        "created_at": row[4],
        "updated_at": row[5],
        "metadata": row[6],
    }


def _format_to_attachment_dictionary(
    path: Path, cont: str, chunks: list, cont_hash: str
):
    """
    Format into the expected dictionary structure for _store_attachments().
    Return None if the file's metadata can't be read.

    "title": {
        "att_type": ,
        "cont_hash": ,
        "cont": ,
        "chunks": [],
        "path": ,
        "metadata": {},
    }
    """
    file_data = unpack_file_metadata(path)
    if not file_data:
        return
    name, mime, size = file_data

    return {
        name: {
            "att_type": path.suffix.lstrip(".").lower()[:20],
            "cont_hash": cont_hash,
            "cont": cont,
            "chunks": chunks,
            "path": str(path.resolve()),
            "metadata": {
                "mime_type": mime,
                "size_bytes": size,
            },
        }
    }


class Attachments:
    """
    Stores attachments and links them to ONE session. Require chat session logs
    classes and database tables to exists first and should be removed when the
    associated session is deleted.

    self.conn:           Connection to postgres.
    self.chat_sess_logs: Existing chat session log class.
    self.file_reader:    For reading uploaded files.
    self.local_tknizr:   LLM token managements.
    """

    def __init__(
        self,
        conn,
        chat_sess_logs: ChatSessionLogs,
        file_reader: FileReader,
    ):
        self.conn = conn

        self.sess_id = chat_sess_logs.get_session_id()
        if self.sess_id:
            self.sess_name = chat_sess_logs.get_session_name(self.sess_id)

        ensure_schema(self.conn)

        self.file_parser = file_reader
        self.local_tknizr = LocalTokenizer(MODEL)

    def _get_attachment_id_by_hash(self, cont_hash: str) -> int | None:
        row = fetch_one(
            conn=self.conn,
            sql="SELECT id FROM attachments WHERE content_hash = %s AND session_id = %s;",
            params=(cont_hash,),
        )
        return int(row[0]) if row else None

    @staticmethod
    def _formatted_chunks_with_data(chunks: list) -> list[dict]:
        """
        Accept plain strings or {"text": str, "metadata": dict} dictionaries.
        Always returns [{"text": str, "metadata": dict}].
        """
        normalised = []
        for chunk in chunks:
            text = chunk["text"]
            normalised.append({"text": text, "metadata": chunk.get("metadata") or {}})
        return normalised

    def _embed_every_chunks_in_attachment(
        self, chunks: list[dict], hooks: AgentHooks | None = None
    ) -> list[tuple[list[float], int]]:
        """
        Embed every chunk, in order. Returns (vector, embed_tokens) per chunk.
        Raises if any chunk fails, before anything is written.
        """
        embed_results = []
        tol = len(chunks)

        for i, chunk in enumerate(chunks, start=1):
            if hooks and hooks.on_loading:
                hooks.on_loading(f"Embedding chunk %d/%d..." % (i, tol))

            result = embedding_content(chunk["text"])
            if not result:
                raise RuntimeError("Failed to embed chunk %d/%d" % (i, tol))
            _, vector, embed_tkns = result

            embed_results.append((vector, embed_tkns or 0))
        return embed_results

    def _add_attachment(
        self,
        title: str,
        attchmnt_type: str,
        cont_hash: str,
        cont: str,
        path: str | None = None,
        metadata: dict | None = None,
    ) -> int | None:
        """
        Insert the attachment row. Returns None if the hash already exists (e.g.
        another process stored it first). Does NOT commit: call insisde
        transaction().
        """
        row = fetch_one(
            conn=self.conn,
            sql=("""
                INSERT INTO attachments
                    (title, type, content_hash, source_path, content,
                    embedding_model, session_id, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (content_hash) DO NOTHING
                RETURNING id;
                """),
            params=(
                title,
                attchmnt_type,
                cont_hash,
                path,
                cont,
                EMBED_MODEL,
                self.sess_id,
                json.dumps(metadata or {}),
            ),
            commit=False,
        )
        return int(row[0]) if row else None

    def _add_all_chunks_from_attachment(
        self,
        doc_id: int,
        chunks: list[dict],
        embed_results: list[tuple[list[float], int]],
    ) -> None:
        """
        Insert every chunk with its embedding.

        Does NOT commit: call inside transaction().
        """
        execute_many(
            conn=self.conn,
            sql=("""
                INSERT INTO attachment_chunks
                    (document_id, chunk_index, embeddings, embed_tokens,
                    content, prompt_tokens, metadata)
                VALUES (%s, %s, %s::vector, %s, %s, %s, %s);
                """),
            params_seq=[
                (
                    doc_id,
                    i,
                    vec,
                    embed_tkns,
                    chunk["text"],
                    self.local_tknizr.count_string_tokens(chunk["text"]) or 0,
                    json.dumps(chunk["metadata"]),
                )
                for i, (chunk, (vec, embed_tkns)) in enumerate(
                    zip(chunks, embed_results)
                )
            ],
            commit=False,
        )

    def _add_entire_attachment_to_expert(
        self,
        *,
        title: str,
        attchmnt_type: str,
        cont_hash: str,
        cont: str,
        chunks: list,
        path: str | None = None,
        metadata: dict | None = None,
        hooks: AgentHooks | None = None,
    ) -> tuple[int, StoreStatus]:
        """
        Store an attachment and link it to the chat session. Returns
        (attachment_id, status).

        New content is embedded first, then the attachment row, every chunk and
        the expert link are written in ONE transaction, so failure leaves
        nothing behind. Content is already stored is not embedded again: it is
        just linked to this expert.
        """
        chunks = self._formatted_chunks_with_data(chunks)
        if not chunks:
            raise ValueError("Cannot add a document with no chunks.")

        # Check if content has already been stored
        existing = self._get_attachment_id_by_hash(cont_hash)
        if existing is not None:
            status = self._link_existing(existing)
            info_msg = "Attachment '%s' already stored (id %d): %s" % (
                title,
                existing,
                status.replace("_", " "),
            )
            app_log.info(info_msg)
            if hooks and hooks.on_stat_change:
                hooks.on_stat_change(info_msg)
            return existing, status

        # Embed before opening the transaction, so it isn't held open during
        # the slow part
        embed_results = self._embed_every_chunks_in_attachment(chunks, hooks)

        doc_id = None
        with transaction(self.conn):
            doc_id = self._add_attachment(
                title=title,
                attchmnt_type=attchmnt_type,
                cont_hash=cont_hash,
                cont=cont,
                path=path,
                metadata=metadata,
            )
            if doc_id is not None:
                self._add_all_chunks_from_attachment(doc_id, chunks, embed_results)
                self._link(doc_id, commit=False)

        # Another process stored the same content embedding
        if doc_id is None:
            existing = self._get_attachment_id_by_hash(cont_hash)
            if existing is None:
                raise RuntimeError("Document '%s' vanished during insert." % title)
            return existing, self._link_existing(existing)

        info_msg = "Added '%s' (%d chunks) to session '%s'" % (
            title,
            len(chunks),
            self.sess_name,
        )
        app_log.info(info_msg)
        if hooks and hooks.on_stat_change:
            hooks.on_stat_change(info_msg)
        return doc_id, "added"

    def _store_documents(
        self,
        doc_dict: dict[str, dict[Any, Any]],
        hooks: AgentHooks | None = None,
    ) -> dict[str, tuple[int, StoreStatus]]:
        """
        Store every document in the dictionary (see
        _format_to_document_dictionary for its structure) and link it to this
        expert.

        Returns {title: (document id, status)}.
        """
        results = {}

        for title, data in doc_dict.items():
            results[title] = self._add_entire_attachment(
                title=title,
                doc_type=data["doc_type"],
                cont_hash=data["cont_hash"],
                cont=data["cont"],
                chunks=data["chunks"],
                path=data["path"],
                metadata=data["metadata"],
                hooks=hooks,
            )
        return results

    def _is_doc_cont_chunk_exist(
        self, new_hash: str
    ) -> tuple[str, list[float], int] | None:
        """
        Check if the same document content chunk was uploaded before (same
        hash, accross sessions). Return embeddings if a duplicate exist.
        """
        row = fetch_one(
            conn=self.conn,
            sql=("""
                SELECT content, embeddings, embed_tokens
                FROM knowledge_base
                WHERE content_hash = %s
                LIMIT 1
                """),
            params=(new_hash,),
        )
        if row is None:
            return None
        return row

    def _upload_to_session(
        self,
        doc_name: str,
        chnk_idx: int,
        embdings: list[float],
        embed_tkns: int,
        cont: str,
        prompt_tkns: int,
        mime: str,
        size: int,
        cont_hash: str,
        exprs_at: datetime | None = None,
        hooks: AgentHooks | None = None,
    ) -> None:
        """Upload a document to knowledge base."""
        metadata = _attachment_entry()

        row = fetch_one(
            conn=self.conn,
            sql=("""
                INSERT INTO knowledge_base 
                    (session_id, type, embeddings, embed_tokens, content,
                    prompt_tokens, content_hash, expires_at, metadata)
                VALUES (%s, %s, %s::vector, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """),
            params=(
                self.chat_sess_logs.get_session_id(),
                "document",
                str(embdings),
                embed_tkns,
                cont,
                prompt_tkns,
                cont_hash,
                exprs_at,
                json.dumps(metadata),
            ),
        )

        if row:
            info_str = "Added document '%s' to session '%s' knowledge base" % (
                doc_name,
                self.chat_sess_logs.sess_name,
            )
            app_log.info(info_str)
            if hooks and hooks.on_stat_change:
                hooks.on_stat_change(info_str)
            return

    def add_embed_to_session(
        self,
        path: Path,
        cont: str | None,
        format: str | None,
        hooks: AgentHooks | None = None,
    ) -> int | None:
        """
        Uses langchain text splitters for file format accordingly,
        embed each chunks and save to knowledge base.
        """
        file_data = unpack_file_metadata(path)
        if not file_data:
            return
        name, mime, size = file_data

        chnks = []

        if not cont and not format:
            out = self.file_parser.read_document(path)
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
                    "Document already exists in session '%s' knowledge base."
                    " Skipping re-embed",
                    self.chat_sess_logs.sess_name,
                )
                chnk_cont, embdings, embed_tkns = doc_chnk
                prompt_tkns = self.local_tknizr.count_string_tokens(chnk_cont)
                if not prompt_tkns:
                    return None

            else:
                response = embedding_content(chnk)
                if not response:
                    continue
                chnk_cont, embdings, embed_tkns = response
                prompt_tkns = self.local_tknizr.count_string_tokens(chnk_cont)
                if not prompt_tkns:
                    return None

            hash = hash_string(chnk_cont)

            # Skip chunks that failed
            if not embdings:
                app_log.warning(
                    "Error occur in embedding chunk in '%s'. Skipping chunk", path
                )
                continue

            # Skip upload if already exists
            if not self._is_doc_cont_chunk_exist(hash):
                result = self._upload_to_session(
                    doc_name=name,
                    chnk_idx=idx,
                    embdings=embdings,
                    embed_tkns=embed_tkns,
                    cont=chnk_cont,
                    prompt_tkns=prompt_tkns,
                    mime=mime,
                    size=size,
                    cont_hash=chnk_hash,
                    hooks=hooks,
                )
            count += 1

        return count

    def store_documents(
        self,
        attchmnt_dict: dict[Path, dict[str, str]],
        hooks: AgentHooks | None = None,
    ) -> None:
        for doc_path, data in attchmnt_dict.items():
            cont = data["content"]
            format = data["format"]
            count = self.add_embed_to_session(
                path=doc_path, cont=cont, format=format, hooks=hooks
            )
            if not count:
                continue
        return

    def clear_sess_kw_bs(self, cont_type: Literal["document"] = "document") -> bool:
        """Clear all contents related to specified type and current session."""
        app_log.info("Clearing session '%s' knowledge base", self.sess_name)
        row = execute(
            conn=self.conn,
            sql=("DELETE FROM knowledge_base WHERE session_id = %s AND type = %s"),
            params=(self.chat_sess_logs.get_session_id(), cont_type),
        )

        if row == 0:
            app_log.warning(
                "Failed to clear session '%s' knowledge base: "
                "'%s' contents not found in database",
                self.sess_name,
                cont_type,
            )
            return False
        return True
