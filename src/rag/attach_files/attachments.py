import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config.config_toml.rag import ENABLE_DOCLING
from src.config.config_toml import MODEL, EMBED_MODEL

from ._schema import ensure_schema

from src.libs.helpers.database_helpers import (
    fetch_one,
    fetch_all,
    execute,
    execute_many,
    transaction,
)
from src.libs.helpers.string_helpers import hash_string
from src.libs.helpers.file_metadata_helpers import unpack_file_metadata
from src.libs.custom_dataclasses import AgentHooks

from src.rag.docling import DoclingChunker
from src.agent.chat_session_logs import ChatSessionLogs
from src.agent.models.embed import embedding_content
from src.agent.tokenizers import LocalTokenizer
from src.rag.file_reader import FileReader
from src.logger import app_logger

app_log = app_logger(f"{__name__}.app")


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
        file_reader: FileReader | None = None,
        chunker: DoclingChunker | None = None,
        tknizr=LocalTokenizer(MODEL),
    ):
        self.conn = conn

        self.sess_id = chat_sess_logs.get_session_id()
        if not self.sess_id:
            raise ValueError("Attachments need an existing chat session.")
        self.sess_name = chat_sess_logs.get_session_name(self.sess_id)

        ensure_schema(self.conn)

        self.file_reader = file_reader
        self.chunker = chunker
        self.local_tknizr = tknizr

    @staticmethod
    def attachment_id_by_title(conn, attchmnt_name: str | int) -> int | None:
        """
        Resolve a attachment title to a attachment id.
        """
        rows = fetch_all(
            conn=conn,
            sql="SELECT id FROM attachments WHERE title = %s ORDER BY id;",
            params=(attchmnt_name,),
        )

        if len(rows) == 1:
            return int(rows[0][0])

        if len(rows) > 1:
            raise ValueError(
                "%d attachments are named '%s'. Use the id instead"
                % (len(rows), attchmnt_name)
            )

    def delete_attachment_by_id(self, attchmnt_id: int) -> bool:
        """Delete an attachment and its chunks from this session."""
        return (
            execute(
                conn=self.conn,
                sql="DELETE FROM attachments WHERE id = %s AND session_id = %s;",
                params=(attchmnt_id, self.sess_id),
            )
            > 0
        )

    def clear_session_attachments(self) -> int:
        """Delete every attachments of this session. Returns the number of deletions."""
        return execute(
            conn=self.conn,
            sql="DELETE FROM attachments WHERE session_id = %s;",
            params=(self.sess_id,),
        )

    def all_attachments_data_from_session(self) -> dict[int, dict[str, Any]]:
        """Return all attachments that are linked to the current session."""
        rows = fetch_all(
            conn=self.conn,
            sql=("""
            SELECT id, title, type, source_path, embedding_model,
                uploaded_at, metadata
            FROM attachments
            WHERE session_id = %s
            ORDER BY title, id;
            """),
            params=(self.sess_id,),
        )
        return {row[0]: _attachment_entry(row) for row in rows}

    @staticmethod
    def all_attachments_data_across_sessions(conn) -> dict[int, dict[str, Any]]:
        """Return every attachments, keyed by attachments id."""
        ensure_schema(conn)
        rows = fetch_all(
            conn=conn,
            sql=("""
            SELECT a.id, a.title, a.type, a.source_path, a.embedding_model, 
                a.uploaded_at, a.metadata, a.session_id
            FROM attachments a
            JOIN chat_sessions s ON s.session_id = a.session_id
            ORDER BY s.session_name, a.title, a.id;
            """),
        )
        return {
            row[0]: {
                **_attachment_entry(row),
                "session_name": row[7],
            }
            for row in rows
        }

    def _chunk_attachment(
        self, path: Path, cont: str, hooks: AgentHooks | None = None
    ) -> dict:
        """Chunk and package an attachment."""
        if not self.chunker:
            return {}
        chunks = self.chunker.chunking(cont, hooks=hooks)
        if not chunks:
            app_log.warning("No chunks produced from attachment '%s'. Skipping", path)
            return {}

        attchmnt = _format_to_attachment_dictionary(
            path, cont, chunks, hash_string(cont)
        )
        if not attchmnt:
            app_log.warning("No file metadata for attachment '%s'. Skipping", path)
            return {}
        return attchmnt

    def store_already_read_attachments(
        self,
        attchmnt_dict: dict[Path, dict[str, str]],
        hooks: AgentHooks | None = None,
    ) -> dict[str, int]:
        """Chunk and store already read attachments."""
        all_attchmnts = {}
        for path, data in attchmnt_dict.items():
            all_attchmnts.update(self._chunk_attachment(path, data["content"], hooks=hooks))
        return self._store_all_attachments(all_attchmnts, hooks=hooks)

    def store_all_attachments_to_session(
        self,
        paths: list[Path],
        hooks: AgentHooks | None = None,
    ) -> dict[str, int] | None:
        """
        Read the attachment, chunk it, embed the chunks and store everything
        to the database labelled by the current session.

        Returns {title: (attachment id)}, or None if the attachment could not
        be read,, has no content, or has no readable metadata.
        """
        if not self.file_reader:
            return None
        all_attchmnts = {}

        for path in paths:
            try:
                out = self.file_reader.read_document(path)
            except (FileNotFoundError, ValueError) as e:
                app_log.warning("Failed to read attachment '%s': %s. Skipping", path, e)
                continue
            if not out:
                app_log.warning("Failed to read attachment '%s'. Skipping", path)
                continue
            cont, _ = out

            attchmnt = self._chunk_attachment(path, cont, hooks=hooks)
            all_attchmnts.update(attchmnt)

        return self._store_all_attachments(all_attchmnts, hooks=hooks)

    def _get_attachment_title_by_id(self, id: int) -> str | None:
        row = fetch_one(
            conn=self.conn,
            sql="SELECT title FROM attachments WHERE id = %s;",
            params=(id,),
        )
        return str(row[0]) if row else None

    def _get_attachment_id_by_hash(self, cont_hash: str) -> int | None:
        row = fetch_one(
            conn=self.conn,
            sql="SELECT id FROM attachments WHERE content_hash = %s AND session_id = %s;",
            params=(cont_hash, self.sess_id),
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
            if isinstance(chunk, str):
                chunk = {"text": chunk}
            text = chunk.get("text", "")
            if text.strip():
                normalised.append(
                    {"text": chunk["text"], "metadata": chunk.get("metadata") or {}}
                )
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
                hooks.on_loading("Embedding chunk %d/%d..." % (i, tol))

            result = embedding_content(chunk["text"])
            if not result:
                raise RuntimeError("Failed to embed chunk %d/%d" % (i, tol))
            _, vector, embed_tkns = result

            embed_results.append((vector, embed_tkns or 0))
        return embed_results

    def _add_attachment_to_session(
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
                ON CONFLICT (session_id, content_hash) DO NOTHING
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
                    (attachment_id, chunk_index, embeddings, embed_tokens,
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

    def _add_entire_attachment(
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
    ) -> int:
        """
        Store an attachment to the database, with the session_id labelled to
        the current session. Returns (attachment_id, status).

        New content is embedded first, then the attachment row, every chunk and
        the expert link are written in ONE transaction, so failure leaves
        nothing behind. Content is already stored is not embedded again: it is
        just linked to this expert.
        """
        chunks = self._formatted_chunks_with_data(chunks)
        if not chunks:
            raise ValueError("Cannot add content with no chunks.")

        # Check if content has already been stored
        existing_id = self._get_attachment_id_by_hash(cont_hash)
        if existing_id is not None:
            existing_name = self._get_attachment_title_by_id(existing_id)
            info_msg = (
                "Attachment '%s' has already been stored in this session '%s' as '%s'"
                % (
                    title,
                    self.sess_name,
                    existing_name,
                )
            )
            app_log.info(info_msg)
            if hooks and hooks.on_stat_change:
                hooks.on_stat_change(info_msg)
            return existing_id

        # Embed before opening the transaction, so it isn't held open during
        # the slow part
        embed_results = self._embed_every_chunks_in_attachment(chunks, hooks)

        doc_id = None
        with transaction(self.conn):
            doc_id = self._add_attachment_to_session(
                title=title,
                attchmnt_type=attchmnt_type,
                cont_hash=cont_hash,
                cont=cont,
                path=path,
                metadata=metadata,
            )
            if doc_id is not None:
                self._add_all_chunks_from_attachment(doc_id, chunks, embed_results)

        # Another process stored the same content embedding
        if doc_id is None:
            existing = self._get_attachment_id_by_hash(cont_hash)
            if existing is None:
                raise RuntimeError("Document '%s' vanished during insert." % title)
            return existing

        info_msg = "Added '%s' (%d chunks) to session '%s'" % (
            title,
            len(chunks),
            self.sess_name,
        )
        app_log.info(info_msg)
        if hooks and hooks.on_stat_change:
            hooks.on_stat_change(info_msg)
        return doc_id

    def _store_all_attachments(
        self,
        doc_dict: dict[str, dict[Any, Any]],
        hooks: AgentHooks | None = None,
    ) -> dict[str, int]:
        """
        Store every attachments in the dictionary (see
        _format_to_attachment_dictionary for its structure) and link it to this
        expert.

        Returns {title: (attachment id)}.
        """
        results = {}

        for title, data in doc_dict.items():
            try:
                results[title] = self._add_entire_attachment(
                    title=title,
                    attchmnt_type=data["type"],
                    cont_hash=data["cont_hash"],
                    cont=data["cont"],
                    chunks=data["chunks"],
                    path=data["path"],
                    metadata=data["metadata"],
                    hooks=hooks,
                )
            except RuntimeError as e:
                app_log.warning(
                    "Failed to store attachment '%s': %s. Skipping", title, e
                )
        return results


def _attachment_entry(row: list) -> dict[str, Any]:
    """
    Format a attachment row into a labelled dictionary.

    The row columns must follow:
    id, title, type, description, updated_at, metadata.
    """
    return {
        "title": row[1],
        "type": row[2],
        "source_path": row[3],
        "embedding_model": row[4],
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
        "type": ,
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
            "type": path.suffix.lstrip(".").lower()[:20],
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
