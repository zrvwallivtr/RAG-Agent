"""
Stores documents in the expert database and links them to an expert.

A document is only stored once: Identified by its content hash to prevent
embedding the same document again.

If the same content for another expert is being stored, the process will only
adds a link.
"""

import json
from typing import Any, Literal
from pathlib import Path

from src.config.prompts import DOC_DES_PROMPT
from src.config.config_toml import (
    MODEL,
    EMBED_MODEL,
    EXPERT_RESERVE_TKNS,
    EXPERT_DOC_FALLBACK_CHARS,
    ENABLE_DOCLING,
)

from src.libs.helpers.database_helpers import (
    fetch_one,
    fetch_all,
    execute,
    execute_many,
    transaction,
)
from src.libs.custom_dataclasses import AgentHooks
from src.libs.helpers.string_helpers import hash_string
from src.libs.helpers.file_metadata_helpers import unpack_file_metadata
from src.libs.custom_dataclasses import AgentHooks

from ._get_or_create import get_expert_id_from_name
from ._schema import ensure_schema

from src.rag.file_reader import FileReader
from src.rag.docling import DoclingChunker
from src.agent.models.llm import model_response_with_new_system_prompt_and_context
from src.agent.models.embed import embedding_content
from src.agent.tokenizers import LocalTokenizer
from src import logger

app_log = logger.app_logger(f"{__name__}.app")

# Document statuses:
#
# added:          New document, embedded and linked to the expert.
# linked:         Document already existed (embedded once before), now linked
#                 to this expert.
# already_linked: Document already existed and was already linked to this
#                 expert.
StoreStatus = Literal[
    "added",
    "linked",
    "already_linked",
]


def _document_entry(row: list) -> dict[str, Any]:
    """
    Format a document row into a labelled dictionary.

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


def _format_to_document_dictionary(
    path: Path, cont: str, chunks: list, cont_hash: str
) -> dict[str, dict[Any, Any]] | None:
    """
    Format into the expected dictionary structure for _store_documents().
    Return None if the file's metadata can't be read.

    "title": {
        "doc_type": ,
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
            "doc_type": path.suffix.lstrip(".").lower()[:20],
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


class ExpertDocuments:
    """
    Stores documents and links them to ONE expert. Can be called without first
    initialising the Expert class.

    A document is stored ONCE (identified by the hash of its parsed content)
    and can be linked to many experts, so adding the same content to a second
    expert reuses the existing chunks and embeddings instead of embedding it
    again.

    self.conn:         Connection to postgres.
    self.exp_name:     Existing expert name.
    self.exp_id:       Optional, specify only from Expert's self.exp_id.
    self.local_tknizr: LLM token managements.
    """

    def __init__(
        self,
        conn,
        exp_name: str,
        exp_id: int | None = None,
        hooks: AgentHooks | None = None,
    ):
        self.conn = conn

        self.hooks = hooks
        self.exp_name = exp_name
        self.exp_id = exp_id or get_expert_id_from_name(self.conn, exp_name)
        if self.exp_id is None:
            err_msg = "Expert '%s' does not exist in the database" % exp_name
            if self.hooks and self.hooks.on_stat_change:
                self.hooks.on_stat_change(err_msg)
            raise ValueError(err_msg)

        self.local_tknizr = LocalTokenizer(MODEL)

    def _get_document_id_by_hash(self, cont_hash: str) -> int | None:
        row = fetch_one(
            conn=self.conn,
            sql="SELECT id FROM documents WHERE content_hash = %s;",
            params=(cont_hash,),
        )
        return int(row[0]) if row else None

    def _link(self, doc_id: int, commit: bool = True) -> bool:
        """
        Link the document to this expert.
        Return False if it was already linked or None if expert does not exist.

        Set 'commit=False' to include it in an open transaction.
        """
        return (
            execute(
                conn=self.conn,
                sql=("""
                INSERT INTO expert_documents (expert_id, document_id)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING;
                """),
                params=(self.exp_id, doc_id),
                commit=commit,
            )
            > 0
        )

    def _link_existing(self, doc_id: int) -> StoreStatus:
        """
        Link an already stored document and report what happend.

        doc_id: Id from an already existing document.
        """
        if self.link_document(doc_id):
            return "linked"
        return "already_linked"

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

    def _embed_every_chunks_in_document(
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
                raise RuntimeError(
                    "Failed to embed chunk %d/%d for expert '%s'"
                    % (i, tol, self.exp_name)
                )
            _, vector, embed_tkns = result

            embed_results.append((vector, embed_tkns or 0))
        return embed_results

    def _add_document(
        self,
        title: str,
        doc_type: str,
        cont_hash: str,
        cont: str,
        path: str | None = None,
        metadata: dict | None = None,
    ) -> int | None:
        """
        Insert the document row. Returns None if the hash already exists (e.g.
        another process stored it first). Does NOT commit: call insisde
        transaction().
        """
        row = fetch_one(
            conn=self.conn,
            sql=("""
                INSERT INTO documents
                    (title, type, content_hash, source_path, content,
                    embedding_model, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (content_hash) DO NOTHING
                RETURNING id;
                """),
            params=(
                title,
                doc_type,
                cont_hash,
                path,
                cont,
                EMBED_MODEL,
                json.dumps(metadata or {}),
            ),
            commit=False,
        )
        return int(row[0]) if row else None

    def _add_all_chunks_from_document(
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
                INSERT INTO document_chunks
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

    def _add_entire_document_to_expert(
        self,
        *,
        title: str,
        doc_type: str,
        cont_hash: str,
        cont: str,
        chunks: list,
        path: str | None = None,
        metadata: dict | None = None,
        hooks: AgentHooks | None = None,
    ) -> tuple[int, StoreStatus]:
        """
        Store a document and link it to this expert. Returns (document_id,
        status).

        New content is embedded first, then the document row, every chunk and
        the expert link are written in ONE transaction, so failure leaves
        nothing behind. Content is already stored is not embedded again: it is
        just linked to this expert.
        """
        chunks = self._formatted_chunks_with_data(chunks)
        if not chunks:
            raise ValueError("Cannot add a document with no chunks.")

        # Check if content has already been stored
        existing = self._get_document_id_by_hash(cont_hash)
        if existing is not None:
            status = self._link_existing(existing)
            info_msg = "Document '%s' already stored (id %d): %s expert '%s'" % (
                title,
                existing,
                status.replace("_", " "),
                self.exp_name,
            )
            app_log.info(info_msg)
            if hooks and hooks.on_stat_change:
                hooks.on_stat_change(info_msg)
            return existing, status

        # Embed before opening the transaction, so it isn't held open during
        # the slow part
        embed_results = self._embed_every_chunks_in_document(chunks, hooks)

        doc_id = None
        with transaction(self.conn):
            doc_id = self._add_document(
                title=title,
                doc_type=doc_type,
                cont_hash=cont_hash,
                cont=cont,
                path=path,
                metadata=metadata,
            )
            if doc_id is not None:
                self._add_all_chunks_from_document(doc_id, chunks, embed_results)
                self._link(doc_id, commit=False)

        # Another process stored the same content embedding
        if doc_id is None:
            existing = self._get_document_id_by_hash(cont_hash)
            if existing is None:
                raise RuntimeError("Document '%s' vanished during insert." % title)
            return existing, self._link_existing(existing)

        info_msg = "Added '%s' (%d chunks) to expert '%s'" % (
            title,
            len(chunks),
            self.exp_name,
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
            results[title] = self._add_entire_document_to_expert(
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

    @staticmethod
    def find_document_id(conn, doc: str | int) -> int:
        """
        Resolve a document title or id to a document id. A title is tried
        first. then a number is treated as an id. Raises ValueError if nothing
        matches, or if several documents share the title (use the id then).
        """
        if isinstance(doc, int):
            row = fetch_one(
                conn=conn,
                sql="SELECT id FROM documents WHERE id = %s;",
                params=(doc,),
            )
            if row is None:
                raise ValueError("No documents with id %d." % doc)
            return int(row[0])

        rows = fetch_all(
            conn=conn,
            sql="SELECT id FROM documents WHERE title = %s ORDER BY id;",
            params=(doc,),
        )
        if len(rows) == 1:
            return int(rows[0][0])

        # More than one titles exists
        if len(rows) > 1:
            raise ValueError(
                "%d documents are named '%s' (ids: %s). Use the id instead"
                % (len(rows), doc, ", ".join(str(r[0]) for r in rows))
            )

        # No matches
        if doc.isdigit():
            return ExpertDocuments.find_document_id(conn, int(doc))
        raise ValueError("No document named '%s'" % doc)

    def all_documents_data_from_self(self) -> dict[int, dict[str, Any]]:
        """Return all documents that are linked to this expert, keyed by id."""
        rows = fetch_all(
            conn=self.conn,
            sql=("""
                SELECT d.id, d.title, d.type, d.description,
                    d.created_at, d.updated_at, d.metadata
                FROM expert_documents ed
                JOIN documents d ON d.id = ed.document_id
                WHERE ed.expert_id = %s
                ORDER BY d.title, d.id;
                """),
            params=(self.exp_id,),
        )

        return {row[0]: _document_entry(row) for row in rows}

    @staticmethod
    def all_documents_data_across_experts(conn) -> dict[int, dict[str, Any]]:
        """
        Return every stored document, including ones no expert uses, keyed by
        document id.
        """
        ensure_schema(conn)
        rows = fetch_all(
            conn=conn,
            sql=("""
                SELECT d.id, d.title, d.type, d.description,
                    d.created_at, d.updated_at, d.metadata,
                    array_agg(e.name ORDER BY e.name)
                        FILTER (WHERE e.name IS NOT NULL)
                FROM documents d
                LEFT JOIN expert_documents ed ON ed.document_id = d.id
                LEFT JOIN experts e ON e.id = ed.expert_id
                GROUP BY d.id
                ORDER BY d.title, d.id;
                """),
        )
        return {
            row[0]: {**_document_entry(row), "experts": row[7] or []} for row in rows
        }

    @staticmethod
    def generate_document_description(
        conn, doc_name: str, hooks: AgentHooks | None = None
    ):
        """
        Have the chat model read the stored document and write a short summary
        of the whole document into documents.description, which helps
        retrieval.

        - Returns the description, or None if the model produced
        nothing.
        - Raises ValueError if the document isn't found.
        """
        ensure_schema(conn)
        doc_id = ExpertDocuments.find_document_id(conn, doc_name)
        row = fetch_one(
            conn=conn,
            sql="SELECT title, content FROM documents WHERE id = %s;",
            params=(doc_id,),
        )
        title, cont = row

        if hooks and hooks.on_loading:
            hooks.on_loading("Generating description for document '%s'..." % title)

        # Keep the document in the model's context
        max_tkns = LocalTokenizer(MODEL).model_max_tkns
        max_chars = (
            max(2000, (max_tkns - EXPERT_RESERVE_TKNS - 500) * 3)
            if max_tkns
            else EXPERT_DOC_FALLBACK_CHARS
        )

        response = model_response_with_new_system_prompt_and_context(
            model=MODEL,
            sys_prompt=DOC_DES_PROMPT,
            prompt="Title: %s\n\n%s" % (title, cont[:max_chars]),
            hooks=hooks,
        )
        if not response:
            return None

        text, _, _, is_stopped = response
        if is_stopped is True or not text.strip():
            return None
        des = text.strip()

        execute(
            conn=conn,
            sql=("""
                UPDATE documents
                SET description = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s;
                """),
            params=(des, doc_id),
        )
        info_msg = "Document '%s' description has been generated" % title
        app_log.info(info_msg)
        if hooks and hooks.on_stat_change:
            hooks.on_stat_change(info_msg)
        return des

    # --- Link / unlink -------------------------------------------------------

    def link_document(
        self,
        doc_id: int,
        embed_model: str = EMBED_MODEL,
        hooks: AgentHooks | None = None,
    ) -> bool:
        """
        Link an ALREADY STORED DOCUMENT to this expert.

        doc_id:      Id from an already existing document.
        embed_model: Current user set embedding model, used for validating
                     if it matches the model that was used during the
                     embedding of the document.

        Returen False if it was already linked or None if expert does not
        exists.

        Warns if the document was embedded with a different model than
        the current one, as search skips those until re-embedded.
        """
        row = fetch_one(
            conn=self.conn,
            sql="SELECT embedding_model FROM documents WHERE id = %s;",
            params=(doc_id,),
        )

        if row is None:
            raise ValueError("Document id %d does not exist." % doc_id)

        if row[0] != embed_model:
            warn_msg = (
                "Document id %d was embedded with '%s' but the current model "
                "is '%s'. It won't appear in searches until re-embedded."
                % (doc_id, row[0], embed_model)
            )
            app_log.warning(warn_msg)
            if hooks and hooks.on_stat_change:
                hooks.on_stat_change(warn_msg)
        return self._link(doc_id)

    def link_multiple_documents(
        self, doc_names: list[str], hooks: AgentHooks | None = None
    ) -> dict[str, str]:
        """
        Link already stored documents (titles or ids) to this expert.
        Returns {name: "linked" | "already_linked" | "failed: <reason>"}.
        """
        results = {}
        for name in doc_names:
            try:
                linked = self.link_document(
                    ExpertDocuments.find_document_id(self.conn, name)
                )
            except ValueError as e:
                # Skip this document if failed
                warn_msg = "Cannot link '%s' to expert '%s': %s" % (
                    name,
                    self.exp_name,
                    e,
                )
                app_log.warning(warn_msg)
                if hooks and hooks.on_stat_change:
                    hooks.on_stat_change(warn_msg)
                results[name] = f"failed: {e}"
                continue

            results[name] = "linked" if linked else "already_linked"
        return results

    def unlink_document(self, doc_id: int) -> bool:
        """
        Unlink a document from this expert.
        The document and its chunks are NOT deleted. Return False if it was't
        linked or None if expert does not exists.

        doc_id: Id from an already existing document.
        """
        return (
            execute(
                conn=self.conn,
                sql=("""
                DELETE FROM expert_documents
                WHERE expert_id = %s AND document_id = %s;
                """),
                params=(self.exp_id, doc_id),
            )
            > 0
        )

    def unlink_multiple_documents(
        self, doc_names: list[str], hooks: AgentHooks | None = None
    ) -> dict[str, str]:
        """
        Unlink documents (titles or ids) from this expert. The documents
        themselves are NOT deleted.
        Returns {name: "unlinked" | "not_linked" | "failed: <reason>"}.
        """
        results = {}
        for name in doc_names:
            try:
                linked = self.unlink_document(
                    ExpertDocuments.find_document_id(self.conn, name)
                )
            except ValueError as e:
                # Skip this document if failed
                warn_str = "Cannot link '%s' to expert '%s': %s" % (
                    name,
                    self.exp_name,
                    e,
                )
                app_log.warning(warn_str)
                if hooks and hooks.on_stat_change:
                    hooks.on_stat_change(warn_str)
                results[name] = f"failed: {e}"
                continue

            results[name] = "unlinked" if linked else "not_linked"
        return results

    def chunk_document_to_database(
        self,
        path: Path,
        is_docling: bool = ENABLE_DOCLING,
        hooks: AgentHooks | None = None,
    ) -> dict[str, tuple[int, StoreStatus]] | None:
        """
        Read a file, chunk it, embed the chunks and store everything in the
        database, linked to this expert.

        Returns {title: (document id, status)}, or None if the file could not
        be read, has no content, or has no readable metadata.
        """
        out = FileReader().read_document(path)
        if not out:
            app_log.warning("Failed to read '%s'. Skipping", path)
            return None
        cont, _ = out

        chunks = DoclingChunker().chunking(cont, hooks=hooks)
        if not chunks:
            app_log.warning("No chunks produced from '%s'. Skipping", path)
            return None

        cont_hash = hash_string(cont)
        doc_dict = _format_to_document_dictionary(path, cont, chunks, cont_hash)
        if not doc_dict:
            app_log.warning("No file metadata for '%s'. Skipping", path)
            return None

        return self._store_documents(doc_dict, hooks=hooks)
