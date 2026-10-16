import json
from typing import Literal

from src.config.config_toml import MODEL, EMBED_MODEL

from src.libs.helpers.database_helpers import (
    fetch_one, execute, execute_many, transaction
)
from src.libs.helpers.expert_helpers import to_vector_literal
from src.libs.custom_dataclasses import AgentHooks

from src.agent.models.embed import embedding_content
from src.agent.tokenizers import LocalTokenizer
from src import logger

app_log = logger.app_logger(f"{__name__}.app")

StoreStatus = Literal[
    "added",  # New document, embedded and linked to the expert

    "linked",  # Document already existed (embedded once before),
               # now linked to this expert

    "already_linked",  # Document already existed and was already
                       # linked to this expert
]


class StoreDocuments:
    """
    Stores documents and links them to ONE expert.

    A document is stored ONCE (identified by the hash of its parsed content)
    and can be linked to many experts, so adding the same content to a second
    expert reuses the existing chunks and embeddings instead of embedding it
    again.
    """

    def __init__(self, conn, exp_name: str, exp_id: int):
        self._conn = conn

        self.exp_name = exp_name
        self.exp_id = exp_id

        self.local_tknizr = LocalTokenizer(MODEL)

    # --- Checkers ------------------------------------------------------------

    def get_document_id(self, cont_hash: str) -> int | None:
        """Get id of the stored document from specified content hash."""
        row = fetch_one(
            conn=self._conn,
            sql="SELECT id FROM documents WHERE content_hash = %s;",
            params=(cont_hash,),
        )
        return int(row[0]) if row else None

    def is_linked(self, doc_id: int) -> bool:
        """Check if the current expert is linked to the document."""
        row = fetch_one(
            conn=self._conn,
            sql=(
                """
                SELECT 1 FROM expert_documents
                WHERE expert_id = %s AND document_id = %s;
                """
            ),
            params=(self.exp_id, doc_id)
        )
        return row is not None

    # --- Link / unlink -------------------------------------------------------

    def _link(self, doc_id: int, commit: bool = True) -> bool:
        """Link the document to this expert. False if it was already linked."""
        return execute(
            conn=self._conn,
            sql=(
                """
                INSERT INTO expert_documents (expert_id, document_id)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING;
                """
            ),
            params=(self.exp_id, doc_id),
            commit=commit,
        ) > 0

    def link_document(self, doc_id: int) -> bool:
        """
        Link an already stored document to this expert. False if it was already
        linked. Warns if the document was embedded with a different model than
        the current one, as search skips those until re-embedded.
        """
        row = fetch_one(
            conn=self._conn,
            sql="SELECT embedding_model FROM documents WHERE id = %s;",
            params=(doc_id,),
        )
        if row is None:
            raise ValueError("Document id %d does not exist." % doc_id)
        if row[0] != EMBED_MODEL:
            app_log.warning(
                "Document id %d was embedded with '%s' but the current model "
                "is '%s'. It won't appear in searches until re-embedded.",
                doc_id, row[0], EMBED_MODEL,
            )
        return self._link(doc_id)

    def unlink_document(self, doc_id: int) -> bool:
        """
        Unlink a document from this expert. The document and its chunks are
        NOT deleted. False if it was't linked.
        """
        return execute(
            conn=self._conn,
            sql=(
                """
                DELETE FROM expert_documents
                WHERE expert_id = %s AND document_id = %s;
                """
            ),
            params=(self.exp_id, doc_id),
        ) > 0

    # --- Chunks --------------------------------------------------------------

    @staticmethod
    def _normalise_chunks(chunks: list) -> list[dict]:
        """
        Accept plain strings or {"text": str, "metadata": dict} dictionaries.
        Always returns [{"text": str, "metadata": dict}], dropping empty chunks.
        """
        normalised = []
        for chunk in chunks:
            if isinstance(chunk, str):
                chunk = {"text": chunk}
            text = chunk["text"]
            if text.strip():
                normalised.append(
                    {"text": text, "metadata": chunk.get("metadata") or {}}
                )
        return normalised

    # --- Embedding -----------------------------------------------------------

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

    # --- Document ------------------------------------------------------------

    def _add_document(
        self,
        title: str,
        doc_type: str,
        cont_hash: str,
        cont: str,
        path: str | None = None,
        metadata: dict | None = None,
    ) -> int | None:
        """Insert the document row. Does NOT commit: call inside transaction()."""
        row = fetch_one(
            conn=self._conn,
            sql=(
                """
                INSERT INTO documents
                    (title, type, content_hash, source_path, content,
                    embedding_model, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (content_hash) DO NOTHING
                RETURNING id;
                """
            ),
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
        """Insert every chunk. Does NOT commit: call inside transaction()."""
        execute_many(
            conn=self._conn,
            sql=(
                """
                INSERT INTO expert_chunks
                    (document_id, chunk_index, embeddings, embed_tokens,
                    content, prompt_tokens, metadata)
                VALUES (%s, %s, %s::vector, %s, %s, %s, %s);
                """
            ),
            params_seq=[
                (
                    doc_id,
                    i,
                    to_vector_literal(vec),
                    embed_tkns,
                    chunk["text"],
                    self.local_tknizr.count_string_tokens(chunk["text"]) or 0,
                    json.dumps(chunk["metadata"]),
                )
                for i, (chunk, (vec, embed_tkns)) in enumerate(zip(chunks, embed_results))
            ],
            commit=False,
        )

    def _link_existing(self, doc_id: int) -> StoreStatus:
        return "linked" if self.link_document(doc_id) else "already_linked"

    def add_entire_document_to_expert(
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
        Store a document and all its chunks in ONE transaction.

        The shared helpers commit per call, so this uses its own cursor to keep the
        document row and its chunks all-or-nothing.
        """
        chunks = self._normalise_chunks(chunks)
        if not chunks:
            raise ValueError("Cannot add a document with no chunks.")

        existing = self.get_document_id(cont_hash)
        if existing is not None:
            status = self._link_existing(existing)
            app_log.info(
                "Document '%s' already stored (id %d): %s expert '%s'",
                title, existing, status.replace("_", " "), self.exp_name,
            )
            return existing, status

        embed_results = self._embed_every_chunks_in_document(chunks, hooks)

        doc_id = None
        with transaction(self._conn):
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

        if doc_id is None:
            existing = self.get_document_id(cont_hash)
            if existing is None:
                raise RuntimeError("Document '%s' vanished during insert." % title)
            return existing, self._link_existing(existing)

        app_log.info(
            "Added '%s' (%d chunks) to expert '%s'",
            title,
            len(chunks),
            self.exp_name
        )
        return doc_id, "added"
