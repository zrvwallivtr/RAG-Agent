import json

from src.config.config_toml import MODEL

from src.libs.helpers.database_helpers import fetch_one, execute_many, transaction
from src.libs.helpers.expert_helpers import to_vector_literal
from src.libs.custom_dataclasses import AgentHooks

from src.agent.models.embed import embedding_content
from src.agent.tokenizers import LocalTokenizer
from src import logger

app_log = logger.app_logger(f"{__name__}.app")


class StoreDocuments:
    def __init__(self, conn, exp_name: str, exp_id: int):
        self._conn = conn

        self.exp_name = exp_name
        self.exp_id = exp_id

        self.local_tknizr = LocalTokenizer(MODEL)

    # --- Checker -------------------------------------------------------------

    def has_document(self, cont_hash: str) -> bool:
        """
        Check if document has been embedded into the database by checking
        if hash is duplicated.
        """
        row = fetch_one(
            conn=self._conn,
            sql=(
                """
                SELECT 1 FROM expert_documents
                WHERE expert_id = %s AND cont_hash = %s;
                """
            ),
            params=(self.exp_id, cont_hash),
        )
        return row is not None

    # --- Embedding -----------------------------------------------------------

    def _embed_every_chunks_in_document(
        self, chunks: list[str], hooks: AgentHooks | None = None
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

            result = embedding_content(chunk)
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
        path: str | None = None,
        metadata: dict | None = None,
        hooks: AgentHooks | None = None,
    ) -> int | None:
        """Insert the document row. Does NOT commit: call inside transaction()."""
        row = fetch_one(
            conn=self._conn,
            sql=(
                """
                INSERT INTO expert_documents
                    (expert_id, source_path, title, type, cont_hash, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (expert_id, cont_hash) DO NOTHING
                RETURNING id;
                """
            ),
            params=(
                self.exp_id,
                path,
                title,
                doc_type,
                cont_hash,
                json.dumps(metadata or {}),
            ),
            commit=False,
        )
        return int(row[0]) if row else None

    def _add_all_chunks_from_document(
        self,
        doc_id: int,
        chunks: list[str],
        embed_results: list[tuple[list[float], int]],
    ) -> None:
        """Insert every chunk. Does NOT commit: call inside transaction()."""
        execute_many(
            conn=self._conn,
            sql=(
                """
                INSERT INTO expert_chunks
                    (expert_id, document_id, chunk_index, embeddings,
                    token_count, content)
                VALUES (%s, %s, %s, %s::vector, %s, %s, %s);
                """
            ),
            params_seq=[
                (
                    self.exp_id,
                    doc_id,
                    i,
                    to_vector_literal(vec),
                    embed_tkns,
                    chunk,
                    self.local_tknizr.count_string_tokens(chunk) or 0,
                )
                for i, (chunk, (vec, embed_tkns)) in enumerate(zip(chunks, embed_results))
            ],
            commit=False,
        )

    def add_entire_document_to_expert(
        self,
        *,
        title: str,
        doc_type: str,
        cont_hash: str,
        chunks: list[str],
        path: str | None = None,
        metadata: dict | None = None,
        hooks: AgentHooks | None = None,
    ) -> int | None:
        """
        Store a document and all its chunks in ONE transaction. Returns the document
        id, or None if this exact content is already in the expert.

        The shared helpers commit per call, so this uses its own cursor to keep the
        document row and its chunks all-or-nothing.
        """
        if not chunks:
            raise ValueError("Cannot add a document with no chunks.")

        if self.has_document(cont_hash):
            app_log.info("Skipped '%s': already in expert '%s'", title, self.exp_name)
            return None

        embed_results = self._embed_every_chunks_in_document(chunks, hooks)

        with transaction(self._conn):
            doc_id = self._add_document(
                title=title,
                doc_type=doc_type,
                cont_hash=cont_hash,
                path=path,
                metadata=metadata,
                hooks=hooks,
            )
            if doc_id is None:
                return None

            self._add_all_chunks_from_document(doc_id, chunks, embed_results)

        app_log.info(
            "Added '%s' (%d chunks) to expert '%s'", title, len(chunks), self.exp_name
        )
        return doc_id
