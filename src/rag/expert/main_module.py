from typing import Sequence, Any
from pathlib import Path

from src.config.config_toml import (
    MODEL,
    EMBED_MODEL,
    PER_EXPERT_RETRIEVAL_LIMIT,
    ENABLE_DOCLING,
)

from src.libs.helpers.database_helpers import fetch_one, fetch_all, execute
from src.libs.custom_dataclasses import AgentHooks
from src.libs.helpers.string_helpers import hash_string
from src.libs.helpers.file_metadata_helpers import unpack_file_metadata

from .get_or_create import get_or_create_expert
from .query_experts import QueryExperts, REPLY_RESERVE_TKNS
from .store_documents import StoreDocuments, StoreStatus

from src.rag.file_parser import FileParser
from src.rag.docling import DoclingChunker
from src.agent.tokenizers import LocalTokenizer
from src.rag.expert.schema import ensure_schema
from src.agent.models.llm import model_response_with_new_system_prompt_and_context
from src import logger

app_log = logger.app_logger(f"{__name__}.app")

DESCRIBE_PROMPT = (
    "Write short descriptions of documents so a retrieval system can tell "
    "what each document covers. Read the document and reply with ONE paragraph "
    "(at most 120 words) describing its subject, its scope and the kinds of "
    "questions it can answer. Do not add headings, lists or commentary."
)

# Characters of a documents sent to the model when its token limits is unknown
_DESCRIBE_FALLBACK_CHARS = 12000


def query_multiple_experts(
    conn,
    prompt: str,
    exp_ids: Sequence[int],
    per_exp_lim = PER_EXPERT_RETRIEVAL_LIMIT,
    tkn_budget: int | None = None,
) -> list[dict]:
    return QueryExperts(conn).query_from_prompt(
        prompt=prompt,
        exp_ids=exp_ids,
        per_exp_lim=per_exp_lim,
        tkn_budget=tkn_budget,
    )


def _format_to_document_dictionary(
    path: Path, cont: str, chunks: list, cont_hash: str
) -> dict[str, dict[Any, Any]] | None:
    """
    Format into the expected dictionary structure for _store_documents():

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
            }
        }
    }


def _document_entry(row) -> dict[str, Any]:
    """Row columns: id, title, type, description, created_at, updated_at, metadata."""
    return {
        "title": row[1],
        "type": row[2],
        "description": row[3],
        "created_at": row[4],
        "updated_at": row[5],
        "metadata": row[6],
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
        self.file_parser = FileParser()

        # Chunk size is measured with the embedding model's tokenizer
        self.chunker = DoclingChunker(LocalTokenizer(EMBED_MODEL))

        ensure_schema(self.conn)
        self.exp_id = get_or_create_expert(self, des)

        self.qry_exp = QueryExperts(conn=self.conn)
        self.store_doc = StoreDocuments(
            conn=self.conn, exp_name=self.exp_name, exp_id=self.exp_id
        )

    # --- Experts -------------------------------------------------------------

    def delete_expert(self) -> bool:
        """
        Delete the entry that matches self.exp_id from the experts
        table in the database.

        The foreign keys remove only this expert's link rows (expert_documents
        and session_experts). Documents and chunks are not touched.

        NO DOCUMENTS ARE DELETED HERE.
        """
        return execute(
            conn=self.conn,
            sql=(
                """
                DELETE FROM experts
                WHERE id = %s;
                """
            ),
            params=(self.exp_id,)
        ) > 0

    def all_experts_data(self) -> dict[str, Any] | None:
        """Return all created experts from the database table."""
        rows = fetch_all(
            conn=self.conn,
            sql=(
                """
                SELECT e.name, e.description, e.created_at, e.updated_at 
                    COUNT(ed.document_id)
                FROM experts e
                LEFT JOIN experts_documents ed ON ed.expert_id = e.id
                GROUP BY e.id
                ORDER BY e.name;
                """
            ),
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

    # --- Documents -----------------------------------------------------------

    def all_documents_data_from_self(self) -> dict[int, dict[str, Any]]:
        """Return all documents that are linked to this expert."""
        rows = fetch_all(
            conn=self.conn,
            sql=(
                """
                SELECT d.id, d.title, d.type, d.description,
                    d.created_at, d.updated_at, d.metadata
                FROM expert_documents ed
                JOIN documents d ON d.id = ed.document_id
                WHERE ed.expert_id = %s
                ORDER BY d.title, d.id;
                """
            ),
            params=(self.exp_id,),
        )

        return {row[0]: _document_entry(row) for row in rows}

    def all_documents_data_across_experts(self) -> dict[int, dict[str, Any]]:
        """
        Return every stored document, including ones no expert uses, keyed by
        document id.
        """
        rows = fetch_all(
            conn=self.conn,
            sql=(
                """
                SELECT d.id, d.title, d.type, d.description,
                    d.created_at, d.updated_at, d.metadata,
                    array_agg(e.name ORDER BY e.name)
                        FILTER (WHERE e.name IS NOT NULL)
                FROM documents d
                LEFT JOIN expert_documents ed ON ed.document_id = d.id
                LEFT JOIN experts e ON e.id = ed.expert_id
                GROUP BY d.id
                ORDER BY d.title, d.id;
                """
            ),
        )
        return {
            row[0]: {**_document_entry(row), "experts": row[7] or []}
            for row in rows
        }

    def _find_document_id(self, doc: str | int) -> int:
        """
        Resolve a document title or id to a document id. A title is tried
        first. then a number is treated as an id. Raises ValueError if nothing
        matches, or if several documents share the title (use the id then).
        """
        if isinstance(doc, int):
            row = fetch_one(
                conn=self.conn,
                sql="SELECT id FROM documents WHERE id = %s;",
                params=(doc,),
            )
            if row is None:
                raise ValueError("No documents with id %d." % doc)
            return int(row[0])

        rows = fetch_all(
            conn=self.conn,
            sql="SELECT id FROM documents WHERE title = %s ORDER BY id;",
            params=(doc,),
        )
        if len(rows) == 1:
            raise ValueError(
                "%d documents are named '%s' (ids: %s). Use the id instead."
                % (len(rows), doc, ", ".join(str(r[0]) for r in rows))
            )
        if doc.isdigit():
            return self._find_document_id(int(doc))
        raise ValueError("No document named '%s'." % doc)

    def link_document(self, doc_name: str) -> bool:
        """
        Link an already stored document (title or id) to this expert.
        False if it was already linked.
        """
        return self.store_doc.link_document(self._find_document_id(doc_name))

    def unlink_document(self, doc_name: str) -> bool:
        """
        Unlink a document (title or id) from the expert. The document itself
        is NOT deleted. False if it wasn't linked.
        """
        return self.store_doc.unlink_document(self._find_document_id(doc_name))

    def generate_document_descriptoin(
        self,
        doc_name: str,
        hooks: AgentHooks | None = None
    ):
        """
        Read the stored document and write a short summary of the whole document
        into documents.description. Returns the description.
        """
        doc_id = self._find_document_id(doc_name)
        row = fetch_one(
            conn=self.conn,
            sql="SELECT title, content FROM documents WHERE id = %s;",
            params=(doc_id,),
        )
        title, cont = row

        if hooks and hooks.on_loading:
            hooks.on_loading("Generating description for document '%s'..." % title)

        max_tkns = LocalTokenizer(MODEL).model_max_tkns
        max_chars = (
            max(2000, (max_tkns - REPLY_RESERVE_TKNS - 500) * 3)
            if max_tkns else _DESCRIBE_FALLBACK_CHARS
        )

        response = model_response_with_new_system_prompt_and_context(
            model=MODEL,
            sys_prompt=DESCRIBE_PROMPT,
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
            conn=self.conn,
            sql=(
                """
                UPDATE documents
                SET description = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s;
                """
            ),
            params=(des, doc_id),
        )
        app_log.info("Document '%s' description generated", title)
        return des

    # --- Query ---------------------------------------------------------------

    def query_expert(
        self,
        prompt: str,
        exp_ids: Sequence[int],
        tkn_budget: int | None = None,
    ) -> list[dict]:
        if len(exp_ids) > 1:
            per_exp_lim = PER_EXPERT_RETRIEVAL_LIMIT
        else:
            per_exp_lim = None

        return self.qry_exp.query_from_prompt(
            prompt=prompt,
            exp_ids=exp_ids,
            per_exp_lim=per_exp_lim,
            tkn_budget=tkn_budget
        )

    # --- Store ----------------------------------------------------------------

    def _store_documents(
        self,
        doc_dict: dict[str, dict[Any, Any]],
        hooks: AgentHooks | None = None,
    ) -> dict[str, tuple[int, StoreStatus]]:
        """
        Store every document in the dictionary (see
        _format_to_document_dictionary for its structure) and link it to this
        expert.
        """
        results = {}

        for title, data in doc_dict.items():
            results[title] = self.store_doc.add_entire_document_to_expert(
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

    def chunk_document_to_database(
        self,
        path: Path,
        is_docling: bool = ENABLE_DOCLING,
        hooks: AgentHooks | None = None
    ) -> dict[str, tuple[int, StoreStatus]] | None:
        """
        Parse the document for the docling chunker, using the docling
        chunker to chunk the document and store to the database.
        """
        out = self.file_parser.read_document(path)
        if not out:
            return None
        cont, _ = out

        chunks = self.chunker.chunking(cont)
        if not chunks:
            app_log.warning("No chunks produced from '%s'. Skipping", path)
            return None

        cont_hash = hash_string(cont)
        doc_dict = _format_to_document_dictionary(path, cont, chunks, cont_hash)
        if not doc_dict:
            return None

        return self._store_documents(doc_dict, hooks=hooks)
