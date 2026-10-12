from typing import Sequence

from src.libs.helpers.database_helpers import fetch_one, fetch_all
from src.libs.helpers.expert_helpers import to_vector_literal

from src.agent.models.embed import embedding_content
from src import logger

app_log = logger.app_logger(f"{__name__}.app")

# Cosine distance
_SEAR_SQL = (
    """
    SELECT e.name, d.title, d.source_path, c.content, c.metadata,
        c.prompt_tokens, c.embedings <=> %s::vector AS distance
    FROM expert_chunks c
    JOIN experts e          ON e.id = c.expert_id
    JOIN expert_documents d ON d.id = c.document_id
    WHERE c.expert_id = ANY(%s::bigint[])
    ORDER BY c.embeddings <=> %s::vector
    LIMIT %s;
    """
)


class QueryExperts:
    def __init__(self, conn, ef_search: int = 100):
        self._conn = conn
        self._ef_search = ef_search

    def _embed_prompt_for_expert(self, prompt: str) -> str | None:
        """
        Custom embedding process to return a pgvector text literal from prompt
        which will be used in SQL as '%s::vector'.
        """
        result = embedding_content(prompt)
        if not result:
            app_log.warning("Query expert skipped: Failed to embed the prompt")
            return None
        _, qry_vec, _ = result

        return to_vector_literal(qry_vec)

    def search_from_prompt(
        self,
        prompt: str,
        exp_ids: Sequence[int],
        k: int = 8,
        per_exp_k: int | None = None,
        tkn_budget: int | None = None,
    ) -> list[dict] | None:
        """
        Embed the prompt and return the closest chunks across the given experts,
        best first. Return [] if there are no experts or the prompt can't be embedded.

        per_exp_k:  Cap on how many chunks any single expert may contribute, so a
                    large expert can't crowd out a small one. Runs one indexed query
                    per expert, then merges best-first and cuts to k.
        tkn_budget: Stop adding chunks once their token_count total would exceed this.
                    A chunk that doesn't fit is skipped, smaller ones may still fit.
        """
        if not exp_ids:
            return []

        qv = self._embed_prompt_for_expert(prompt)
        if qv is None:
            return []

        self._set_ef_search()

        if per_exp_k:
            rows = []
            for exp_id in exp_ids:
                rows.extend(self._query(qv, [exp_id], per_exp_k))
            rows.sort(key=lambda r: r[6])
            rows = rows[:k]
        else:
            rows = self._query(qv, list(exp_ids), k)

        hits = self._to_hits(rows, tkn_budget)
        app_log.debug(
            "Expert search returned %d chunk(s) from %d expert(s)",
            len(hits), len(exp_ids)
        )
        return hits

    def _set_ef_search(self) -> None:
        fetch_one(
            conn=self._conn,
            sql="SELECT set_config('hnsw.ef_search', %s, false);",
            params=(str(self._ef_search),),
        )

    def _query(self, qv: str, exp_ids: list[int], limit: int) -> list:
        return fetch_all(
            conn=self._conn,
            sql=_SEAR_SQL,
            params=(qv, exp_ids, qv, limit),
        )

    @staticmethod
    def _to_hits(rows: list, tkn_budget: int | None) -> list[dict]:
        hits, used = [], 0
        for exp_name, title, source, content, meta, tkns, dist in rows:
            if tkn_budget is not None and used + tkns > tkn_budget:
                continue
            used += tkns
            hits.append({
                "expert": exp_name,
                "title": title,
                "source": source,
                "content": content,
                "metadata": meta or {},
                "tokens": tkns,
                "distance": float(dist),
            })
        return hits

    @staticmethod
    def format_context(hits: list[dict]) -> str:
        """Label each excerpt with its expert and source so the model can cross-reference."""
        blocks = []
        for hit in hits:
            label = f"{hit['expert']} | {hit['title'] or hit['source'] or 'untitled'}"
            page = hit["metadata"].get("page")
            if page:
                label += f" p.{page}"
            blocks.append(f"[{label}]\n{hit['content']}")
        return "\n\n".join(blocks)
