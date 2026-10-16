from typing import Sequence

from src.config.config_toml import (
    EMBED_MODEL, EXPLORATION_FACTOR, EXPERT_RETRIEVE_LIMIT
)

from src.libs.helpers.database_helpers import fetch_one, fetch_all
from src.libs.helpers.expert_helpers import to_vector_literal

from src.agent.models.embed import embedding_content
from src import logger

app_log = logger.app_logger(f"{__name__}.app")

# Cosine distance
_SEAR_SQL = (
    """
    SELECT c.id,
        (
            SELECT array_agg(e.name ORDER BY e.name)
            FROM expert_documents ed
            JOIN experts e ON e.id = ed.expert_id
            WHERE ed.document_id = d.id AND ed.expert_id = ANY(%s::bigint[])
        ) AS experts,
        d.title, d.source_path, c.content, c.metadata,
        c.prompt_tokens, c.embeddings <=> %s::vector AS distance
    FROM document_chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE d.embedding_model = %s
        AND c.document_id IN (
            SELECT document_id FROM expert_documents
            WHERE expert_id = ANY(%s::bigint[])
        )
    ORDER BY c.embeddings <=> %s::vector
    LIMIT %s;
    """
)

_DIST = 7  # Index of the distance column in a result row

REPLY_RESERVE_TKNS = 1000


def expert_tkn_budget(
    model_max: int | None,
    used_tkns: int,
    prompt_tkns: int,
    share: float = 0.5,
) -> int | None:
    """
    Token budget for expert chunks: a share of the context window that is 
    still free after the session history, the prompt and the reply reserve.
    Returns None (no limit) if the model's token limit is unknown.
    """
    if not model_max:
        return None
    free = model_max - used_tkns - prompt_tkns - REPLY_RESERVE_TKNS
    return max(0, int(free * share))


class QueryExperts:
    def __init__(self, conn, ef_search: int = EXPLORATION_FACTOR):
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

    def query_from_prompt(
        self,
        prompt: str,
        exp_ids: Sequence[int],
        retrieve_lim: int = EXPERT_RETRIEVE_LIMIT,
        per_exp_lim: int | None = None,
        tkn_budget: int | None = None,
    ) -> list[dict]:
        """
        Embed the prompt and return the closest chunks across the given experts,
        best first. Return [] if there are no experts or the prompt can't be 
        embedded.

        per_exp_k:  Cap on how many chunks a single expert may contribute, so a
                    large expert can't crowd out a small one.
                    USE THIS ONLY IF MORE THAN ONE EXPERT IS ENABLED.

        tkn_budget: Stop adding chunks once their token_count total would 
                    exceed this. A chunk that doesn't fit is skipped, smaller
                    ones may still fit. None means no limit. Use
                    expert_tkn_budget().
        """
        if not exp_ids:
            return []

        qv = self._embed_prompt_for_expert(prompt)
        if qv is None:
            return []

        self._set_ef_search()

        if per_exp_lim:
            rows = []
            for exp_id in exp_ids:
                rows.extend(self._query(qv, [exp_id], per_exp_lim))
            rows = self._merge_duplicates(rows)
            rows.sort(key=lambda r: r[_DIST])
            rows = rows[:retrieve_lim]
        else:
            rows = self._query(qv, list(exp_ids), retrieve_lim)

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
            params=(exp_ids, qv, EMBED_MODEL, exp_ids, qv, limit),
        )

    @staticmethod
    def _merge_duplicates(rows: list) -> list:
        """
        A document linked to several experts is found once per expert when
        each is searched separately. Keep one row per chunk and combine the
        epxert name.
        """
        merged: dict = {}
        for row in rows:
            chunk_id = row[0]
            if chunk_id in merged:
                prev = merged[chunk_id]
                names = sorted(set(prev[1]) | set(row[1]))
                merged[chunk_id] = (prev[0], names, *prev[2:])
            else:
                merged[chunk_id] = row
        return list(merged.values())

    @staticmethod
    def _to_hits(rows: list, tkn_budget: int | None) -> list[dict]:
        hits, used = [], 0
        for _, exp_names, title, source, content, meta, tkns, dist in rows:
            if tkn_budget is not None and used + tkns > tkn_budget:
                continue
            used += tkns
            hits.append({
                "expert": ", ".join(exp_names),
                "experts": list(exp_names),
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
