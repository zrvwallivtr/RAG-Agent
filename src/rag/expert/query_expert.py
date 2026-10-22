"""Vector search over the documents linked to experts."""

from src.config.config_toml import (
    EMBED_MODEL,
    EXPLORATION_FACTOR,
    EXPERT_RETRIEVE_LIMIT,
    EXP_MIN_SIM_SCORE,
    EXPERT_RESERVE_TKNS,
)

from src.libs.helpers.database_helpers import fetch_one, fetch_all

from src.agent.models.embed import embedding_content
from src import logger

app_log = logger.app_logger(f"{__name__}.app")

# Cosine distance, nearest first.
#
# Placeholders, in order: expert ids, query vector, embedding model,
# expert ids, query vector, row limit.
#
# Columns: chunk id, names of the given experts that use the chunk's document,
# title, source path, content, metadata, prompt tokens, distance.
#
# Documents embedded with a different model are skipped, because vectors from
# different models can't be compared.
_SEAR_SQL = """
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


def expert_tkn_budget(
    model_max: int | None,
    used_tkns: int,
    prompt_tkns: int,
    reply_reserve_tkns: int = EXPERT_RESERVE_TKNS,
    share: float = 0.5,
) -> int | None:
    """
    Token budget for expert chunks: a share of the context window that is
    still free after the session history, the prompt and the reply reserve.
    Returns None (no limit) if the model's token limit is unknown.
    """
    if not model_max:
        return None
    free = model_max - used_tkns - prompt_tkns - EXPERT_RESERVE_TKNS
    return max(0, int(free * share))


class QueryExpert:
    """Searches the chunks of one or more experts for a prompt."""

    def __init__(self, conn, ef_search: int = EXPLORATION_FACTOR):
        self._conn = conn
        self._ef_search = ef_search

    def _set_ef_search(self) -> None:
        """Apply the HNSW search setting to this connection's session."""
        fetch_one(
            conn=self._conn,
            sql="SELECT set_config('hnsw.ef_search', %s, false);",
            params=(str(self._ef_search),),
        )

    def _query(
        self,
        qv: list[float],
        exp_ids: list[int],
        retrieval_lim: int = EXPERT_RETRIEVE_LIMIT,
    ) -> list:
        """Run search query. Rows are sorted nearest first."""
        return fetch_all(
            conn=self._conn,
            sql=_SEAR_SQL,
            params=(exp_ids, qv, EMBED_MODEL, exp_ids, qv, retrieval_lim),
        )

    @staticmethod
    def _format_query_result(
        rows: list,
        tkn_budget: int | None,
        min_sim_score: float = EXP_MIN_SIM_SCORE,
    ) -> list[dict]:
        """
        Turn result rows dictionaries.

        The rows are sorted nearest first. Only return results that scores
        higher than the minimum similarity score.
        """
        hits, used = [], 0
        for _, exp_names, title, source, content, meta, tkns, dist in rows:
            score = 1 - float(dist)  # Cosine similarity, higher is better

            # Must score higher than the set minimum
            if score < min_sim_score:
                break

            # Token budget check
            if tkn_budget is not None and used + tkns > tkn_budget:
                continue
            used += tkns

            hits.append(
                {
                    "expert": ", ".join(exp_names),
                    "experts": list(exp_names),
                    "title": title,
                    "source": source,
                    "content": content,
                    "metadata": meta or {},
                    "tokens": tkns,
                    "distance": float(dist),
                }
            )

        app_log.info("Expert search returned %d chunk(s)", len(hits))
        return hits

    def query_from_prompt(
        self,
        prompt: str,
        exp_ids: list[int],
        tkn_budget: int | None = None,
    ) -> list[dict]:
        """
        Embed the prompt and return the closest chunks across the given
        experts.

        Returns [] if the prompt can't be embedded.

        exp_ids:      List of experts, allows more than one expert search.

        retrieve_lim: Maximum number of chunks to fetch.

        tkn_budget:   Stop adding chunks once their token_count total would
                      exceed this. A chunk that doesn't fit is skipped, smaller
                      ones may still fit. None means no limit. Use
                      expert_tkn_budget().
        """
        result = embedding_content(prompt)
        if not result:
            app_log.warning("Query expert skipped: Failed to embed the prompt")
            return []
        _, qry_vec, _ = result

        self._set_ef_search()

        rows = self._query(qry_vec, exp_ids)

        hits = self._format_query_result(rows, tkn_budget)
        return hits
