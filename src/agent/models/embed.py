from src.config.config_toml import MODEL, EMBED_MODEL

from src.libs.custom_dataclasses import AgentHooks

from src.agent.models.ollama import ollama_clt
from src import logger

app_log = logger.app_logger(f"{__name__}.app")


def embedding_content(
    cont: str,
    model: str = MODEL,
    emb_model: str = EMBED_MODEL,
    hooks: AgentHooks | None = None,
) -> tuple[str, list[float], int] | None:
    """
    Generate embedding from given texts.

    Parameters
    ----------
        cont:      String to embed.
        model:     Language model name.
        emb_model: Embedding model name.
        hooks:     Function hooks for cli/tui integration.

    Returns
    -------
        - The original string.
        - Embeddings of the string.
        - Token used for embedding.
    """
    try:
        response = ollama_clt.embed(model=emb_model, input=cont)
        embdings = response["embeddings"][0]

        if not embdings:
            warn_msg = "Model '%s' failed to generate vector embeddings" % model
            app_log.warning(warn_msg)
            if hooks and hooks.on_stat_change:
                hooks.on_stat_change(warn_msg)
            return None

        cont_tkns = response.get("prompt_eval_count", 0)
        return cont, embdings, cont_tkns

    except Exception as e:
        err_msg = "Model '%s' failed to generate vector embeddings: %s" % (model, e)
        app_log.warning(err_msg)
        if hooks and hooks.on_stat_change:
            hooks.on_stat_change(err_msg)
        return None
