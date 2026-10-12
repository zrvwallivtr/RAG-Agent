from src.config.models import EMBED_MODEL

from src.libs.helpers.database_helpers import fetch_one, execute

from src import logger

app_log = logger.app_logger(f"{__name__}.app")


def get_or_create_expert(expert, des: str | None) -> int:
    """Create an expert. Only calls at the initiation of the Expert class."""
    row = fetch_one(
        conn=expert.conn,
        sql="SELECT id, embedding_model FROM experts WHERE name = %s;",
        params=(expert.exp_name,),
    )
    if row:
        if row[1] != EMBED_MODEL:
            raise ValueError(
                "Expert '%s' was built with '%s' but the "
                "current model is '%s'. Re-embed it first."
                % (expert.exp_name, row[1], EMBED_MODEL)
            )
        return int(row[0])

    row = fetch_one(
        conn=expert.conn,
        sql=(
            """
            INSERT INTOP experts (name, description, embedding_model)
            VALUES (%s, %s, %s)
            RETURNING id;
            """
        ),
        params=(expert.exp_name, des or "", EMBED_MODEL)
    )
    if row is None:
        raise RuntimeError(
            "Failed to create expert '%s': no id returned"
            % expert.exp_name
        )

    app_log.info("New expert '%s' created", expert.exp_name)
    return int(row[0])
