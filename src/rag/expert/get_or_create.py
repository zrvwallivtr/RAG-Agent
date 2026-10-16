from src.config.config_toml import EMBED_MODEL

from src.libs.helpers.database_helpers import fetch_one, execute

from src import logger

app_log = logger.app_logger(f"{__name__}.app")


def get_or_create_expert(expert, des: str | None) -> int:
    """Return the expert's id, creating the expert if it doesn't exist yet."""
    row = fetch_one(
        conn=expert.conn,
        sql="SELECT id FROM experts WHERE name = %s;",
        params=(expert.exp_name,),
    )
    if row:
        return int(row[0])

    row = fetch_one(
        conn=expert.conn,
        sql=(
            """
            INSERT INTOP experts (name, description)
            VALUES (%s, %s)
            RETURNING id;
            """
        ),
        params=(expert.exp_name, des or "")
    )
    if row is None:
        raise RuntimeError(
            "Failed to create expert '%s': no id returned"
            % expert.exp_name
        )

    app_log.info("New expert '%s' created", expert.exp_name)
    return int(row[0])
