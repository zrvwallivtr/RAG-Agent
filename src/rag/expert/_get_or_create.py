"""
Functions for looking up or create expert, only call during
__init__().
"""

from src.libs.helpers.database_helpers import fetch_one
from src.libs.custom_dataclasses import AgentHooks

from src import logger

app_log = logger.app_logger(f"{__name__}.app")


def get_expert_id_from_name(conn, exp_name: str) -> int | None:
    """
    Get expert id by name.

    Parameters
    ----------
        conn:     Connection to the database.
        exp_name: Name of the expert.

    Returns
    -------
        - The expert id that matches the name.
    """
    row = fetch_one(
        conn=conn,
        sql="SELECT id FROM experts WHERE name = %s;",
        params=(exp_name,),
    )
    if row:
        return int(row[0])


def get_or_create_expert(
    conn, exp_name: str, des: str | None, hooks: AgentHooks | None = None
) -> int:
    """
    Get the expert's id, creating the expert if it doesn't exist yet.

    Parameters
    ----------
        conn:   Connection to the database.
        expert: Name of the expert.
        des:    Only relevant if expert has not been created yet.
        hooks:  Function hooks for cli/tui integration.
    """
    exp_id = get_expert_id_from_name(conn, exp_name)
    if exp_id is not None:
        return exp_id

    # Create new expert
    row = fetch_one(
        conn=conn,
        sql=("""
            INSERT INTO experts (name, description)
            VALUES (%s, %s)
            RETURNING id;
            """),
        params=(exp_name, des or ""),
    )
    if row is None:
        raise RuntimeError("Failed to create expert '%s': no id returned" % exp_name)

    info_msg = "New expert '%s' created" % exp_name
    app_log.info(info_msg)
    if hooks and hooks.on_stat_change:
        hooks.on_stat_change(info_msg)
    return int(row[0])
