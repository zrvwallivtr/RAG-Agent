from src.config.models import EMBED_MODEL

from src.libs.helpers.database_helpers import fetch_one, execute
from src.libs.custom_dataclasses import AgentHooks

from .get_or_create import get_or_create_expert

from src.rag.expert.schema import ensure_link_schema, ensure_schema
from src import logger

app_log = logger.app_logger(f"{__name__}.app")


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
        self.cur  = self.conn.cursor()
        self.exp_name = exp_name

        ensure_schema(self.conn)
        ensure_link_schema(self.conn)
        self.exp_id = get_or_create_expert(self, des)
