"""
Optional links between experts and chat sessions. Experts work fine without
any of this. sess_id is the UUID string from
ChatSessionLogs.get_or_create_session_id().

Deleting a session or an expert removes only the link rows.
"""

from src.libs.helpers.database_helpers import fetch_all, execute

from .schema import ensure_link_schema

from src import logger

app_log = logger.app_logger(f"{__name__}.app")


def sessions_using_expert(conn, exp_id: int) -> list[tuple[str, str]]:
    """
    Find all the session the specified expert is used in.
    (session_id, session_name of every session this expert is enabled in).
    """
    ensure_link_schema(conn)
    rows = fetch_all(
        conn=conn,
        sql=(
            """
            SELECT s.session_id, s.session_name
            FROM session_experts se
            JOIN chat_sessions s ON s.session_id = se.session_id
            WHERE se.expert_id = %s
            ORDER BY s.session_name;
            """
        ),
        params=(exp_id,),
    )
    return [(str(row[0]), str(row[1])) for row in rows]


class SessionExperts:
    """The experts enabled in a chat session (multiple experts per session)."""

    def __init__(
        self,
        conn,
        sess_id: str,
        exp_id: int | None = None,
    ):
        self._conn = conn
        self.sess_id = sess_id
        self.exp_id = exp_id
        ensure_link_schema(self._conn)

    def enable_expert(self) -> None:
        """Enable an expert in this session."""
        if self.exp_id is None:
            raise ValueError("enable_expert needs an exp_id.")

        execute(
            conn=self._conn,
            sql=(
                """
                INSERT INTO session_experts (session_id, expert_id)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING;
                """
            ),
            params=(self.sess_id, self.exp_id),
        )
        app_log.info("Expert id '%s' enable in session '%s'", self.exp_id, self.sess_id)

    def disable_expert(self) -> bool:
        """
        Disable an expert for a session. Returns false if the expert
        wasn't enabled in this session.
        """
        if self.exp_id is None:
            raise ValueError("enable_expert needs an exp_id.")

        return execute(
            conn=self._conn,
            sql="DELETE FROM session_experts WHERE session_id = %s AND expert_id = %s;",
            params=(self.sess_id, self.exp_id),
        ) > 0

    def enabled_experts(self) -> list[tuple[int, str]]:
        """(id, name) of every expert enable in this session."""
        rows = fetch_all(
            conn=self._conn,
            sql=(
                """
                SELECT e.id, e.name
                FROM session_experts se
                JOIN experts e ON e.id = se.expert_id
                WHERE se.session_id = %s
                ORDER BY e.name;
                """
            ),
            params=(self.sess_id,),
        )
        return [(int(r[0]), str(r[1])) for r in rows]

    def enable_ids(self) -> list[int]:
        return [exp_id for exp_id, _ in self.enabled_experts()]
