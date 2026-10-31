"""
Optional links between experts and chat sessions. Experts work fine without
any of this. sess_id is the UUID string from
ChatSessionLogs.get_or_create_session_id().

Deleting a session or an expert removes only the link rows.
"""

from typing import Any

from src.libs.helpers.database_helpers import fetch_one, fetch_all, execute

from ._schema import ensure_schema, ensure_link_schema
from ._get_or_create import get_expert_id_from_name

from src.libs.custom_dataclasses import AgentHooks

from src import logger

app_log = logger.app_logger(f"{__name__}.app")


def _find_session_id_from_name(
    conn, sess_name: str, hooks: AgentHooks | None = None
) -> str | None:
    """
    Resolve a session name to its id. Raises ValueError if missing.

    Parameters
    ----------
        conn:      Connection to the database.
        sess_name: Name of this session.
        hooks:     Function hooks for cli/tui integration.

    Returns
    -------
        - Session id.
    """
    row = fetch_one(
        conn=conn,
        sql="SELECT session_id FROM chat_sessions WHERE session_name = %s",
        params=(sess_name,),
    )
    if row is None:
        err_msg = "No session named '%s'" % sess_name
        if hooks and hooks.on_stat_change:
            hooks.on_stat_change(err_msg)
        raise ValueError(err_msg)
    return row[0]


def _link_table_exists(conn) -> bool:
    """
    session_experts is only created once an experts is first enabled.

    Parameters
    ----------
        conn: Connection to the database.

    Returns
    -------
        - True if link table exists.
    """
    row = fetch_one(conn=conn, sql="SELECT to_regclass('session_experts');")
    return bool(row and row[0])


def sessions_using_expert(conn, exp_id: int) -> list[tuple[str, str]]:
    """
    Find all the session the specified expert is enabled in.
    Returns (session_id, session_name) for each.

    Parameters
    ----------
        conn:   Connection to the database.
        exp_id: Expert's id.

    Returns
    -------
        - List of session id and name that uses that expert.
    """
    ensure_link_schema(conn)
    rows = fetch_all(
        conn=conn,
        sql=("""
            SELECT s.session_id, s.session_name
            FROM session_experts se
            JOIN chat_sessions s ON s.session_id = se.session_id
            WHERE se.expert_id = %s
            ORDER BY s.session_name;
            """),
        params=(exp_id,),
    )
    return [(str(row[0]), str(row[1])) for row in rows]


class SessionExperts:
    """
    The experts enabled in a chat session (multiple experts per session).

    Parameters
    ----------
        self.conn:     Connection to postgres.
        self.sess_id:  Given session id.
        self.exp_name: Existing expert name.
        self.exp_id:   Optional, specify only from Expert's self.exp_id.
    """

    def __init__(
        self,
        conn,
        sess_name: str,
        exp_name: str,
        exp_id: int | None = None,
        sess_id: str | None = None,
        hooks: AgentHooks | None = None,
    ):
        self.conn = conn

        self.sess_id = sess_id or _find_session_id_from_name(
            conn=self.conn, sess_name=sess_name
        )

        self.exp_id = exp_id or get_expert_id_from_name(self.conn, exp_name)
        if self.exp_id is None:
            err_msg = "Expert '%s' does not exist in the database" % exp_name
            if hooks and hooks.on_stat_change:
                hooks.on_stat_change(err_msg)
            raise ValueError(err_msg)

        ensure_link_schema(self.conn)

    def session_enabled_experts(self) -> list[tuple[int, str]]:
        """
        (id, name) of every expert enable in this session.

        Returns
        -------
            - List of session id and name that uses this expert.
        """
        rows = fetch_all(
            conn=self.conn,
            sql=("""
                SELECT e.id, e.name
                FROM session_experts se
                JOIN experts e ON e.id = se.expert_id
                WHERE se.session_id = %s
                ORDER BY e.name;
                """),
            params=(self.sess_id,),
        )
        return [(int(r[0]), str(r[1])) for r in rows]

    def session_enabled_expert_ids(self) -> list[int]:
        """
        Ids of every expert enabled in this session.

        Returns
        -------
            - List of Ids of every expert enabled in this session.
        """
        return [exp_id for exp_id, _ in self.session_enabled_experts()]

    def enable_expert(self) -> None:
        """Enable an expert in this session."""
        execute(
            conn=self.conn,
            sql=("""
                INSERT INTO session_experts (session_id, expert_id)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING;
                """),
            params=(self.sess_id, self.exp_id),
        )
        app_log.info("Expert id '%s' enable in session '%s'", self.exp_id, self.sess_id)

    def disable_expert(self) -> bool:
        """
        Disable an expert for a session. Returns false if the expert
        wasn't enabled in this session.

        Returns
        -------
            - A boolean if the expert has been disabled for this session.
        """
        return (
            execute(
                conn=self.conn,
                sql="DELETE FROM session_experts WHERE session_id = %s AND expert_id = %s;",
                params=(self.sess_id, self.exp_id),
            )
            > 0
        )

    def link_expert_to_session(self, sess_name: str) -> bool:
        """
        Enable this expert in a session. Raises ValueError
        if the session does not exist.

        Parameters
        ----------
            sess_name: Session name.

        Returns
        -------
            - False if it was already enabled.
        """
        ensure_link_schema(self.conn)  # Also creates chat_sessions
        sess_id = _find_session_id_from_name(self.conn, sess_name)
        return (
            execute(
                conn=self.conn,
                sql=("""
                INSERT INTO session_experts (session_id, expert_id)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING;
                """),
                params=(sess_id, self.exp_id),
            )
            > 0
        )

    def unlink_expert_from_session(self, sess_name: str) -> bool:
        """
        Disable this expert in a session, only the link row is
        removed. Raises ValueError if the session does not exist.

        Parameters
        ----------
            sess_name: Session name.

        Returns
        -------
            - False if it was not enabled.
        """
        if not _link_table_exists(self.conn):
            return False
        sess_id = _find_session_id_from_name(self.conn, sess_name)
        return (
            execute(
                conn=self.conn,
                sql=("""
                DELETE FROM session_experts
                WHERE session_id = %s AND expert_id = %s;
                """),
                params=(sess_id, self.exp_id),
            )
            > 0
        )

    @staticmethod
    def all_expert_data_from_session(
        conn, sess_name: str | None = None
    ) -> dict[str, dict[str, Any]]:
        """
        Return the experts with relevant data enabled in a session,
        keyed by expert name.

        Parameters
        ----------
            conn:      Connection to the database.
            sess_name: Session name.

        Returns
        -------
            - Structured dictionary about this session's expert's
              details.
        """
        ensure_schema(conn)
        if not sess_name or not _link_table_exists(conn):
            return {}
        rows = fetch_all(
            conn=conn,
            sql=("""
                SELECT e.name, e.description, e.created_at, e.updated_at,
                    se.enabled_at,
                    (SELECT COUNT(*) FROM expert_documents ed
                    WHERE ed.expert_id = e.id)
                FROM session_experts se
                JOIN experts e ON e.id = se.expert_id
                JOIN chat_sessions s ON s.session_id = se.session_id
                WHERE s.session_name = %s
                ORDER BY e.name;
                """),
            params=(sess_name,),
        )
        return {
            row[0]: {
                "description": row[1],
                "created_at": row[2],
                "updated_at": row[3],
                "enabled_at": row[4],
                "document_count": row[5],
            }
            for row in rows
        }
