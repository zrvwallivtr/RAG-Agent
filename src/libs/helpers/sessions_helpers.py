from src.config.postgres import CONN

from src.libs.helpers.database_helpers import execute
from src.libs.custom_dataclasses import AgentHooks

from src.agent.chat_session_logs import ChatSessionLogs
from src.rag.knowledge_base import KnowledgeBase
from src import logger

app_log = logger.app_logger(f"{__name__}.app")


def _delete_chat_session(
    chat_sess_logs: ChatSessionLogs,
    sess_name: str | None = None,
    hooks: AgentHooks | None = None,
):
    """
    Delete the chat session entry on the chat session table from the
    database.

    ONLY CALL THIS WHEN ALL THE OTHER DATABASE LINKED TO A SESSION IS
    DELETED.
    """
    sess_id = chat_sess_logs.get_session_id()
    try:
        row = execute(
            conn=CONN,
            sql=("""
                DELETE FROM chat_sessions
                WHERE session_id = %s;
                """),
            params=(sess_id,),
        )
        info_str = (
            "Session '%s' chat logs and all related contents "
            "has been deleted from the database" % sess_name
        )
        app_log.info(info_str)
        if hooks and hooks.on_stat_change:
            hooks.on_stat_change(info_str)
    except Exception as e:
        err_msg = Exception("Failed to delete session '%s': %s" % (sess_name, e))
        app_log.warning(err_msg)
        if hooks and hooks.on_error:
            hooks.on_error(err_msg)


def delete_session(
    sess_name: str | None = None,
    hooks: AgentHooks | None = None,
) -> None:
    """
    Delete session related chat logs and database contents. If 'session'
    is not specified, delete default session related contents.
    """
    chat_sess_logs = ChatSessionLogs(conn=CONN, sess_name=sess_name)
    kw_bs = KnowledgeBase(conn=CONN, chat_sess_logs=chat_sess_logs)

    chat_sess_logs.clear_session_chat_logs()
    kw_bs.clear_sess_kw_bs()

    _delete_chat_session(chat_sess_logs, sess_name, hooks=hooks)
