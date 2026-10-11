from src.config.postgres import CONN

from src.agent.chat_session_logs import ChatSessionLogs
from src.rag.knowledge_base import KnowledgeBase
from src import logger

CUR = CONN.cursor()

app_log = logger.app_logger(f"{__name__}.app")


def del_sess(sess_name: str | None = None) -> None:
    """
    Delete session related chat logs and database contents. If 'session'
    is not specified, delete default session related contents.
    """
    chat_sess_logs   = ChatSessionLogs(conn=CONN, sess_name=sess_name)
    kw_bs       = KnowledgeBase(conn=CONN, chat_sess_logs=chat_sess_logs)

    sess_id = chat_sess_logs.get_session_id()
    if not sess_id:
        app_log.warning("Failed to delete session: Session '%s' does not exist", sess_name)
        return

    # Clear session chat logs
    has_del_chat = chat_sess_logs.clear_session_chat_logs()

    # Clear session knowledge base
    response = kw_bs.clear_sess_kw_bs("document")
    # response = kw_bs.clear_sess_kw_bs("web_search")

    # Delete session - ensure all session related contents are cleared
    if not has_del_chat:
        app_log.warning("Failed to clear session '%s' chat logs", sess_name)

    CUR.execute(
        """
        DELETE FROM chat_sessions
        WHERE session_id = %s;
        """,
        (sess_id,)
    )
    CONN.commit()
    del_count = CUR.rowcount
    CONN.commit()

    if del_count == 0:
        app_log.warning("Failed to delete session: Session '%s' does not exist", sess_name)
        return
    app_log.info("Session '%s' and related data deleted from database", sess_name)
    return
