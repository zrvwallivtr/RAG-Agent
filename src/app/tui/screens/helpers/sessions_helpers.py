from src.config import models, postgres


def get_session_list() -> list | None:
    """Return all existing session as a list."""
    from src.agent.chat_logs import ChatLogs
    chat_logs = ChatLogs(conn=postgres.conn)

    sess_dict = chat_logs.get_all_existing_sess_metadata()
    if not sess_dict:
        return

    sess_list = []

    for sess in sess_dict:
        sess_list.append(sess_dict[sess]["session_name"])
    return sess_list
