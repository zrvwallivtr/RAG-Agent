from datetime import datetime

from src.config.postgres import CONN


def trimmed_date(iso_str: str | None) -> str:
    """
    Trim the date format from 'ISO 8601 with microseconds' to
    'DD/MM/YYYY HH:MM'.
    """
    if not iso_str:  # Fallback
        return "    ---         "

    dt = datetime.fromisoformat(iso_str.split("+")[0])
    return dt.strftime("%d/%m/%Y %H:%M")


def get_session_with_dates(sess_dict: dict, sess_id: str) -> str | None:
    """
    Return a string containing session's:
    last modified at    created at    session name

    Applicable for a selection list, not suitable for a table.
    """
    from src.agent.chat_session_logs import ChatSessionLogs

    chat_session_logs = ChatSessionLogs(conn=CONN)

    sess_name = sess_dict[sess_id]["session_name"]
    created_at = trimmed_date(iso_str=sess_dict[sess_id]["created_at"])

    last_mod_iso = chat_session_logs.get_session_last_modified_time(sess_id=sess_id)
    if not last_mod_iso:
        last_mod_iso = sess_dict[sess_id]["created_at"]

    last_mod = trimmed_date(iso_str=last_mod_iso)
    sep = " "

    return f"[dim]{last_mod}{sep}\t" f"{created_at}{sep}[/]\t" f"{sess_name}"
