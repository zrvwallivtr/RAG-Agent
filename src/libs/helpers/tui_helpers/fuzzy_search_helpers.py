from difflib import get_close_matches

from textual.widgets import Input, OptionList
from textual.widgets.option_list import Option

from src.config import postgres

from src.libs.helpers.tui_helpers import sessions_helpers, date_helpers


TITLE = "Last modified\t\tCreated at\t\tSession name\n"
_sessions: list[tuple[str, str]] = []


def load_sessions() -> None:
    """Query the database ONCE. Call when the picker opens and after create/delete."""
    from src.agent.chat_logs import ChatSessionLogs
    sess_dict = ChatSessionLogs(conn=postgres.conn).get_all_existing_session_metadata() or {}

    loaded: list[tuple[str, str]] = []
    for sess_id, data in sess_dict.items():
        name = data["session_name"]
        label = date_helpers.get_session_with_dates(sess_dict=sess_dict, sess_id=sess_id)
        loaded.append((name, label or name)) # fallback to the bare name

    _sessions[:] = loaded # Store to the existing list


def show_session_list(sess_list: OptionList) -> None:
    """Show list of sessions with created time, modified time and session name."""
    from src.agent.chat_logs import ChatSessionLogs
    chat_logs = ChatSessionLogs(conn=postgres.conn)

    sess_dict = chat_logs.get_all_existing_session_metadata()
    if not sess_dict:
        return

    sess_list.clear_options()

    list_title = Option("Last modified\t\tCreated at\t\tSession name\n", id="session-option-list-title")
    list_title.disabled = True
    sess_list.add_option(list_title)

    for sess_id in sess_dict:
        sess_name = sess_dict[sess_id]["session_name"]
        sess_with_dates = date_helpers.get_session_with_dates(
            sess_dict=sess_dict, sess_id=sess_id
        )
        sess_list.add_option(Option(sess_with_dates, id=sess_name))

    sess_list.remove_class("hidden")


def fuzzy_search_behaviour(event: Input.Changed, opt_list: Option) -> None:
    qry = event.value.strip().lower()
    names_lower = [n.lower() for n, _ in _sessions]

    if not qry:
        hits = _sessions
    else:
        close = set(get_close_matches(qry, names_lower, n=5, cutoff=0.4))
        hits = [s for s, low in zip(_sessions, names_lower) if qry in low or low in close]

    opt_list.clear_options()
    opt_list.add_options(
        [Option(TITLE, disabled=True)] + [Option(label, id=name) for name, label in hits]
    )
    if hits:
        opt_list.highlighted = 1
