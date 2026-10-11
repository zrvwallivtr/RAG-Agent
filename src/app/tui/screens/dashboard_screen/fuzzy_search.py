from difflib import get_close_matches

from textual.widgets import Input, OptionList
from textual.widgets.option_list import Option

from src.config.postgres import CONN

from src.libs.helpers.date_helpers import get_session_with_dates


TITLE = "Last modified\t\tCreated at\t\tSession name\n"
sessions: list[tuple[str, str]] = []


def load_sessions() -> None:
    """Query the database ONCE. Call when the picker opens and after create/delete."""
    from src.agent.chat_session_logs import ChatSessionLogs
    sess_dict = ChatSessionLogs(conn=CONN).get_all_existing_session_metadata() or {}

    loaded: list[tuple[str, str]] = []
    for sess_id, data in sess_dict.items():
        name = data["session_name"]
        label = get_session_with_dates(sess_dict=sess_dict, sess_id=sess_id)
        loaded.append((name, label or name)) # fallback to the bare name

    sessions[:] = loaded # Store to the existing list


def show_session_list(sess_list: OptionList) -> None:
    """Show list of sessions with created time, modified time and session name."""
    from src.agent.chat_session_logs import ChatSessionLogs
    chat_sess_logs = ChatSessionLogs(conn=CONN)

    sess_dict = chat_sess_logs.get_all_existing_session_metadata()
    if not sess_dict:
        return

    sess_list.clear_options()

    list_title = Option("Last modified\t\tCreated at\t\tSession name\n", id="session-option-list-title")
    list_title.disabled = True
    sess_list.add_option(list_title)

    for sess_id in sess_dict:
        sess_name = sess_dict[sess_id]["session_name"]
        sess_with_dates = get_session_with_dates(
            sess_dict=sess_dict, sess_id=sess_id
        )
        sess_list.add_option(Option(sess_with_dates, id=sess_name))

    sess_list.remove_class("hidden")


def fuzzy_search_behaviour(event: Input.Changed, opt_list: Option) -> None:
    qry = event.value.strip().lower()
    names_lower = [n.lower() for n, _ in sessions]

    if not qry:
        hits = sessions
    else:
        close = set(get_close_matches(qry, names_lower, n=5, cutoff=0.4))
        hits = [s for s, low in zip(sessions, names_lower) if qry in low or low in close]

    opt_list.clear_options()
    opt_list.add_options(
        [Option(TITLE, disabled=True)] + [Option(label, id=name) for name, label in hits]
    )
    if hits:
        opt_list.highlighted = 1
