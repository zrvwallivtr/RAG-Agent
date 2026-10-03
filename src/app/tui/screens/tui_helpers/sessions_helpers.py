from difflib import get_close_matches

from textual.widgets import Input, OptionList
from textual.widgets.option_list import Option

from src.config import postgres

from src.app.tui.screens.helpers import date_helpers


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


def _show_session_list(sess_list: OptionList) -> None:
    """
    Show list of sessions with created time, modified time and session name.
    """
    from src.agent.chat_logs import ChatLogs
    chat_logs = ChatLogs(conn=postgres.conn)

    sess_dict = chat_logs.get_all_existing_sess_metadata()
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


def _fuzzy_search_behaviour(
    event: Input.Changed,
    opt_list: OptionList
) -> None:
    """
    Filters the OptionList based on fuzzy matching.
    Controls:
    - The behaviour of the model list when the
      search input is changed.
    """
    from src.agent.chat_logs import ChatLogs
    chat_logs = ChatLogs(conn=postgres.conn)

    sess_dict = chat_logs.get_all_existing_sess_metadata()
    if not sess_dict:
        return

    sess_list = get_session_list()
    if not sess_list:
        return

    qry = event.value.strip().lower()
    opt_list.clear_options()

    list_title = Option("Last modified\t\tCreated at\t\tSession name\n")
    list_title.disabled = True
    opt_list.add_option(list_title)

    if not qry:
        # Show entire list if query is empty
        _show_session_list(sess_list=opt_list)

    else:
        # Create an option for every matches
        exact_matches = [entry for entry in sess_list if qry in entry.lower()]
        fuzzy_matches = get_close_matches(
            qry,
            [entry.lower() for entry in sess_list],
            n=5,
            cutoff=0.4
        )

        name_list = []
        for name in sess_list:
            if name in exact_matches or name.lower() in fuzzy_matches:
                name_list.append(name)

        for name in name_list:
            sess_id = chat_logs.get_sess_id_from_name(sess_name=name)
            if not sess_id:
                continue

            if not sess_dict:
                return

            sess_with_dates = date_helpers.get_session_with_dates(
                sess_dict=sess_dict, sess_id=sess_id
            )
            opt_list.add_option(Option(sess_with_dates, id=name))

    # Re-highlight first match after every update
    if opt_list.option_count > 1:
        opt_list.highlighted = 1
