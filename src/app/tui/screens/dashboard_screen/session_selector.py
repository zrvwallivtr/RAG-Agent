from psycopg2.errors import UniqueViolation

from textual.widgets import Static, Input, OptionList

from src.config.postgres import CONN

from src.libs.helpers.sessions_helpers import delete_session

from .fuzzy_search import load_sessions, show_session_list, sessions
from .action_bar import show_session_action_bar


def go_to_session(screen, sess_name: str) -> None:
    """Go to specified chat session screen."""
    # screen.notify(f"Opening session: {sess_name}")
    from src.app.tui.screens.chat_screen import ChatScreen
    screen.app.push_screen(ChatScreen(sess_name=sess_name))


def reset_session_list_layout(screen) -> None:
    """Return to default sessions view if in an input mode."""
    if screen.in_sess_search or screen.in_new_sess_input or screen.in_del_sess_input:
        screen.in_new_sess_input = False
        screen.in_del_sess_input = False
        show_sessions(screen)


def option_list_highlight_none(screen) -> None:
    """Remove all highlightings if any OptionList is highlighted."""
    for opt_list in screen.query(OptionList):
        if opt_list.highlighted is not None:
            opt_list.highlighted = None


def show_sessions(screen) -> None:
    """
    Consist:
    - Show latest sessions. If selected go to the chat screen for said session.
    - Show go to session option. If selected, the fuzzy search session selector
      will popup.
    """
    screen.in_sess_search = False
    screen.in_new_sess_input = False
    screen.in_del_sess_input = False
    screen.curr_to_sess = None

    load_sessions()
    count = len(sessions)

    title = (
        f"[bold]Sessions[/bold] ({count}) created"
        if count
        else f"[bold]Sessions[/bold] (0 created)\n"
    )
    screen.query_one("#session-section-title", Static).update(title)

    # Session list
    sess_list = screen.query_one("#session-list", OptionList)
    show_session_list(sess_list=sess_list)

    show_session_action_bar(screen)

    # Hide input bar
    screen.query_one("#session-search-input", Input).add_class("hidden")
    screen.query_one("#new-session-input", Input).add_class("hidden")
    screen.query_one("#delete-session-input", Input).add_class("hidden")


# --- Search session ---------------------------------------------

def show_session_search_input(screen) -> None:
    """
    Consist of the search input bar and available sessions list.
    Session list will be updated according to the search bar.
    """
    screen.in_sess_search = True # User is searching sessions

    # Session option list
    sess_list = screen.query_one("#session-list", OptionList)

    # Highlight first match
    if sess_list.option_count > 1:
        sess_list.highlighted = 1

    # Fuzzy search
    search_input = screen.query_one("#session-search-input", Input)
    search_input.value = ""
    search_input.remove_class("hidden")
    sess_list.remove_class("hidden")
    search_input.focus()

    # Hide session search input trigger
    screen.query_one("#session-action-bar").add_class("hidden")


# --- Create new session -----------------------------------------

def show_new_session_input(screen) -> None:
    """
    Consist of the input bar for naming the new session.
    Show existing session list with fuzzy search on.
    """
    screen.in_new_sess_input = True # User is entering session name
    sess_list = screen.query_one("#session-list", OptionList)
    show_session_list(sess_list=sess_list)

    # Existing session list
    sess_list = screen.query_one("#session-list", OptionList)

    # Highlight first match
    if sess_list.option_count > 1:
        sess_list.highlighted = 1

    # Input bar
    new_sess_input = screen.query_one("#new-session-input", Input)
    new_sess_input.value = ""
    new_sess_input.remove_class("hidden")
    sess_list.remove_class("hidden")
    new_sess_input.focus()

    # Hide widgets
    screen.query_one("#session-action-bar").add_class("hidden")


def create_new_session(screen, sess_name: str) -> None:
    """Create new session."""
    clean_name = sess_name.strip()
    if not clean_name:
        screen.notify("Session name cannot be empty.", severity="error")
        return

    from src.agent.chat_session_logs import ChatSessionLogs
    chat_sess_logs = ChatSessionLogs(conn=CONN, sess_name=clean_name)

    try:
        chat_sess_logs.create_session()

    except UniqueViolation:
        chat_sess_logs.conn.rollback()
        screen.notify(f"Session '{clean_name}' already exists", severity="error")
        return

    screen.notify(f"New session created: {clean_name}")
    reset_session_list_layout(screen)
    go_to_session(screen, clean_name)


# --- Delete session ---------------------------------------------

def show_delete_session_input(screen) -> None:
    """
    Consist of the input bar for available sessions.
    Show existing session list with fuzzy search on.
    """
    screen.in_del_sess_input = True # User is entering session name

    # Existing session list
    sess_list = screen.query_one("#session-list", OptionList)

    # Input bar
    del_sess_input = screen.query_one("#delete-session-input", Input)
    del_sess_input.value = ""
    del_sess_input.remove_class("hidden")
    sess_list.remove_class("hidden")
    del_sess_input.focus()

    # Hide widgets
    screen.query_one("#session-action-bar").add_class("hidden")


def _delete_session(screen, sess_name: str) -> None:
    """Delete session."""
    result = delete_session(sess_name=sess_name)

    if result and "fail" in result.lower():
        screen.notify(result, severity="error")
        return

    screen.notify(f"Session deleted: {sess_name}")
    show_sessions(screen)


def confirm_and_delete_session(screen, sess_name: str) -> None:
    clean_name = sess_name.strip()
    if not clean_name:
        screen.notify("Session name cannot be empty.", severity="error")
        return

    from src.app.tui.screens.confirmation_popup_screen import ConfirmationPopupScreen

    def handle_confirmation(is_confirm: bool | None) -> None:
        if is_confirm:
            _delete_session(screen, sess_name=clean_name)

    title = f"Proceed to delete session '{sess_name}'?"
    positive = "Proceed"
    negative = "Cancel"
    screen.app.push_screen(
        ConfirmationPopupScreen(title=title, positive=positive, negative=negative),
        callback=handle_confirmation
    )
