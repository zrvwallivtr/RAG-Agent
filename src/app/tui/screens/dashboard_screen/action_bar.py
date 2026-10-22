from textual.widgets import Button


def show_session_action_bar(screen) -> None:
    """Option list for search session, new session, delete session."""
    screen.in_sess_search = False  # User not searching sessions
    screen.curr_to_sess = None

    action_bar = screen.query_one("#session-action-bar")
    action_bar.remove_class("hidden")
    screen.query_one("#search-session-button", Button).focus()
