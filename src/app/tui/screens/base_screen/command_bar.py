from textual.widgets import Input


def _clear_command_bar(screen) -> None:
    """Empty the bar after a message. Skipped if the user started typing."""
    if not screen.is_displaying_message:
        return
    cmd_input = screen.query_one("#cmd-input", Input)
    cmd_input.value = ""
    screen.is_displaying_message = False


def show_command_message(screen, msg: str, duration: float = 3.0) -> None:
    """Display message in the command bar and unfoucs."""
    cmd_input = screen.query_one("#cmd-input", Input)

    screen.is_displaying_message = True
    cmd_input.value = msg
    screen.set_focus(None)

    screen.set_timer(duration, lambda: _clear_command_bar(screen))


def reset_command_bar(screen) -> None:
    """Empty the bar and drop focus."""
    cmd_input = screen.query_one("#cmd-input", Input)
    screen.is_displaying_message = False
    cmd_input.value = ""
    screen.set_focus(None)


def showcmd_text() -> str:
    """Right side of the command line (pending keys)."""
    return ""
