from textual.app import App
from textual.widgets import Input, OptionList

from src.core import Agent
from src.agent.models import ollama

from src.app.tui.screens.base_screen import BaseScreen
from src.app.tui.screens.dashboard_screen import DashboardScreen
from src.app.tui.screens.chat_screen import ChatScreen


class MainApp(App):
    """Entry point for the TUI."""

    BINDINGS = [
        ("escape", "escape_handler"),
    ]

    def __init__(self, sess_name: str | None = None):
        super().__init__()
        self.sess_name = sess_name

    def compose(self):
        """Returns nothing."""
        return []

    def on_mount(self) -> None:
        """Show menu screen on startup."""
        if self.sess_name:
            self.push_screen(ChatScreen(sess_name=self.sess_name))
        else:
            self.push_screen(DashboardScreen())

    def action_escape_handler(self) -> None:
        """All scenarios for when the escape button is pressed."""
        cmd_input = self.screen.query_one("#cmd-input", Input)

        # Close command bar if focused
        if cmd_input.has_focus:
            cmd_input.value = ""
            cmd_input.add_class("hidden")
            self.set_focus(None)
            return

        # Unfocus screen if focused
        if self.focused is not None:
            self.set_focus(None)
            return

        # Unhighlight if highlighted
        for opt_list in self.query(OptionList):
            if opt_list.highlighted is not None:
                opt_list.highlighted = None
