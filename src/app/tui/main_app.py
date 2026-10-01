from textual.app import App
from textual.widgets import Input, OptionList

from src.config import models
from src.core import Agent
from src.agent.models import ollama

from src.app.tui.screens.base_screen import BaseScreen
from src.app.tui.screens.dashboard_screen import DashboardScreen
from src.app.tui.screens.chat_screen import ChatScreen


class MainApp(App):
    BINDINGS = [
        ("escape", "escape_handler"),
    ]


    def compose(self):
        """Returns nothing."""
        return []


    def on_mount(self) -> None:
        """Show menu screen on startup."""
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
