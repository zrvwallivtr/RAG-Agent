import shlex
from pathlib import Path

from textual.screen import Screen
from textual.widgets import Input
from textual.events import Key


class BaseScreen(Screen):
    CSS_PATH = "tcss/base.tcss"


    def __init__(self):
        super().__init__()
        self.is_displaying_message = False


    def compose_command_bar(self):
        """Shared command bar for every screen in the app."""
        yield Input(id="cmd-input", classes="hidden", select_on_focus=False)


    def on_key(self, event: Key) -> None:
        """Show command bar on ':' key press."""
        cmd_input = self.query_one("#cmd-input", Input)

        if event.character == ":":
            # Do not activate command bar if focus is on prompt input
            if self.focused and self.focused.id == "prompt-input":
                return

            if not cmd_input.has_focus:
                event.prevent_default()
                event.stop()
                self.is_displaying_message = False
                cmd_input.remove_class("hidden")
                cmd_input.value = ":"
                cmd_input.cursor_position = 1
                cmd_input.focus()

        if event.key == "escape":
            if cmd_input.has_focus:
                cmd_input.value = ""
                cmd_input.add_class("hidden")
                self.set_focus(None)


    def on_input_changed(self, event: Input.Changed) -> None:
        """Hide command bar automatically if ':' is deleted."""
        if event.input.id != "cmd-input":
            return

        # Do not disapear if a message is displaying
        if self.is_displaying_message:
            return

        if not event.value.startswith(":"):
            event.input.value = ""
            event.input.add_class("hidden")
            self.set_focus(None)


    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Valid commands, show error message when invalid command is submitted."""
        if event.input.id != "cmd-input":
            return

        cmd = event.value.strip()

        from src.app.tui.screens.dashboard_screen import DashboardScreen
        from src.app.tui.screens.chat_screen import ChatScreen

        if cmd in (":q", ":quit"):
            self.app.exit()
            self._reset_and_hide_command_bar(event=event)

        elif cmd in (":d", ":dashboard"):
            self.app.switch_screen(DashboardScreen())
            self._reset_and_hide_command_bar(event=event)

        elif cmd in (":s", ":session"):
            self.app.push_screen(ChatScreen(sess_name=None))
            self._reset_and_hide_command_bar(event=event)

        elif cmd.startswith(":s ") or cmd.startswith(":session "):
            prefix = ":s " if cmd.startswith(":s ") else ":session "
            sess_name = cmd.split(prefix, 1)[1].strip()
            self.app.push_screen(ChatScreen(sess_name=sess_name))
            self._reset_and_hide_command_bar(event=event)

        elif isinstance(self, ChatScreen):
            if cmd.startswith(":a ") or cmd.startswith(":attach "):
                prefix = ":a " if cmd.startswith(":a ") else ":attach "
                self.filter_and_add_pending(cmd=cmd, prefix=prefix)
                return

        else:
            self.show_command_message(f"Not a valid command: {cmd}")
            return


    def show_command_message(self, msg: str, duration: float = 3.0) -> None:
        """Display message in the command bar and unfoucs."""
        cmd_input = self.query_one("#cmd-input", Input)

        self.is_displaying_message = True

        cmd_input.remove_class("hidden")
        cmd_input.value = msg

        self.set_focus(None)

        self.set_timer(duration, self._clear_command_bar)


    def _clear_command_bar(self) -> None:
        """Clear all contents in the command bar and hides it."""
        cmd_input = self.query_one("#cmd-input", Input)
        cmd_input.value = ""
        cmd_input.add_class("hidden")


    def _reset_and_hide_command_bar(self, event: Input.Submitted) -> None:
        event.input.value = ""
        event.input.add_class("hidden")
        self.set_focus(None)
