import shlex
from pathlib import Path

from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import Input, Static
from textual.events import Key, DescendantFocus, DescendantBlur

from .tab_completion import tab_completion


class BaseScreen(Screen):
    CSS_PATH = "../tcss/base.tcss"

    def __init__(self):
        super().__init__()
        self.is_displaying_message = False

    def compose_command_bar(self):
        """
        Shared command bar for every screen:
            row 1: ruler (location)
            row 2: command input (mode as placeholder) + pending-key display
        """
        with Vertical(id="bottom-bar"):
            yield Static("", id="ruler")
            with Horizontal(id="cmd-row"):
                yield Input(id="cmd-input", select_on_focus=False)
                yield Static("", id="showcmd")

    def mode_text(self) -> str:
        """Left side of the command bar, e.g. '-- INSERT --'."""
        focused = self.focused
        if isinstance(focused, Input) and focused.id != "cmd-input":
            return "-- INSERT --"
        return ""

    def ruler_text(self) -> str:
        """Line above the command line (location). Empty hides the line."""
        return ""

    def showcmd_text(self) -> str:
        """Right side of the command line (pending keys)."""
        return ""

    def refresh_status(self) -> None:
        """Redraw mode, ruler and pending keys."""
        try:
            cmd_input = self.query_one("#cmd-input", Input)
            ruler = self.query_one("#ruler", Static)
            showcmd = self.query_one("#showcmd", Static)
        except NoMatches:
            return
        
        cmd_input.placeholder = self.mode_text()
        text = self.ruler_text()
        ruler.update(text)
        ruler.display = bool(text)

        showcmd.update(self.showcmd_text())

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        self.call_after_refresh(self.refresh_status)

    def on_descendant_blur(self, event: DescendantBlur) -> None:
        self.call_after_refresh(self.refresh_status)

    def on_key(self, event: Key) -> None:
        """Focus the command bar on ':' key press."""
        self.call_after_refresh(self.refresh_status)

        cmd_input = self.query_one("#cmd-input", Input)

        if event.character == ":":
            # Do not activate command bar if focus is on prompt input
            if self.focused and self.focused.id == "prompt-input":
                return

            if not cmd_input.has_focus:
                event.prevent_default()
                event.stop()
                self.is_displaying_message = False
                cmd_input.value = ":"
                cmd_input.cursor_position = 1
                cmd_input.focus()

        if event.key == "escape":
            if cmd_input.has_focus:
                self._reset_command_bar()

        if event.key == "tab":
            if cmd_input.has_focus:
                tab_completion(self, cmd_input=cmd_input, event=event)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Hide command bar automatically if ':' is deleted."""
        if event.input.id != "cmd-input":
            return

        # Do not disapear if a message is displaying
        if self.is_displaying_message:
            return

        # Only react to the user typing in the bar, not to programmatic clears
        if not event.input.has_focus:
            return

        if not event.value.startswith(":"):
            self._reset_command_bar()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Valid commands, show error message when invalid command is submitted."""
        if event.input.id != "cmd-input":
            return

        cmd = event.value.strip()

        from src.app.tui.screens.dashboard_screen import DashboardScreen
        from src.app.tui.screens.chat_screen import ChatScreen

        if cmd in (":q", ":quit"):
            self.app.exit()
            self._reset_command_bar()

        elif cmd in (":d", ":dashboard"):
            self.app.switch_screen(DashboardScreen())
            self._reset_command_bar()

        elif cmd in (":s", ":session"):
            self.app.push_screen(ChatScreen(sess_name=None))
            self._reset_command_bar()

        elif cmd.startswith(":s ") or cmd.startswith(":session "):
            prefix = ":s " if cmd.startswith(":s ") else ":session "
            sess_name = cmd.split(prefix, 1)[1].strip()
            self.app.push_screen(ChatScreen(sess_name=sess_name))
            self._reset_command_bar()

        elif isinstance(self, ChatScreen) and cmd.startswith((":a ", ":attach ")):
            prefix = ":a " if cmd.startswith(":a ") else ":attach "
            self.run_process_add_pending_attachments(cmd=cmd, prefix=prefix)

        else:
            self.show_command_message(f"Not a valid command: {cmd}")

    def show_command_message(self, msg: str, duration: float = 3.0) -> None:
        """Display message in the command bar and unfoucs."""
        cmd_input = self.query_one("#cmd-input", Input)

        self.is_displaying_message = True

        cmd_input.value = msg

        self.set_focus(None)

        self.set_timer(duration, self._clear_command_bar)

    def _clear_command_bar(self) -> None:
        """Empty the bar after a message. Skipped if the user started typing."""
        if not self.is_displaying_message:
            return
        cmd_input = self.query_one("#cmd-input", Input)
        cmd_input.value = ""
        self.is_displaying_message = False

    def _reset_command_bar(self) -> None:
        """Empty the bar and drop focus."""
        cmd_input = self.query_one("#cmd-input", Input)
        self.is_displaying_message = False
        cmd_input.value = ""
        self.set_focus(None)
