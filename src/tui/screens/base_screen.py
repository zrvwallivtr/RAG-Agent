from textual.screen import Screen
from textual.widgets import Input
from textual.events import Key


class BaseScreen(Screen):
    def __init__(self):
        super().__init__()
        self.is_displaying_message = False


    def compose_command_bar(self):
        """Shared command bar for every screen in the app."""
        yield Input(id="cmd_input", classes="hidden", select_on_focus=False)


    def on_input_changed(self, event: Input.Changed) -> None:
        """Hide command bar automatically if ':' is deleted."""
        if event.input.id != "cmd_input":
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
        if event.input.id != "cmd_input":
            return

        cmd = event.value.strip()

        from src.tui.screens.menu_screen import MenuScreen
        from src.tui.screens.chat_screen import ChatScreen

        if cmd in (":q", ":quit"):
            self.app.exit()

        elif cmd in (":m", ":menu"):
            self.app.switch_screen(MenuScreen())

        elif cmd in (":s", ":session"):
            self.app.push_screen(ChatScreen(sess_name=None))

        elif cmd.startswith(":s ") or cmd.startswith(":session "):
            prefix = ":s " if cmd.startswith(":s ") else ":session "
            sess_name = cmd.split(prefix, 1)[1].strip()
            self.app.push_screen(ChatScreen(sess_name=sess_name))

        else:
            self.show_command_message(f"Not a valid command: {cmd}")
            return

        # Reset and hide command bar after submission
        event.input.value = ""
        event.input.add_class("hidden")
        self.set_focus(None)


    def show_command_message(self, msg: str, duration: float = 3.0) -> None:
        """Display message in the command bar and unfoucs."""
        cmd_input = self.query_one("#cmd_input", Input)

        self.is_displaying_message = True

        cmd_input.remove_class("hidden")
        cmd_input.value = msg

        self.set_focus(None)

        self.set_timer(duration, self._clear_command_bar)


    def _clear_command_bar(self) -> None:
        """Clear all contents in the command bar and hides it."""
        cmd_input = self.query_one("#cmd_input", Input)
        cmd_input.value = ""
        cmd_input.add_class("hidden")


    def on_key(self, event: Key) -> None:
        """Show command bar on ':' key press."""
        cmd_input = self.query_one("#cmd_input", Input)

        if event.character == ":":
            if self.focused and self.focused.id == "prompt_input":
                return

            if not cmd_input.has_focus:
                event.prevent_default()
                event.stop()

                self.is_displaying_message = False
                cmd_input.remove_class("hidden")
                cmd_input.value = ":"
                cmd_input.cursor_position = 1
                cmd_input.focus()
