from typing import Literal

from textual import work
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static, Input, Markdown
from textual.events import Key

from src.core import Agent
from src.tui.screens.base_screen import BaseScreen


class ChatScreen(BaseScreen):
    CSS_PATH = "tcss/chat_interface.tcss"


    def __init__(self, sess_name: str | None):
        super().__init__()
        self.agent = Agent(sess_name=sess_name)
        self.sess_name = sess_name or "Default session"
        self.pending_key: str | None = None


    def compose(self) -> ComposeResult:
        """Draw chat interface."""
        with Vertical():
            with VerticalScroll(id="chat_container"):
                pass
            yield Input(placeholder="Write a message...", id="prompt_input", select_on_focus=False)
        yield from self.compose_command_bar()


    def on_mount(self) -> None:
        """Load entire history on start up."""
        self.load_chat_history()


    def load_chat_history(self) -> None:
        """Get entire chat history from the database and display it on to the interface."""
        chat_container = self.query_one("#chat_container", VerticalScroll)

        chat_hist = self.agent.chat_logs.get_chat_history(filter="all")
        if not chat_hist:
            return

        for msg in chat_hist:
            role = msg.get("role")
            cont = msg.get("content", "")

            if role == "user":
                msg_box = self.create_message_box(
                    role_label="USER", cont_widget=Static(cont), css_class="user-box"
                )
                chat_container.mount(msg_box)

            elif role == "assistant":
                msg_box = self.create_message_box(
                    role_label="ASSISTANT", cont_widget=Markdown(cont), css_class="assistant-box"
                )
                chat_container.mount(msg_box)

        chat_container.scroll_end(animate=False)


    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Displays user prompt and assistant response."""
        super().on_input_submitted(event)

        if event.input.id != "prompt_input":
            return

        prompt = event.value.strip()
        if not prompt:
            return

        chat_container = self.query_one("#chat_container", VerticalScroll)
        chat_container.mount(
            self.create_message_box(
                role_label="USER", cont_widget=Static(prompt), css_class="user-box"
            )
        )
        chat_container.scroll_end(animate=False)

        self.query_one("#prompt_input", Input).clear()
        self.fetch_agent_response(prompt)


    @work(exclusive=True, thread=True)
    def fetch_agent_response(self, prompt: str) -> None:
        """Sends user prompt to agent and streams its response in Markdown format."""
        chat_container = self.query_one("#chat_container", VerticalScroll)
        md_widget = Markdown("")

        assistant_box = self.create_message_box(
            role_label="ASSISTANT", cont_widget=md_widget, css_class="assistant-box"
        )
        self.app.call_from_thread(chat_container.mount, assistant_box)

        full_txt = ""

        # Allow streaming
        def on_token(tkn: str) -> None:
            nonlocal full_txt
            full_txt += tkn
            self.app.call_from_thread(md_widget.update, full_txt)
            self.app.call_from_thread(chat_container.scroll_end, animate=False)

        self.agent.ask(prompt=prompt, callback=on_token)


    def create_message_box(
        self,
        role_label: Literal["USER", "ASSISTANT"],
        cont_widget: Static | Markdown,
        css_class: str
    ) -> Vertical:
        """Contains the chat content from user/assistant."""
        return Vertical(
            Static(f"[{role_label}]:", classes="msg-label", markup=False),
            cont_widget,
            classes=f"msg-box {css_class}"
        )


    def on_key(self, event: Key) -> None:
        """Keybindings in normal mode."""
        super().on_key(event)
        prompt_input = self.query_one("#prompt_input", Input)
        cmd_input = self.query_one("#cmd_input", Input)

        if prompt_input.has_focus or cmd_input.has_focus:
            return

        scroll_bar = self.query_one("#chat_container", VerticalScroll)

        if event.key == "k":
            scroll_bar.scroll_up()
            self.pending_key = None

        elif event.key == "j":
            scroll_bar.scroll_down()
            self.pending_key = None

        elif event.character == "G":
            scroll_bar.scroll_end(animate=False)
            self.pending_key = None

        elif event.character == "g":
            if self.pending_key == "g":
                scroll_bar.scroll_home(animate=False)
                self.pending_key = None
            else:
                self.pending_key = "g"

        else:
            self.pending_key = None
