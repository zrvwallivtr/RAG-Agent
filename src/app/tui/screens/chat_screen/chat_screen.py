import threading
from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.widgets import Static, Input, Markdown, Button
from textual.events import Key

from src.config.models import MODEL
from src.config.postgres import conn

from .agent_response import run_fetch_agent_response
from .chat_container import (
    create_user_message_box,
    create_assistant_message_box,
    load_chat_history
)
from .pending_attachments import to_attachments_popup
from .token_counter import (
    create_session_used_tokens_widget,
    refresh_session_token_cache,
    debounced_token_count,
    static_update_session_used_tokens_widget
)
from .interface import request_stop

from src.core import Agent 
from src.agent.chat_session_logs import ChatSessionLogs
from src.agent.tokenizers import LocalTokenizer
from src.app.tui.screens.base_screen import BaseScreen


class ChatScreen(BaseScreen):
    CSS_PATH = "../tcss/chat.tcss"

    def __init__(self, sess_name: str | None):
        super().__init__()

        self.agent = Agent(sess_name=sess_name)
        self.local_tknizr = LocalTokenizer(model=MODEL)
        self.chat_sess_logs = ChatSessionLogs(conn=conn, sess_name=sess_name)

        self.sess_name = sess_name or "Default session"
        self.pending_key: str | None = None
        self.pending_attchmnt: list[Path] = []
        self.stop_event = threading.Event()

        self._cached_sess_tkns: int = 0
        self._awaiting_response = False


    def compose(self) -> ComposeResult:
        with Vertical():
            # Chat container:
            # - All chat message boxes (user and assistant) only
            with VerticalScroll(id="chat-container"):
                pass

            # User section
            with Vertical(id="user-section", classes="user-box"):
                # User bar widgets
                with Horizontal(id="sess-status", classes="user-bar"):
                    # Pending attachment indicator
                    yield Static(
                        "Attachments (0 pending)",
                        id="pending-attachments-counter",
                        classes="user-item"
                    )

                    # Tokens counter
                    yield Static(
                        create_session_used_tokens_widget(self),
                        id="tokens-counter",
                        classes="user-item"
                    )

                    # Button for stop model response
                    yield Button("Stop", id="stop-button", classes="hidden")

                # Prompt input bar
                yield Input(
                    placeholder="Write a message...",
                    id="prompt-input",
                    select_on_focus=False
                )

        yield from self.compose_command_bar()

    def on_mount(self) -> None:
        load_chat_history(self)
        refresh_session_token_cache(self)
        static_update_session_used_tokens_widget(self)
        self.query_one("#stop-button", Button).add_class("hidden")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "stop-button":
            request_stop(self)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "prompt-input" or self._awaiting_response:
            return

        # Live token count
        prompt = event.value.strip()
        debounced_token_count(self, prompt)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "prompt-input":
            return

        prompt = event.value.strip()
        if not prompt:
            return

        # Update widget to show token count loading
        self._awaiting_response = True
        if getattr(self, "_token_timer", None):
            self._token_timer.stop()
        self.query_one("#tokens-counter", Static).update(f"Token count loading...")

        # Update widget to show stop button widget
        self.stop_event.clear()
        stop_btn = self.query_one("#stop-button", Button)
        stop_btn.disabled = False
        stop_btn.remove_class("hidden")

        # Mount user prompt box
        user_msg_box = create_user_message_box(Static(prompt))
        chat_container = self.query_one("#chat-container", VerticalScroll)
        chat_container.mount(user_msg_box)
        chat_container.scroll_end(animate=False)
        self.query_one("#prompt-input", Input).clear()

        # Prepare Markdown widget and assistant container 
        md_widget = Markdown("")
        assist_msg_box = create_assistant_message_box(
            cont_widget=md_widget, tol_tkns=0
        )

        # Trigger worker thread
        run_fetch_agent_response(
            self,
            prompt=prompt,
            md_widget=md_widget,
            user_msg_box=user_msg_box,
            chat_container=chat_container,
            assist_msg_box=assist_msg_box
        )

    def on_key(self, event: Key) -> None:
        """
        Keybindings in normal model (not focused on anything):
        - 'k', 'j', 'h' and 'l' -> move up, down, left and right.
        - 'G' -> go to the bottom.
        - 'gg' -> go to the top.
        - 'space' then 'a' -> go to manage attachments screen.
        - 'i' or 'a' -> focuse on to the input bar.
        """
        super().on_key(event)

        # All following keybinds only works if none of them are in focus (normal mode)
        prompt_input = self.query_one("#prompt-input", Input)
        cmd_input = self.query_one("#cmd-input", Input)
        if prompt_input.has_focus or cmd_input.has_focus:
            return

        scroll_bar = self.query_one("#chat-container", VerticalScroll)

        # Set pending key 'g'
        if event.character == "g":
            # Set as first pending key
            if self.pending_key == None:
                self.pending_key = "g"

            # 'g' then 'g'
            elif self.pending_key == "g":
                scroll_bar.scroll_home(animate=False)
                self.pending_key = None

        # Set pending key 'space'
        elif event.key == "space":
            # Set as first pending key
            if self.pending_key == None:
                self.pending_key = "space"

        elif event.character == "a":
            # 'space' then 'a'
            if self.pending_key == "space":
                to_attachments_popup(self)
                self.pending_key = None

            # Single key
            if self.pending_key == None:
                prompt_input.focus()
                self.pending_key = None

        elif event.key == "k":
            # Single key
            if self.pending_key == None:
                scroll_bar.scroll_up()
                self.pending_key = None

        elif event.key == "j":
            # Single key
            if self.pending_key == None:
                scroll_bar.scroll_down()
                self.pending_key = None

        elif event.character == "G":
            # Single key
            if self.pending_key == None:
                scroll_bar.scroll_end(animate=False)
                self.pending_key = None

        elif event.key == "i":
            # Single key
            if self.pending_key == None:
                prompt_input.focus()
                self.pending_key = None

        # if event.key == "escape" and self._awaiting_response:
        #     self._request_stop()

        else:
            self.pending_key = None
