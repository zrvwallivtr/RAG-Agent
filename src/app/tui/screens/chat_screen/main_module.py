import threading
from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Static, Input, Markdown, Button
from textual.events import Key

from src.config.postgres import CONN
from src.config.config_toml import MODEL

from .agent_response import run_fetch_agent_response
from .chat_container import (
    clear_code_selection,
    create_user_message_box,
    create_assistant_message_box,
    load_chat_history,
    focus_current_message_box,
    jump_messages,
    goto_message,
    jump_code_blocks,
    go_to_code_block,
    scroll_code_block,
    scroll_code_block_edge,
)
from .status_bar import (
    get_location_text,
    showcmd_text,
    session_status_text,
)
from .pending_attachments import to_attachments_popup, process_add_pending_attachments
from .token_counter import (
    set_tokens_text,
    refresh_session_token_cache,
    debounced_token_count,
    static_update_session_used_tokens_widget,
)

from src.core import Agent
from src.agent.chat_session_logs import ChatSessionLogs
from src.agent.tokenizers import LocalTokenizer
from src.app.tui.screens.base_screen import BaseScreen
from src.app.tui.screens.base_screen.command_bar import show_command_message
from src.app.tui.movement_helpers import pending_key_is_digit_counter

LLM_TKNIZR = LocalTokenizer(MODEL)


class ChatScreen(BaseScreen):
    """Chat screen interface managing prompt inputs, session metrics, and chat threads."""

    CSS_PATH = ["../tcss/base.tcss", "../tcss/chat.tcss"]

    BINDINGS = [
        ("ctrl+c", "stop_agent_response"),
    ]

    def __init__(self, sess_name: str | None):
        super().__init__()

        self.agent = Agent(sess_name=sess_name)
        self.local_tknizr = LocalTokenizer(model=MODEL)
        self.chat_sess_logs = ChatSessionLogs(conn=CONN, sess_name=sess_name, llm_tknizr=LLM_TKNIZR)

        self.sess_name = sess_name or "Default session"
        self.pending_key: str | None = None
        self.tokens_text: str = ""
        self.pending_attchmnt: list[Path] = []
        self.stop_event = threading.Event()

        self._cached_sess_tkns: int = 0
        self._awaiting_response = False

        self.focused_msg_box = None
        self.code_idx: int | None = None

        # Modes
        self.in_insert_mode = False  # Insert mode
        self.in_msg_mode = False  # Message mode

    def compose(self) -> ComposeResult:
        with Vertical():
            # Chat container:
            # - All chat message boxes (user and assistant) only
            with VerticalScroll(id="chat-container"):
                pass

            # User section
            with Vertical(id="user-section", classes="user-box"):
                # User bar widgets
                with Horizontal(id="prompt-row"):
                    yield Static(">", id="prompt-prefix")

                    # Prompt input bar
                    yield Input(
                        placeholder="Write a message...",
                        id="prompt-input",
                        select_on_focus=False,
                        compact=True,
                    )

                    # Button for stop model response
                    yield Button(
                        "Stop",
                        id="stop-button",
                        classes="hidden",
                        compact=True,
                    )

        yield from self.compose_command_bar()

    def on_mount(self) -> None:
        """
        Initialises chat history, token cache counters,
        and button visibility.
        """
        load_chat_history(self)
        refresh_session_token_cache(self)
        static_update_session_used_tokens_widget(self)
        self.query_one("#stop-button", Button).add_class("hidden")

        chat = self.query_one("#chat-container", VerticalScroll)
        self.watch(chat, "scroll_y", self._on_chat_scrolled, init=False)
        self.refresh_status()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "stop-button":
            self.request_stop()

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
        set_tokens_text(self, "Estimating tokens used...")

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
        assist_msg_box = create_assistant_message_box(cont_widget=md_widget, tol_tkns=0)

        # Trigger worker thread
        run_fetch_agent_response(
            self,
            prompt=prompt,
            md_widget=md_widget,
            user_msg_box=user_msg_box,
            chat_container=chat_container,
            assist_msg_box=assist_msg_box,
        )

    def on_key(self, event: Key) -> None:
        super().on_key(event)

        # All following keybinds only works if none of them are in focus (normal mode)
        prompt_input = self.query_one("#prompt-input", Input)
        cmd_input = self.query_one("#cmd-input", Input)

        if prompt_input.has_focus and event.key == "escape":
            self.in_insert_mode = False
            self.set_focus(None)
            return

        if prompt_input.has_focus or cmd_input.has_focus:
            return

        scroll_bar = self.query_one("#chat-container", VerticalScroll)

        if event.character and event.character.isdigit():
            if event.character == "0" and not (
                self.pending_key and self.pending_key.isdigit()
            ):
                if self.in_msg_mode:
                    scroll_code_block_edge(self, "start")
                self.pending_key = None
                return

            if self.pending_key is None or not self.pending_key.isdigit():
                self.pending_key = event.character  # Set first digit
            else:
                self.pending_key += event.character  # Regular pending key
            return

        # Leader key
        if event.key == "space":
            # <space>
            if self.pending_key == None:
                self.pending_key = "space"

            else:
                self.pending_key = None

        elif event.character == "g":
            # <g>
            if self.pending_key is None:
                self.pending_key = "g"

            # <gg>
            elif self.pending_key == "g":
                if self.in_msg_mode:
                    # In message mode
                    go_to_code_block(self, 1)

                else:
                    # In normal mode
                    scroll_bar.scroll_home(animate=False)

                self.pending_key = None
            else:
                self.pending_key = None

        elif event.character == "a":
            # <a>
            if self.pending_key is None:
                self.in_insert_mode = True
                prompt_input.focus()

            # <space>a
            elif self.pending_key == "space":
                # In all modes
                to_attachments_popup(self)
                self.pending_key = None

            self.pending_key = None

        elif event.key in ("k", "j"):
            direction = "up" if event.key == "k" else "down"
            count = pending_key_is_digit_counter(self)

            # <k> or <j>
            if self.pending_key is None:
                if self.in_msg_mode:
                    jump_code_blocks(self, direction, 1)
                else:
                    if direction == "up":
                        scroll_bar.scroll_up()
                    else:
                        scroll_bar.scroll_down()

            # <n>k or <n>j
            elif count is not None:
                # In message mode
                if self.in_msg_mode:
                    jump_code_blocks(self, direction, count)
                # In normal mode
                else:
                    jump_messages(self, direction, count=count)

            self.pending_key = None

        elif event.key in ("h", "l"):
            direction = "left" if event.key == "h" else "right"
            count = pending_key_is_digit_counter(self)

            if self.pending_key is None:
                if self.in_msg_mode:
                    scroll_code_block(self, direction, 1)

            elif count is not None:
                if self.in_msg_mode:
                    scroll_code_block(self, direction, count)

            self.pending_key = None

        elif event.character == "$":
            if self.pending_key is None and self.in_msg_mode:
                scroll_code_block_edge(self, "end")

            self.pending_key = None

        elif event.character in ("K", "J"):
            direction = "up" if event.character == "K" else "down"

            # In message mode
            if self.in_msg_mode:
                jump_messages(self, direction, count=1)
            # In normal mode
            else:
                jump_messages(self, direction, count=1)

            self.pending_key = None

        elif event.character == "G":
            count = pending_key_is_digit_counter(self)

            # <G>
            if self.pending_key is None:
                # In message mode
                if self.in_msg_mode:
                    go_to_code_block(self, None)
                # In normal mode
                else:
                    scroll_bar.scroll_end(animate=False)

            # <n>G
            elif count is not None:
                # In message mode
                if self.in_msg_mode:
                    go_to_code_block(self, count)
                # In normal mode
                else:
                    goto_message(self, count)

            self.pending_key = None

        elif event.key == "i":
            # <i>
            if self.pending_key is None:
                self.in_insert_mode = True
                prompt_input.focus()

            self.pending_key = None

        elif event.key == "m":
            # <m>
            if self.pending_key is None:
                focus_current_message_box(self)

            self.pending_key = None

        elif event.key == "escape":
            # <escape>
            self.pending_key = None
            # In mesage mode
            if self.in_msg_mode:
                self.in_msg_mode = False
                self.focused_msg_box = None
                clear_code_selection(self)
                self.set_focus(None)
            # In insert mode
            if self.in_insert_mode:
                self.in_insert_mode = False
                self.set_focus(None)

        else:
            self.pending_key = None

    def run_process_add_pending_attachments(self, cmd: str, prefix: str) -> None:
        process_add_pending_attachments(self, cmd=cmd, prefix=prefix)

    def _on_chat_scrolled(self) -> None:
        self.refresh_status()

    # --- Cmd bar -------------------------------------------------------------

    def mode_text(self) -> str:
        if self.focused is self.query_one("#prompt-input", Input):
            return "-- INSERT --"
        if self.in_msg_mode:
            return "-- CODE BLOCK --" if self.code_idx is not None else "-- MESSAGE --"
        return ""

    # --- Status bar ----------------------------------------------------------

    def ruler_text(self) -> str:
        return get_location_text(self)

    def showcmd_text(self) -> str:
        return showcmd_text(self)

    def status_text(self) -> str:
        return session_status_text(self)

    def refresh_status(self) -> None:
        """Redraw session status, mode, ruler and pending keys."""
        try:
            status_left = self.query_one("#status-left", Static)
            cmd_input = self.query_one("#cmd-input", Input)
            ruler = self.query_one("#ruler", Static)
            showcmd = self.query_one("#showcmd", Static)
        except NoMatches:
            return

        status_left.update(self.status_text())
        cmd_input.placeholder = self.mode_text()
        ruler.update(self.ruler_text())
        showcmd.update(self.showcmd_text())

    # --- Stop button ---------------------------------------------------------

    def request_stop(self) -> None:
        """
        Sets stop flags and disables the stop button to prevent double-click
        race conditions.

        Only call when agent is generating/streaming response.
        """
        if not self._awaiting_response:
            return
        self.stop_event.set()
        self.query_one("#stop-button", Button).disabled = True  # No double clicks
        show_command_message(self, "Stopping agent response...")

    def action_stop_agent_response(self) -> None:
        prompt_input = self.query_one("#prompt-input", Input)

        if prompt_input.has_focus:
            self.request_stop()
