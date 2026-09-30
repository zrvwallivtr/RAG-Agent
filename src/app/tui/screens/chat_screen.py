import shlex
from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.widgets import Static, Input, Markdown, Button
from textual.events import Key

from src.config import models, postgres

from src.core import Agent 
from src.agent.chat_logs import ChatLogs
from src.agent.tokenizers import Tknizr
from src.app.tui.screens.base_screen import BaseScreen
from src.app.tui.screens.manage_attachments_screen import ManageAttachmentsScreen


class ChatScreen(BaseScreen):
    CSS_PATH = "tcss/chat.tcss"


    def __init__(self, sess_name: str | None):
        super().__init__()
        self.agent = Agent(sess_name=sess_name)
        self.tknizr = Tknizr(model=models.MODEL)
        self.chat_logs = ChatLogs(conn=postgres.conn, sess_name=sess_name)

        self.sess_name = sess_name or "Default session"
        self.pending_key: str | None = None
        self.pending_attchmnt: list[Path] = []

        self._cached_sess_tkns: int = 0


    def compose(self) -> ComposeResult:
        """
        Chat interface, with chat container (contains all the chat messages)
        and user section (contains session status, attachment indicator and
        tokens counter).
        """
        with Vertical():

            with VerticalScroll(id="chat-container"):
                pass

            with Vertical(id="user-section", classes="user-box"):

                with Horizontal(id="sess-status", classes="user-bar"):
                    yield Static(
                        "0 pending attachments",
                        id="pending-attachments-counter",
                        classes="user-item"
                    )
                    yield Static(
                        self._session_used_tokens_status(),
                        id="tokens-counter",
                        classes="user-item"
                    )

                yield Input(
                    placeholder="Write a message...", id="prompt-input", select_on_focus=False
                )

        yield from self.compose_command_bar()


    def on_mount(self) -> None:
        """Load entire history into the chat container on start up."""
        self._load_chat_history()
        self._refresh_session_token_cache()


    def on_input_changed(self, event: Input.Changed) -> None:
        """Enter prompts into the prompt input bar."""
        if event.input.id != "prompt-input":
            return

        prompt = event.value.strip()
        self._debounced_token_count(prompt)


    def on_input_submitted(self, event: Input.Submitted) -> None:
        """
        Once the prompt is submitted, the application displays
        user prompt and assistant response.
        """
        if event.input.id != "prompt-input":
            return

        prompt = event.value.strip()
        if not prompt:
            return

        chat_container = self.query_one("#chat-container", VerticalScroll)
        chat_container.mount(
            self._create_user_message_box(Static(prompt))
        )
        chat_container.scroll_end(animate=False)

        self.query_one("#prompt-input", Input).clear()
        self._fetch_agent_response(prompt)


    def on_key(self, event: Key) -> None:
        """
        Keybindings in normal model (not focused on anything):
        - 'k', 'j', 'h' and 'l' -> move up, down, left and right.
        - 'G' -> go to the bottom.
        - 'gg' -> go to the top.
        - 'space' then 'a' -> go to manage attachments screen.
        """
        super().on_key(event)
        prompt_input = self.query_one("#prompt-input", Input)
        cmd_input = self.query_one("#cmd-input", Input)

        if prompt_input.has_focus or cmd_input.has_focus:
            return

        scroll_bar = self.query_one("#chat-container", VerticalScroll)

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

        elif event.key == "space":
            self.pending_key = "space"

        elif event.character == "a":
            if self.pending_key == "space":
                self._to_manage_attachments_screen()
                self.pending_key = None

        else:
            self.pending_key = None


    def _load_chat_history(self) -> None:
        """Get entire chat history from the database and display it on to the interface."""
        chat_container = self.query_one("#chat-container", VerticalScroll)

        chat_hist = self.agent.chat_logs.get_chat_history(filter="all")
        if not chat_hist:
            return

        for msg in chat_hist:
            role = msg.get("role")
            cont = msg.get("content", "")
            p_tkns = msg.get("prompt_tokens") or 0
            o_tkns = msg.get("output_tokens") or 0
            tol_tkns = p_tkns + o_tkns

            if role == "user":
                msg_box = self._create_user_message_box(cont_widget=Static(cont))
                chat_container.mount(msg_box)

            elif role == "assistant":
                msg_box, _ = self._create_assistant_message_box(cont_widget=Markdown(cont), tol_tkns=tol_tkns)
                chat_container.mount(msg_box)

        chat_container.scroll_end(animate=False)


    def _create_user_message_box(
        self,
        cont_widget: Static | Markdown,
    ) -> Vertical:
        """Contains the content sent from the user."""
        return Vertical(
            cont_widget,
            classes=f"msg-box user-box"
        )


    def _create_assistant_message_box(
        self,
        cont_widget: Static | Markdown,
        tol_tkns: int
    ) -> tuple[Vertical, Static]:
        """Contains the assistant's response and token count for said response."""
        tkn_widget = Static(f"{tol_tkns} token used", classes="msg-token-count")

        box = Vertical(
            cont_widget,
            tkn_widget,
            classes=f"msg-box assistant-box"
        )
        return box, tkn_widget


    def _session_used_tokens_status(self) -> str | None:
        """
        Returns the string displaying the number of tokens used throughout
        the session against the max tokens limit of the active chat model.
        """
        if not self.tknizr.model_max_tkns:
            return

        used_tkns = self.tknizr.count_history_tokens(self.chat_logs.get_actv_convs())
        if not used_tkns:
            return

        return f"{used_tkns}/{self.tknizr.model_max_tkns} tokens"

    
    def _refresh_session_token_cache(self) -> None:
        """
        Refresh the value for tokens used throughout session.
        Call ONLY WHEN CHAT HISTORY CHANGES (e.g. on mount,
        after sending a message).
        """
        if not self.tknizr.model_max_tkns:
            self._cached_sess_tkns = 0
            return
        self._cached_sess_tkns = self.tknizr.count_history_tokens(self.chat_logs.get_actv_convs()) or 0


    def _debounced_token_count(self, prompt: str) -> None:
        """
        Reset a short timer on every keystroke; only the
        LAST keystroke in a burst actually trigger 
        tokenization (so rapid typing does not run it dozens
        of times per second).
        """
        if hasattr(self, "_token_timer") and self._token_timer:
            self._token_timer.stop()
        self._token_timer = self.set_timer(0.15, lambda: self._render_token_count_widget(prompt))


    @work(exclusive=True, thread=True)
    def _render_token_count_widget(self, prompt: str):
        if not self.tknizr.model_max_tkns:
            return

        pending_tkns = self.tknizr.count_string_tokens(text=prompt) or 0
        curr_tkns = self._cached_sess_tkns + pending_tkns

        self.app.call_from_thread(
            self.query_one("#tokens-counter", Static).update,
            f"{curr_tkns}/{self.tknizr.model_max_tkns} tokens"
        )


    def _add_pending_attachments(self, paths: list[Path]) -> None:
        """
        Appends paths to the pending attachment list, update the
        number of attachments in the attachment indicator widget
        in the user section and show command message for action
        feedback."""
        self.pending_attchmnt.extend(paths)
        self.query_one("#pending-attachments-counter", Static).update(
            f"{len(self.pending_attchmnt)} pending attachment(s)"
        )
        self.show_command_message(
            f"{len(paths)} file(s) attached. "
            f"{len(self.pending_attchmnt)} file(s) in total"
        )


    def _reset_pending_attachment_status(self) -> None:
        """
        Clear pending attachment path list, reset attachment
        count in the attachment indicator widget, refresh session
        used token cache value, and update tokens counter in the
        user section.
        ONLY call this when user has sent the prompt to the agent.
        """
        self.pending_attchmnt = []
        self.query_one("#pending-attachments-counter", Static).update(
            f"{len(self.pending_attchmnt)} pending attachment(s)"
        )
        self.app.call_from_thread(self._refresh_session_token_cache())
        self.app.call_from_thread(
            self.query_one("#tokens-counter", Static).update,
            f"{self._cached_sess_tkns}/{self.tknizr.model_max_tkns} tokens"
        )


    def filter_and_add_pending(self, cmd: str, prefix: str) -> None:
        """
        Retrieve all provided path(s) in the user input command;
        display message for: zero given file path, invalid syntax,
        not found path. Add all found files to the pending 
        attachment list.
        """
        paths = cmd[len(prefix):].strip()

        if not paths:
            self.show_command_message("No file path(s) provided")
            return

        try:
            all_paths = shlex.split(paths)
        except ValueError as e:
            self.show_command_message(f"Invalid path syntax: {e}")
            return

        attchmnts = []
        missing = []

        # Filter out the non-existing paths
        for p in all_paths:
            path = Path(p).expanduser()
            if path.exists():
                attchmnts.append(path)
            else:
                missing.append(p)

        if missing:
            self.show_command_message(f"{len(missing)} file(s) not found: {', '.join(missing)}")
            return

        # Only add valid paths to pending
        self._add_pending_attachments(attchmnts)


    def _to_manage_attachments_screen(self) -> None:
        def _on_manage_attachments_close(updated_paths: list[Path] | None) -> None:
            if updated_paths is not None:
                self.pending_attchmnt = updated_paths
                self.query_one("#pending-attachments-counter", Static).update(
                    f"{len(self.pending_attchmnt)} pending attachment(s)"
                )

        self.app.push_screen(
            ManageAttachmentsScreen(self.pending_attchmnt),
            _on_manage_attachments_close
        )
        return


    @work(exclusive=True, thread=True)
    def _fetch_agent_response(self, prompt: str) -> None:
        """
        Stream agent response in Markdown format in the newly created
        assistant message box; update token count and reset pending
        status when done. Call ONCE when the user submits the prompt.
        """
        chat_container = self.query_one("#chat-container", VerticalScroll)
        md_widget = Markdown("")

        assistant_box, tkn_widget = self._create_assistant_message_box(cont_widget=md_widget, tol_tkns=0)
        tkn_widget.add_class("hidden")

        self.app.call_from_thread(chat_container.mount, assistant_box)

        full_txt = ""

        # Token streaming
        def on_token(tkn: str) -> None:
            nonlocal full_txt
            full_txt += tkn
            self.app.call_from_thread(md_widget.update, full_txt)
            self.app.call_from_thread(chat_container.scroll_end, animate=False)

        attchmnts = self.pending_attchmnt or None
        result = self.agent.ask(
            prompt=prompt, callback=on_token, is_attchmnt=bool(attchmnts), paths=attchmnts
        )

        # Update number of tokens used from current response,
        # reveal widget agent finished responding.
        if result:
            _, p_tkns, o_tkns = result
            msg_tol_tkns = p_tkns + o_tkns

            def _reveal_token_count() -> None:
                tkn_widget.update(f"{msg_tol_tkns} tokens used")
                tkn_widget.remove_class("hidden")

            self.app.call_from_thread(_reveal_token_count)

        # Reset pending attachment status
        self._reset_pending_attachment_status()
