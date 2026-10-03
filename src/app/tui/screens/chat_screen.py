import shlex
from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.widgets import Static, Input, Markdown, Button
from textual.events import Key

from src.config import models, postgres, files_and_directories

from src.core import Agent 
from src.agent.chat_logs import ChatLogs
from src.agent.tokenizers import Tknizr
from src.app.tui.screens.base_screen import BaseScreen
from src.app.tui.screens.attachments_popup_screen import AttachmentsPopupScreen

from src.app.tui.screens.tui_helpers import size_bytes_helpers, agent_response_helpers


UPLOAD_DIR = Path(files_and_directories.UPLOAD_DIR).expanduser()


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
                        "Attachments (0 pending)", id="pending-attachments-counter", classes="user-item"
                    )

                    # Tokens counter
                    yield Static(
                        self._create_session_used_tokens_widget(), id="tokens-counter", classes="user-item"
                    )

                # Prompt input bar
                yield Input(placeholder="Write a message...", id="prompt-input", select_on_focus=False)

        yield from self.compose_command_bar()


    def on_mount(self) -> None:
        """Load entire history into the chat container on start up."""
        self._load_chat_history()
        self._refresh_session_token_cache()


    def on_input_changed(self, event: Input.Changed) -> None:
        """Contains actions only for content change in prompt input bar."""
        if event.input.id != "prompt-input" or self._awaiting_response:
            return

        # Live token count
        self._debounced_token_count(event.value.strip())


    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Contains actions only for prompt submitted by user."""
        if event.input.id != "prompt-input":
            return

        prompt = event.value.strip()
        if not prompt:
            return

        # Update widget to show token count loading
        self._awaiting_response = True
        if getattr(self, "_token_timer", None):
            self._token_timer.stop()
        self.query_one("#tokens-counter", Static).update(f"Loading...")

        # Create new user message box with new prompt
        user_msg_box = self._create_user_message_box(Static(prompt))

        # Show new message box in chat-container immediately after sent
        chat_container = self.query_one("#chat-container", VerticalScroll)
        chat_container.mount(user_msg_box)
        chat_container.scroll_end(animate=False)
        self.query_one("#prompt-input", Input).clear()

        md_widget = Markdown("")
        assist_msg_box = self._create_assistant_message_box(cont_widget=md_widget, tol_tkns=0)

        # Trigger worker thread
        self._run_fetch_agent_response(
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
                self._to_attachments_popup()
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

        else:
            self.pending_key = None


    # The following functions are for drawing the chat container interface, 
    # consisting:
    #
    # 1. User message box - contains the prompt submitted by the
    #    user and attachment widget(s) showing the filename and size.
    #
    # 2. Assistant message box - contains the model response message
    #    and tokens used by the model in that conversation turn.
    #
    # 3. Chat history - Draws the entire chat container iteratively
    #    for every conversation turn(s) throughout the chat history.

    def _create_user_message_box(
        self,
        cont_widget: Static | Markdown,
        attchmnt_metadata: dict | None = None
    ) -> Vertical:
        """
        Contains the content sent from the user, show widget
        if attachments was attached in that conversation turn.
        """
        widgets = [cont_widget]

        if attchmnt_metadata:
            for filename, metadata in attchmnt_metadata.items():
                size = metadata.get("size_bytes")
                fmt_size = size_bytes_helpers.format_size_bytes(size)
                widgets.append(Static(f"{filename} ({fmt_size})", classes="user-attachments"))

        return Vertical(*widgets, classes=f"msg-box user-box")


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

            attchmnt_metadata = msg.get("attachments", {}) or None

            if role == "user":
                msg_box = self._create_user_message_box(
                    cont_widget=Static(cont), attchmnt_metadata=attchmnt_metadata
                )
                chat_container.mount(msg_box)

            elif role == "assistant":
                msg_box, _ = self._create_assistant_message_box(cont_widget=Markdown(cont), tol_tkns=tol_tkns)
                chat_container.mount(msg_box)

        chat_container.scroll_end(animate=False)


    # The following functions dictates the behaviour of the pending
    # attachment widget in the user section. It indicates the number
    # of attachments that is about to be sent to the model when the
    # prompt is submitted.

    def _add_pending_attachments(self, paths: list[Path]) -> None:
        """
        Appends paths to the pending attachment list, update the
        number of attachments in the attachment indicator widget
        in the user section and show command message for action
        feedback."""
        self.pending_attchmnt.extend(paths)
        self.query_one("#pending-attachments-counter", Static).update(
            f"Attachment ({len(self.pending_attchmnt)} pending)"
        )
        self.show_command_message(
            f"{len(paths)} file(s) attached. "
            f"{len(self.pending_attchmnt)} file(s) in total"
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
            path = (UPLOAD_DIR / Path(p)).expanduser()
            if path.exists():
                attchmnts.append(path)
            else:
                missing.append(p)

        if missing:
            self.show_command_message(f"{len(missing)} file(s) not found: {', '.join(missing)}")
            return

        # Only add valid paths to pending
        self._add_pending_attachments(attchmnts)


    def _reset_pending_attachment_widget(self) -> None:
        """
        Clear pending attachment path list, reset attachment
        count in the attachment indicator widget.

        ONLY call this when there is an update on the pending attachment count:
        - User has sent the prompt to the agent.
        - User has deleted attachment(s).
        """
        self.pending_attchmnt = []
        self.query_one("#pending-attachments-counter", Static).update(
            f"Attachments ({len(self.pending_attchmnt)} pending)"
        )


    # The following controls how the chat screen behaves when the attachment popup
    # screen is trigger or closed.

    def _to_attachments_popup(self) -> None:
        def _on_manage_attachments_close(updated_paths: list[Path] | None) -> None:
            if updated_paths is not None:
                self.pending_attchmnt = updated_paths
                self.query_one("#pending-attachments-counter", Static).update(
                    f"Attachments ({len(self.pending_attchmnt)} pending)"
                )

        self.app.push_screen(
            AttachmentsPopupScreen(self.pending_attchmnt),
            _on_manage_attachments_close
        )
        return


    # The following functions describes how the token counter in the
    # user section behaves. It features a live token count for showing
    # the how many tokens will be in the current session if the prompt
    # is submitted (session used tokens + prompt tokens).

    def _create_session_used_tokens_widget(self) -> str | None:
        """
        Returns the string displaying the number of tokens used throughout
        the session against the max tokens limit of the active chat model.
        """
        if not self.tknizr.model_max_tkns:
            return

        # Used tokens throughout the entire session chat history
        used_tkns = self.tknizr.count_history_tokens(self.chat_logs.get_active_conversations())
        if not used_tkns:
            return

        return f"{used_tkns}/{self.tknizr.model_max_tkns} tokens" # Widget string

    
    def _refresh_session_token_cache(self) -> None:
        """
        Refresh the value for tokens used throughout session.
        Call ONLY WHEN CHAT HISTORY CHANGES:
        - On mount.
        - After sending a message.
        - Model finished responding.
        """
        result = self.chat_logs.get_session_data()
        if not result:
            return
        _, _, self._cached_sess_tkns = result


    @work(exclusive=True, thread=True, group="token_count")
    def _session_used_tokens_and_prompt_tokens_count(self, prompt: str):
        """
        Update of the session used tokens widget:
        - Session used tokens + current tokens count in prompt input.
        """
        if self._awaiting_response or not self.tknizr.model_max_tkns:
            return

        pending_tkns = self.tknizr.count_string_tokens(text=prompt) or 0
        curr_tkns = self._cached_sess_tkns + pending_tkns

        self.app.call_from_thread(
            self.query_one("#tokens-counter", Static).update,
            f"{curr_tkns}/{self.tknizr.model_max_tkns} tokens"
        )


    def _debounced_token_count(self, prompt: str) -> None:
        """
        Reset a short timer on every keystroke; only the LAST keystroke in
        a burst actually trigger tokenization (so rapid typing does not run
        it dozens of times per second).
        """
        if hasattr(self, "_token_timer") and self._token_timer:
            self._token_timer.stop()

        self._token_timer = self.set_timer(
            0.15, lambda: self._session_used_tokens_and_prompt_tokens_count(prompt)
        )


    def _static_update_session_used_tokens_widget(self) -> None:
        """
        Refresh session used token cache value, and update
        tokens counter widget in the user section.

        ONLY call this when:
        - Model done responding.
        """
        self._awaiting_response = False
        self._refresh_session_token_cache() # Execute cache refresh on the main thread

        # Update widget using cached value
        self.query_one("#tokens-counter", Static).update(
            f"{self._cached_sess_tkns}/{self.tknizr.model_max_tkns} tokens"
        )


    # The following controls the behaviour of the assistant message box
    # when the model is responding.

    @work(exclusive=True, thread=True)
    def _run_fetch_agent_response(
        self,
        prompt: str,
        md_widget: Markdown,
        user_msg_box: Vertical,
        chat_container: VerticalScroll,
        assist_msg_box: tuple[Vertical, Static]
    ):
        try:
            agent_response_helpers.fetch_agent_response(
                screen=self,
                prompt=prompt,
                md_widget=md_widget,
                user_msg_box=user_msg_box,
                chat_container=chat_container,
                assist_msg_box=assist_msg_box,
                pending_attchmnt=self.pending_attchmnt,
                agent=self.agent
            )

        finally: # After model done responding
            self.app.call_from_thread(self._static_update_session_used_tokens_widget)
