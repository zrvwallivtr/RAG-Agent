from difflib import get_close_matches
from typing import Literal
from datetime import datetime
from psycopg2.errors import UniqueViolation

from rich import box
from rich.panel import Panel
from rich.align import Align 

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Static, Input, OptionList, Button
from textual.widgets.option_list import Option
from textual.events import Key

from src.config import models, postgres
from src.agent.models import ollama
from src.agent import tokenizers
from src.app.tui.screens.base_screen import BaseScreen
from assests.icons import app_icon_ascii


MODEL_ROLES = ["chat_model", "memory_model", "web_search_model", "embedding_model"]


def _get_tknizr_list() -> list | None:
    """Return all installed tokenizers as a list."""
    tknizr_dict = tokenizers.fetch_all_installed_tokenizers()
    if not tknizr_dict:
        return

    tknizr_list = []

    for tknizr in tknizr_dict:
        tknizr_list.append(tknizr["name"])
    return tknizr_list


def _get_session_list() -> list | None:
    """Return all existing session as a list."""
    from src.agent.chat_logs import ChatLogs
    chat_logs = ChatLogs(conn=postgres.conn)

    sess_dict = chat_logs.get_all_existing_sess_metadata()
    if not sess_dict:
        return

    sess_list = []

    for sess in sess_dict:
        sess_list.append(sess_dict[sess]["session_name"])
    return sess_list


def _trimmed_date(iso_str: str | None) -> str:
    """
    Trim the date format from 'ISO 8601 with microseconds'
    to 'DD/MM/YYYY HH:MM'
    """
    if not iso_str: # Fallback
        return "    ---         "

    dt = datetime.fromisoformat(iso_str.split("+")[0])
    return dt.strftime("%d/%m/%Y %H:%M")
   

def _get_session_with_dates(sess_dict: dict, sess_id: str) -> str | None:
    """Return a string containing session last modified at, created at and session name."""
    from src.agent.chat_logs import ChatLogs
    chat_logs = ChatLogs(conn=postgres.conn)

    sess_name = sess_dict[sess_id]["session_name"]
    created_at = _trimmed_date(iso_str=sess_dict[sess_id]["created_at"])

    last_mod_iso = chat_logs.get_session_last_modified_time(sess_id=sess_id)
    if not last_mod_iso:
        last_mod_iso = sess_dict[sess_id]["created_at"]

    last_mod = _trimmed_date(
        iso_str=last_mod_iso
    )
    sep = " "

    return (
        f"[magenta]{last_mod}[/]{sep}\t"
        f"[green]{created_at}[/]{sep}\t"
        f"[yellow]{sess_name}[/]"
    )


def app_icon() -> Panel:
    """Return app icon as Panel."""
    return Panel(
        Align.center(f"[bright_cyan bold not italic]{app_icon_ascii.RAG}", vertical="middle"),
        height=13,
        width=26,
        box=box.SQUARE
    )


class DashboardScreen(BaseScreen):
    CSS_PATH = "tcss/dashboard.tcss"


    def __init__(self):
        super().__init__()
        self.ava_models = ollama.ollama_models_list()
        self.selected_models = {
            MODEL_ROLES[0]: models.MODEL,
            MODEL_ROLES[1]: models.MEM_MODEL,
            MODEL_ROLES[2]: models.SEAR_MODEL,
            MODEL_ROLES[3]: models.EMBED_MODEL,
        }
        self.fb_tknizr = models.FALLBACK_TOKENIZER
        self.curr_model_role: str | None = None
        self.curr_fb_tknizr: str | None = None
        self.curr_to_sess: str | None = None

        self.in_sess_search: bool = False
        self.in_new_sess_input: bool = False
        self.in_del_sess_input: bool = False


    def compose(self) -> ComposeResult:
        """
        The top container contains:
        - App icon
        - App description

        The status container contains:
        - Models section
        - Tokenizers section
        - Sessions section
        """
        with Horizontal(id="top-container"):
            yield Static(id="icon", classes="box")
            yield Static(id="status-container", classes="box")

        with Vertical(id="sessions", classes="box"):

            # Sessions title
            yield Static(id="sess-section-title")

            # Sessions actions
            with Horizontal(id="sess-actions", classes="action-bar"):
                yield Button("Search session", id="sear-sess", classes="action-item")
                yield Button("New session", id="new-sess", classes="action-item")
                yield Button("Delete session", id="del-sess", classes="action-item")

            yield Input(
                id="sess-sear-input",
                placeholder="Search session name...",
                classes="hidden",
                select_on_focus=False
            )
            yield Input(
                id="new-sess-input",
                placeholder="New session name...",
                classes="hidden",
                select_on_focus=False
            )
            yield Input(
                id="del-sess-input",
                placeholder="Delete session name...",
                classes="hidden",
                select_on_focus=False
            )

            # Sessions list
            yield OptionList(id="sess-list", classes="hidden")

        yield from self.compose_command_bar()


    def on_mount(self) -> None:
        """Show app icon, app description and status on startup."""
        self.query_one("#icon", Static).update(app_icon())
        self._show_status_container_panel()
        self._show_sessions()
        self.refresh(layout=True)


    def on_button_pressed(self, event: Button.Pressed) -> None:
        """
        Controls the behaviour when action bar button is pressed.
        """
        if event.button.id == "sear-sess":
            self._show_session_search_input()

        elif event.button.id == "new-sess":
            self._show_new_session_input()

        elif event.button.id == "del-sess":
            self._show_delete_session_input()


    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """
        Controls the behaviour when option is selected.
        """
        list_id = event.option_list.id

        if list_id == "sess-list":
            if not event.option.id:
                return

            elif self.in_sess_search or self.in_new_sess_input:
                self._go_to_session(sess_name=event.option.id)

            elif self.in_del_sess_input:
                self._delete_session(sess_name=event.option.id)

            else:
                self._go_to_session(sess_name=event.option.id)


    def on_input_changed(self, event: Input.Changed) -> None:
        """Fuzzy search for models, tokenizers and sessions."""
        super().on_input_changed(event)

        if event.input.id == "sess-sear-input":
            self._fuzzy_search_behaviour(event=event, id="#sess-list")

        if event.input.id == "new-sess-input":
            self._fuzzy_search_behaviour(event=event, id="#sess-list")

        if event.input.id == "del-sess-input":
            self._fuzzy_search_behaviour(event=event, id="#sess-list")


    def on_input_submitted(self, event: Input.Submitted) -> None:
        """
        Behaviour of the search bars for models, tokenizers and sessions
        when input is submitted.
        """
        super().on_input_submitted(event)

        # === SESSION SEARCH BAR ===========================
        if event.input.id == "sess-sear-input":

            sess_list = self.query_one("#sess-list", OptionList)
            if sess_list.highlighted is None:
                return

            selected_option = sess_list.get_option_at_index(sess_list.highlighted)
            if not selected_option.id:
                return

            self._go_to_session(sess_name=selected_option.id)

        # === NEW SESSION INPUT BAR ========================
        if event.input.id == "new-sess-input":
            self._create_new_session(sess_name=event.value)
            return

        # === DELETE SESSION INPUT BAR =====================
        if event.input.id == "del-sess-input":
            self._delete_session(sess_name=event.value)
            return


    def on_key(self, event: Key) -> None:
        """Menu and option list keybinds."""
        super().on_key(event)

        # List navigation
        if isinstance(self.focused, OptionList):
            if event.key == "k":
                event.prevent_default()
                event.stop()
                self.focused.action_cursor_up()

            elif event.key == "j":
                event.prevent_default()
                event.stop()
                self.focused.action_cursor_down()

        if event.key == "escape":
            if self.in_sess_search or self.in_new_sess_input or self.in_del_sess_input:
                event.prevent_default()
                event.stop()
                self._show_sessions()

            # Only highlight when focused
            for opt_list in self.query(OptionList):
                if opt_list.highlighted is not None:
                    opt_list.highlighted = None

        # Only highlight when focused
        if event.key == "tab":
            for opt_list in self.query(OptionList):
                if opt_list.highlighted is not None:
                    opt_list.highlighted = None


    def _show_status_container_panel(self) -> None:
        """Renders models and tokenizers info in a single Rich Panel."""

        # Models section
        model_lines = []
        label_width = max(len(role.replace("_", " ").capitalize()) for role in MODEL_ROLES) + 2
        for role in MODEL_ROLES:
            label = role.replace("_", " ").capitalize()
            padded_label = f"{label}:".ljust(label_width)
            model_lines.append(f"\t{padded_label}\t[yellow]{self.selected_models[role]}[/]")
        models_list = "\n".join(model_lines)

        # Tokenizers section
        tknizr_list = _get_tknizr_list()
        tknizr_count = len(tknizr_list) if tknizr_list else 0
        fb_tknizr = f"\tFallback tokenizer:\t[yellow]{models.FALLBACK_TOKENIZER}[/]"

        # Combined sections
        comb_cont = (
            f"[bold]Models[/] ({len(self.ava_models)} installed)\n\n"
            f"{models_list}\n\n"
            f"[bold]Tokenizers[/] ({tknizr_count} installed)\n\n"
            f"{fb_tknizr}"
        )

        panel = Panel(
            comb_cont,
            title="[bright_cyan bold not italic]STATUS[/]",
            box=box.SQUARE
        )

        self.query_one("#status-container", Static).update(panel)


    def _show_session_action_bar(self) -> None:
        """Option list for search session, new session, delete session."""
        self.in_sess_search = False # User not searching sessions
        self.curr_to_sess = None

        action_bar = self.query_one("#sess-actions")
        action_bar.remove_class("hidden")
        self.query_one("#sear-sess", Button).focus()


    def _show_session_list(self) -> None:
        """
        Show list of sessions with created time, modified time and session name.
        """
        from src.agent.chat_logs import ChatLogs
        chat_logs = ChatLogs(conn=postgres.conn)

        sess_dict = chat_logs.get_all_existing_sess_metadata()
        if not sess_dict:
            return

        sess_list = self.query_one("#sess-list", OptionList)
        sess_list.clear_options()

        list_title = Option("Last modified\t\tCreated at\t\tSession name\n")
        list_title.disabled = True
        sess_list.add_option(list_title)

        for sess_id in sess_dict:
            sess_name = sess_dict[sess_id]["session_name"]
            sess_with_dates = _get_session_with_dates(
                sess_dict=sess_dict, sess_id=sess_id
            )
            sess_list.add_option(Option(sess_with_dates, id=sess_name))

        sess_list.remove_class("hidden")


    def _show_sessions(self) -> None:
        """
        Consist:
        - Show latest sessions. If selected go to the chat screen for said session.
        - Show go to session option. If selected, the fuzzy search session selector
          will popup.
        """
        self.in_sess_search = False
        self.curr_to_sess = None

        from src.agent.chat_logs import ChatLogs
        chat_logs = ChatLogs(conn=postgres.conn)

        sess_dict = chat_logs.get_all_existing_sess_metadata()

        # Section title
        title = (
            f"[bold]Sessions[/bold] ({len(sess_dict)} created)"
            if sess_dict
            else f"[bold]Sessions[/bold] (0 created)\n"
        )
        self.query_one("#sess-section-title", Static).update(title)

        # Session selector
        self._show_session_action_bar()
        self._show_session_list()

        # Hide widgets
        self.query_one("#sess-sear-input", Input).add_class("hidden")
        self.query_one("#new-sess-input", Input).add_class("hidden")
        self.query_one("#del-sess-input", Input).add_class("hidden")


    def _show_session_search_input(self) -> None:
        """
        Consist of the search input bar and available sessions list.
        Session list will be updated according to the search bar.
        """
        self.in_sess_search = True # User is searching sessions

        # Session option list
        sess_list = self.query_one("#sess-list", OptionList)

        # Highlight first match
        if sess_list.option_count > 1:
            sess_list.highlighted = 1

        # Fuzzy search
        search_input = self.query_one("#sess-sear-input", Input)
        search_input.value = ""
        search_input.remove_class("hidden")
        sess_list.remove_class("hidden")
        search_input.focus()

        # Hide session search input trigger
        self.query_one("#sess-actions").add_class("hidden")


    def _go_to_session(self, sess_name: str) -> None:
        """Go to specified chat session screen."""
        self.notify(f"Opening session: {sess_name}")
        from src.app.tui.screens.chat_screen import ChatScreen
        self.app.push_screen(ChatScreen(sess_name=sess_name))


    def _show_new_session_input(self) -> None:
        """
        Consist of the input bar for naming the new session.
        Show existing session list with fuzzy search on.
        """
        self.in_new_sess_input = True # User is entering session name
        self._show_session_list()

        # Existing session list
        sess_list = self.query_one("#sess-list", OptionList)

        # Highlight first match
        if sess_list.option_count > 1:
            sess_list.highlighted = 1

        # Input bar
        new_sess_input = self.query_one("#new-sess-input", Input)
        new_sess_input.value = ""
        new_sess_input.remove_class("hidden")
        sess_list.remove_class("hidden")
        new_sess_input.focus()

        # Hide widgets
        self.query_one("#sess-actions").add_class("hidden")


    def _create_new_session(self, sess_name: str) -> None:
        """Create new session."""
        clean_name = sess_name.strip()
        if not clean_name:
            self.notify("Session name cannot be empty.", severity="error")
            return

        from src.agent.chat_logs import ChatLogs
        chat_logs = ChatLogs(conn=postgres.conn, sess_name=clean_name)

        try:
            chat_logs.create_sess()

        except UniqueViolation:
            chat_logs.conn.rollback()
            self.notify(f"Session '{clean_name}' already exists", severity="error")
            return

        self.notify(f"New session created: {clean_name}")
        self._go_to_session(clean_name)


    def _show_delete_session_input(self) -> None:
        """
        Consist of the input bar for available sessions.
        Show existing session list with fuzzy search on.
        """
        self.in_del_sess_input = True # User is entering session name

        # Existing session list
        sess_list = self.query_one("#sess-list", OptionList)

        # Input bar
        del_sess_input = self.query_one("#del-sess-input", Input)
        del_sess_input.value = ""
        del_sess_input.remove_class("hidden")
        sess_list.remove_class("hidden")
        del_sess_input.focus()

        # Hide widgets
        self.query_one("#sess-actions").add_class("hidden")


    def _delete_session(self, sess_name: str) -> None:
        """Delete session."""
        clean_name = sess_name.strip()
        if not clean_name:
            self.notify("Session name cannot be empty.", severity="error")
            return

        from src.app.operations import sessions
        result = sessions.del_sess(sess_name=clean_name)

        if result and "fail" in result.lower():
            self.notify(result, severity="error")
            return

        self.notify(f"Session deleted: {sess_name}")
        self._show_sessions()


    def _fuzzy_search_behaviour(
        self,
        event: Input.Changed,
        id: Literal["#sess-list"],
    ) -> None:
        """
        Filters the OptionList based on fuzzy matching.
        Controls:
        - The behaviour of the model list when the
          search input is changed.
        """
        from src.agent.chat_logs import ChatLogs
        chat_logs = ChatLogs(conn=postgres.conn)

        sess_dict = chat_logs.get_all_existing_sess_metadata()
        if not sess_dict:
            return

        sess_list = _get_session_list()
        if not sess_list:
            return

        qry = event.value.strip().lower()
        opt_list = self.query_one(id, OptionList)
        opt_list.clear_options()

        list_title = Option("Last modified\t\tCreated at\t\tSession name\n")
        list_title.disabled = True
        opt_list.add_option(list_title)

        if not qry:
            # Show entire list if query is empty
            self._show_session_list()

        else:
            # Create an option for every matches
            exact_matches = [entry for entry in sess_list if qry in entry.lower()]
            fuzzy_matches = get_close_matches(
                qry,
                [entry.lower() for entry in sess_list],
                n=5,
                cutoff=0.4
            )

            name_list = []
            for name in sess_list:
                if name in exact_matches or name.lower() in fuzzy_matches:
                    name_list.append(name)

            for name in name_list:
                sess_id = chat_logs.get_sess_id_from_name(sess_name=name)
                if not sess_id:
                    continue

                if not sess_dict:
                    return

                sess_with_dates = _get_session_with_dates(
                    sess_dict=sess_dict, sess_id=sess_id
                )
                opt_list.add_option(Option(sess_with_dates, id=name))

        # Re-highlight first match after every update
        if opt_list.option_count > 1:
            opt_list.highlighted = 1
