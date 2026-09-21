from difflib import get_close_matches
from typing import Literal
from datetime import datetime
#import pytz

from rich import box
from rich.panel import Panel

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Static, Input, OptionList
from textual.widgets.option_list import Option
from textual.events import Key

from src.config import models, postgres
from src.agent.chat_logs import ChatLogs
from src.agent.models import ollama
from src.agent import tokenizers
from src.tui.screens.base_screen import BaseScreen
from assests.icons import app_icon_ascii


chat_logs = ChatLogs(conn=postgres.conn)

MODEL_ROLES = ["chat_model", "memory_model", "web_search_model", "embedding_model"]

TKNIZR_DICT = tokenizers.fetch_all_installed_tokenizers()
TKNIZR_LIST = (
    [tknizr["name"] for tknizr in TKNIZR_DICT]
    if TKNIZR_DICT
    else None
)

def _trimmed_date(iso_str: str | None) -> str:
    """
    Trim the date format from 'ISO 8601 with microseconds'
    to 'DD/MM/YYYY HH:MM'
    """
    if not iso_str:
        return "    ---         "
        return "%d/%m/%Y %H:%M"

    dt = datetime.fromisoformat(iso_str.split("+")[0])
    return dt.strftime("%d/%m/%Y %H:%M")
    

SESS_DICT = chat_logs.get_all_existing_sess_metadata()
SESS_LIST = (
    [SESS_DICT[sess]["session_name"] for sess in SESS_DICT]
    if SESS_DICT
    else None
)


def _get_session_with_dates(sess_dict: dict, sess_id: str) -> str | None:
    """Return a string containing session last modified at, created at and session name."""
    sess_name = sess_dict[sess_id]["session_name"]
    created_at = _trimmed_date(iso_str=sess_dict[sess_id]["created_at"])
    last_mod = _trimmed_date(
        iso_str=chat_logs.get_session_last_modified_time(sess_id=sess_id)
    )
    sep = " "

    return (
        f"Last modified: [magenta]{last_mod}[/]{sep}"
        f"Created at: [green]{created_at}[/]{sep}"
        f"[yellow]{sess_name}[/]"
    )


def app_icon() -> Panel:
    return Panel(
        f"[bright_cyan bold not italic]{app_icon_ascii.RAG}",
        height=11,
        width=26,
        box=box.SQUARE
    )


def app_description() -> Panel:
    return Panel(
        (
            "A local Command-Line Interface (CLI) AI assistant featuring long-term memory, "
            "file context injection, (isolated web crawling / search and automated token management)."
        ),
        title=f"[bright_cyan bold not italic]DESCRIPTION",
        box=box.SQUARE
    )


class MenuScreen(BaseScreen):
    CSS_PATH = "tcss/menu.tcss"


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
        with Horizontal(id="top_container"):
            yield Static(id="icon", classes="box")
            yield Static(id="description", classes="box")

        with Vertical(id="status_container", classes="box"):

            # === MODEL STATUS ========================
            yield Static(id="model_section_title")
            yield Static(id="model_role_list")

            # === TOKENIZERS STATUS ===================
            yield Static(id="tknizr_section_title")
            yield Static(id="fb_tknizr")

            # === SESSIONS STATUS =====================
            yield Static(id="sess_section_title")
            yield OptionList(id="sess_sear_input_trigg")
            yield Input(
                id="sess_sear_input",
                placeholder="Search session...",
                classes="hidden",
                select_on_focus=False
            )
            yield OptionList(id="sess_list", classes="hidden")

        yield from self.compose_command_bar()


    def on_mount(self) -> None:
        """Show app icon, app description and status on startup."""
        self.query_one("#icon", Static).update(app_icon())
        self.query_one("#description", Static).update(app_description())

        self.show_model_roles()
        self.show_fallback_tokenizer()
        self.show_sessions()


    # ================================================
    # MODEL SECTION
    # ================================================

    def show_model_roles(self) -> None:
        """
        List of all roles that available for model selection,
        displaying roles and corresponding selected model.
        """
        # Section title
        title = f"[bold]Models[/] ({len(self.ava_models)} installed)"
        self.query_one("#model_section_title", Static).update(title)

        # Show model roles option list
        model_role_list = self.query_one("#model_role_list", Static)

        label_width = max(len(role.replace("_", " ").capitalize()) for role in MODEL_ROLES) + 2

        lines = []
        for role in MODEL_ROLES:
            label = role.replace("_", " ").capitalize()
            padded_label = f"{label}:".ljust(label_width)
            lines.append(f"{padded_label}\t[yellow]{self.selected_models[role]}[/]")

        cont = "\n".join(lines)
        model_role_list.update(cont)


    # ================================================
    # TOKENIZER SECTION
    # ================================================

    def show_fallback_tokenizer(self) -> None:
        """
        Show current selected fallback tokenizer for
        the models. If selected, the fuzzy search
        tokenizer selector will popup.
        """
        # Section title
        title = (
            f"[bold]Tokenizers[/bold] (0 installed)"
            if not TKNIZR_LIST
            else f"[bold]Tokenizers[/bold] ({len(TKNIZR_LIST)} installed)"
        )
        self.query_one("#tknizr_section_title", Static).update(title)

        # Show fallback tokenizer option
        fb_tknizr = self.query_one("#fb_tknizr", Static)
        fb_tknizr.update(f"Fallback tokenizer:\t[yellow]{models.FALLBACK_TOKENIZER}[/]")
        fb_tknizr.remove_class("hidden")
        fb_tknizr.focus()


    # ================================================
    # SESSION SECTION
    # ================================================

    def show_sessions(self) -> None:
        """
        Consist:
        - Show latest sessions. If selected go to the chat screen for said session.
        - Show go to session option. If selected, the fuzzy search session selector
          will popup.
        """
        self.in_sess_search = False
        self.curr_to_sess = None

        chat_logs = ChatLogs(conn=postgres.conn)
        sess_dict = chat_logs.get_all_existing_sess_metadata()

        # Section title
        title = (
            f"[bold]Sessions[/bold] ({len(sess_dict)} created)"
            if sess_dict
            else f"[bold]Sessions[/bold] (0 created)\n"
        )
        self.query_one("#sess_section_title", Static).update(title)

        # Session selector
        self.show_session_search_input_trigger()
        self.show_session_list()

        # Hide session fuzzy search bar
        self.query_one("#sess_sear_input", Input).add_class("hidden")


    def show_session_search_input_trigger(self) -> None:
        """Button when press triggers the show session search bar."""
        self.in_sess_search = False # User not searching sessions
        self.curr_to_sess = None

        # Sesion search bar trigger
        trigg = self.query_one("#sess_sear_input_trigg", OptionList)
        trigg.clear_options()
        trigg.add_option(Option("Search session"))
        trigg.remove_class("hidden")
        trigg.focus()

    def show_session_list(self) -> None:
        """
        Show list of sessions with created time, modified time and session name.
        """
        if not SESS_DICT:
            return

        sess_list = self.query_one("#sess_list", OptionList)
        sess_list.clear_options()

        for sess_id in SESS_DICT:
            sess_name = SESS_DICT[sess_id]["session_name"]
            sess_with_dates = _get_session_with_dates(
                sess_dict=SESS_DICT, sess_id=sess_id
            )
            sess_list.add_option(Option(sess_with_dates, id=sess_name))

        sess_list.remove_class("hidden")


    def show_session_search_input(self, sess: str) -> None:
        """
        Consist of the search input bar and available sessions list.
        Session list will be updated according to the search bar.
        """
        self.in_sess_search = True # User is searching sessions
        self.curr_to_sess = sess

        # Session option list
        sess_list = self.query_one("#sess_list", OptionList)

        # Highlight first match
        if sess_list.option_count > 0:
            sess_list.highlighted = 0

        # Fuzzy search
        search_input = self.query_one("#sess_sear_input", Input)
        search_input.value = ""
        search_input.remove_class("hidden")
        sess_list.remove_class("hidden")
        search_input.focus()

        # Hide session search input trigger
        self.query_one("#sess_sear_input_trigg", OptionList).add_class("hidden")


    def _go_to_session(self, sess_name: str) -> None:
        """Go to specified chat session screen."""
        self.notify(f"Opening session: {sess_name}")
        from src.tui.screens.chat_screen import ChatScreen
        self.app.push_screen(ChatScreen(sess_name=sess_name))


    # ================================================
    # OPTION LIST BEHAVIOUR
    # ================================================

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """
        Controls the behaviour when option is selected on the
        model role list and the model list
        """
        list_id = event.option_list.id

        # === GO TO SESSION SELECTOR =======================
        if list_id == "sess_sear_input_trigg":
            self.show_session_search_input(sess=event.option.id)

        # === SESSION SELECTED =============================
        elif list_id == "sess_list":
            if not event.option.id:
                return
            self._go_to_session(event.option.id)


    # ================================================
    # INPUT BEHAVIOUR
    # ================================================

    def fuzzy_search_behaviour(
        self,
        event: Input.Changed,
        id: Literal["#sess_list"]
    ) -> None:
        """
        Filters the OptionList based on fuzzy matching.
        Controls:
        - The behaviour of the model list when the
          search input is changed.
        """
        if not SESS_LIST:
            return

        qry = event.value.strip().lower()
        opt_list = self.query_one(id, OptionList)
        opt_list.clear_options()

        if not qry:
            # Show entire list if query is empty
            self.show_session_list()

        else:
            # Create option for matches
            exact_matches = [entry for entry in SESS_LIST if qry in entry.lower()]
            fuzzy_matches = get_close_matches(
                qry,
                [entry.lower() for entry in SESS_LIST],
                n=5,
                cutoff=0.4
            )

            name_list = []
            for name in SESS_LIST:
                if name in exact_matches or name.lower() in fuzzy_matches:
                    name_list += name

            for name in name_list:
                sess_id = chat_logs.get_sess_id_from_name(sess_name=name)
                if not sess_id:
                    continue

                if not SESS_DICT:
                    return

                sess_with_dates = _get_session_with_dates(
                    sess_dict=SESS_DICT, sess_id=sess_id
                )

                opt_list.add_option(Option(sess_with_dates, id=name))

        # Re-highlight first match after every update
        if opt_list.option_count > 0:
            opt_list.highlighted = 0


    def on_input_changed(self, event: Input.Changed) -> None:
        """Fuzzy search for models, tokenizers and sessions."""
        super().on_input_changed(event)

        if event.input.id == "sess_sear_input":
            self.fuzzy_search_behaviour(event=event, id="#sess_list")


    def on_input_submitted(self, event: Input.Submitted) -> None:
        """
        Behaviour of the search bars for models, tokenizers and sessions
        when input is submitted.
        """
        super().on_input_submitted(event)

        # === SESSION SEARCH BAR ===========================
        if event.input.id == "sess_sear_input":

            sess_list = self.query_one("#sess_list", OptionList)
            if sess_list.highlighted is None:
                return

            selected_option = sess_list.get_option_at_index(sess_list.highlighted)
            if not selected_option.id or not self.curr_to_sess:
                return

            self._go_to_session(selected_option.id)


    # ================================================
    # KEYBINDS
    # ================================================

    def no_highlight_when_not_focused(self, id: str) -> None:
        """Highlight disable when the widgent was not in focused."""
        list = self.query_one(id, OptionList)
        if not list.has_focus:
            list.highlighted = None


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
            if self.in_sess_search:
                event.prevent_default()
                event.stop()
                self.show_sessions()

            # Only highlight when focused
            for opt_list in self.query(OptionList):
                if opt_list.highlighted is not None:
                    opt_list.highlighted = None

        # Only highlight when focused
        if event.key == "tab":
            for opt_list in self.query(OptionList):
                if opt_list.highlighted is not None:
                    opt_list.highlighted = None
