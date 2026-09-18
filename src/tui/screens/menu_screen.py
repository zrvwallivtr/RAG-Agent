from difflib import get_close_matches
from typing import Literal

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
SESS_DICT = chat_logs.get_all_existing_sess_metadata()
SESS_LIST = (
    [SESS_DICT[sess]["session_name"] for sess in SESS_DICT]
    if SESS_DICT
    else None
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

        self.in_model_search: bool = False
        self.in_tknizr_search: bool = False
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
            yield Static(id="model_section")

            # Model role list
            yield OptionList(id="model_role_list")

            # Modle search bar (show when role selected)
            yield Input(
                id="model_sear_input",
                placeholder="Search model...",
                classes="hidden",
                select_on_focus=False
            )
            yield OptionList(id="model_list", classes="hidden")

            # === TOKENIZERS STATUS ===================
            yield Static(id="tknizr_section")

            # Fallback tokenizer
            yield OptionList(id="fb_tknizr")

            # Tokenizer search bar (show when fallback tokenizer selected)
            yield Input(
                id="tknizr_sear_input",
                placeholder="Search tokenizer...",
                classes="hidden",
                select_on_focus=False
            )
            yield OptionList(id="tknizr_list", classes="hidden")

            # === SESSIONS STATUS =====================
            yield Static(id="sess_section")

            # Latest sessions
            yield OptionList(id="latest_sess")

            # Go to session
            yield OptionList(id="to_sess")

            # Session search bar (show when to session is selected)
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
        #self.show_to_session()


    # ================================================
    # MODEL SECTION
    # ================================================

    def show_model_roles(self) -> None:
        """
        List of all roles that available for model selection,
        displaying roles and corresponding selected model.
        """
        self.in_model_search = False # User not searching models
        self.curr_model_role = None

        # Section title
        title = f"[bold]Models[/] ({len(self.ava_models)} installed)"
        self.query_one("#model_section", Static).update(title)

        # Show model roles option list
        model_role_list = self.query_one("#model_role_list", OptionList)
        model_role_list.clear_options()

        label_width = max(len(role.replace("_", " ").capitalize()) for role in MODEL_ROLES) + 2

        for role in MODEL_ROLES:
            label = role.replace("_", " ").capitalize()
            padded_label = f"{label}:".ljust(label_width)
            model_role_list.add_option(Option(f"{padded_label}\t[yellow]{self.selected_models[role]}[/]", id=role))
        model_role_list.remove_class("hidden")
        model_role_list.focus()

        # Hide model fuzzy search bar
        self.query_one("#model_sear_input", Input).add_class("hidden")
        self.query_one("#model_list", OptionList).add_class("hidden")


    def show_model_selector(self, role: str) -> None:
        """
        Consist of the search input bar and all available model
        list. Model list will be updated according to the search bar.
        """
        self.in_model_search = True # User is searching models
        self.curr_model_role = role

        # Models option list
        model_list = self.query_one("#model_list", OptionList)
        model_list.clear_options()
        for model in self.ava_models:
            model_list.add_option(Option(model, id=model))

        # Highlight first match
        if model_list.option_count > 0:
            model_list.highlighted = 0

        # Fuzzy search
        search_input = self.query_one("#model_sear_input", Input)
        search_input.value = ""
        search_input.remove_class("hidden")
        model_list.remove_class("hidden")
        search_input.focus()


    # ================================================
    # TOKENIZER SECTION
    # ================================================

    def show_fallback_tokenizer(self) -> None:
        """
        Show current selected fallback tokenizer for
        the models. If selected, the fuzzy search
        tokenizer selector will popup.
        """
        self.in_tknizr_search = False # User not searching tokenizers
        self.curr_fb_tknizr = None

        # Section title
        title = (
            f"[bold]Tokenizers[/bold] (0 installed)"
            if not TKNIZR_LIST
            else f"[bold]Tokenizers[/bold] ({len(TKNIZR_LIST)} installed)"
        )
        self.query_one("#tknizr_section", Static).update(title)

        # Show fallback tokenizer option
        fb_tknizr = self.query_one("#fb_tknizr", OptionList)
        fb_tknizr.clear_options()
        fb_tknizr.add_option(Option(f"Fallback tokenizer:\t[yellow]{models.FALLBACK_TOKENIZER}[/]", id=models.FALLBACK_TOKENIZER))
        fb_tknizr.remove_class("hidden")
        fb_tknizr.focus()

        # Hide tokenizer fuzzy search bar
        self.query_one("#tknizr_sear_input", Input).add_class("hidden")
        self.query_one("#tknizr_list", OptionList).add_class("hidden")


    def show_fallback_tokenizer_selector(self, tknizr: str) -> None:
        """
        Consist of the search input bar and all available tokenizer
        list. Tokenizer list will be updated according to the search bar.
        """
        self.in_tknizr_search = True # User is searching tokenizers
        self.curr_fb_tknizr = tknizr

        # Tokenizer option list
        tknizr_list = self.query_one("#tknizr_list", OptionList)
        tknizr_list.clear_options()
        if not TKNIZR_LIST:
            return
        for tknizr in TKNIZR_LIST:
            tknizr_list.add_option(Option(tknizr, id=tknizr))

        # Highlight first match
        if tknizr_list.option_count > 0:
            tknizr_list.highlighted = 0

        # Fuzzy search
        search_input = self.query_one("#tknizr_sear_input", Input)
        search_input.value = ""
        search_input.remove_class("hidden")
        tknizr_list.remove_class("hidden")
        search_input.focus()


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
        latest = chat_logs.latest_modified_chat_session()
        sess_name, sess_dt = latest if latest else (None, None)

        title = (
            f"[bold]Sessions[/bold] ({len(sess_dict)} created)"
            if sess_dict
            else f"[bold]Sessions[/bold] (0 created)\n"
        )
        self.query_one("#sess_section", Static).update(title)

        if not latest:
            return

        # Show created sessions option
        latest_sess = self.query_one("#latest_sess", OptionList)
        latest_sess.clear_options()
        latest_sess.add_option(Option(f"Latest: [blue]{sess_dt}[/blue] [yellow]{sess_name}[/yellow]", id=sess_name))
        latest_sess.remove_class("hidden")
        latest_sess.focus()

        # Show go to session option
        to_sess = self.query_one("#to_sess", OptionList)
        to_sess.clear_options()
        to_sess.add_option(Option("Go to session"))
        to_sess.remove_class("hidden")
        to_sess.focus()

        # Hide session fuzzy search bar
        self.query_one("#sess_sear_input", Input).add_class("hidden")
        self.query_one("#sess_list", OptionList).add_class("hidden")


    # def show_to_session(self) -> None:
    #     """
    #     Show 'Go to session' option. If selected, the fuzzy search
    #     session selector will popup.
    #     """
    #     self.in_sess_search = False # User not searching sessions
    #     self.curr_to_sess = None

    #     # Show go to session option
    #     to_sess = self.query_one("#to_sess", OptionList)
    #     to_sess.clear_options()
    #     to_sess.add_option(Option("Go to session"))
    #     to_sess.remove_class("hidden")
    #     to_sess.focus()

    #     # Hide session fuzzy search bar
    #     self.query_one("#sess_sear_input", Input).add_class("hidden")
    #     self.query_one("#sess_list", OptionList).add_class("hidden")


    def show_session_selector(self, sess: str) -> None:
        """
        Consist of the search input bar and available sessions list.
        Session list will be updated according to the search bar.
        """
        self.in_sess_search = True # User is searching sessions
        self.curr_to_sess = sess

        # Session option list
        sess_list = self.query_one("#sess_list", OptionList)
        sess_list.clear_options()
        if not SESS_LIST:
            return
        for sess in SESS_LIST:
            sess_list.add_option(Option(sess, id=sess))

        # Highlight first match
        if sess_list.option_count > 0:
            sess_list.highlighted = 0

        # Fuzzy search
        search_input = self.query_one("#sess_sear_input", Input)
        search_input.value = ""
        search_input.remove_class("hidden")
        sess_list.remove_class("hidden")
        search_input.focus()

        # Hide go to session option
        self.query_one("#to_sess", OptionList).add_class("hidden")


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
 
        # === MODEL ROLE SELECTED ==========================
        if list_id == "model_role_list":
            self.show_model_selector(role=event.option.id)

        # === MODEL SELECTED FOR ROLE ======================
        elif list_id == "model_list":
            if not event.option.id:
                return
            if not self.curr_model_role:
                return
            selected_model = event.option.id
            self.selected_models[self.curr_model_role] = selected_model # Update model role list
            self.notify(
                f"{self.curr_model_role.replace('_', ' ').capitalize()} set to: {selected_model}"
            )
            self.show_model_roles()

        # === FALLBACK TOKENIZER SELECTED ==================
        elif list_id == "fb_tknizr":
            self.show_fallback_tokenizer_selector(tknizr=event.option.id)

        # === TOKENIZER SELECTED ===========================
        elif list_id == "tknizr_list":
            if not event.option.id:
                return
            if not self.curr_fb_tknizr:
                return
            selected_tknizr = event.option.id
            self.fb_tknizr = selected_tknizr # Update tokenizer list
            self.notify(
                f"Fallback tokenizer set to: {selected_tknizr}"
            )
            self.show_fallback_tokenizer()

        # === LATEST SESSIONS SELECTED =====================
        elif list_id == "latest_sess":
            if not event.option.id:
                return
            self._go_to_session(event.option.id)

        # === GO TO SESSION SELECTED =======================
        elif list_id == "to_sess":
            self.show_session_selector(sess=event.option.id)

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
        id: Literal["#model_list", "#tknizr_list", "#sess_list"]
    ) -> None:
        """
        Filters the OptionList based on fuzzy matching.
        Controls:
        - The behaviour of the model list when the
          search input is changed.
        """
        if id == "#model_list":
            ava_list = self.ava_models
        elif id == "#tknizr_list":
            if not TKNIZR_LIST:
                return
            ava_list = TKNIZR_LIST
        elif id == "#sess_list":
            if not SESS_LIST:
                return
            ava_list = SESS_LIST

        qry = event.value.strip().lower()
        opt_list = self.query_one(id, OptionList)
        opt_list.clear_options()

        if not qry:
            # Show entire list if query is empty
            for entry in ava_list:
                opt_list.add_option(Option(entry, id=entry))

        else:
            exact_matches = [entry for entry in ava_list if qry in entry.lower()]
            fuzzy_matches = get_close_matches(
                qry,
                [entry.lower() for entry in ava_list],
                n=5,
                cutoff=0.4
            )
            results = [entry for entry in ava_list if entry in exact_matches or entry.lower() in fuzzy_matches]
            for r in results:
                opt_list.add_option(Option(r, id=r))

        # Re-highlight first match after every update
        if opt_list.option_count > 0:
            opt_list.highlighted = 0


    def on_input_changed(self, event: Input.Changed) -> None:
        """Fuzzy search for models, tokenizers and sessions."""
        super().on_input_changed(event)

        if event.input.id == "model_sear_input":
            self.fuzzy_search_behaviour(event=event, id="#model_list")
            return

        if event.input.id == "tknizr_sear_input":
            self.fuzzy_search_behaviour(event=event, id="#tknizr_list")
            return

        if event.input.id == "sess_sear_input":
            self.fuzzy_search_behaviour(event=event, id="#sess_list")


    def on_input_submitted(self, event: Input.Submitted) -> None:
        """
        Behaviour of the search bars for models, tokenizers and sessions
        when input is submitted.
        """
        super().on_input_submitted(event)

        # === MODEL SEARCH BAR =============================
        if event.input.id == "model_sear_input":

            model_list = self.query_one("#model_list", OptionList)
            if model_list.highlighted is None:
                return

            selected_option = model_list.get_option_at_index(model_list.highlighted)
            if not selected_option.id or not self.curr_model_role:
                return

            selected_model = selected_option.id
            self.selected_models[self.curr_model_role] = selected_model
            self.notify(f"{self.curr_model_role.replace('_', ' ').capitalize()} set to: {selected_model}")
            self.show_model_roles()

        # === TOKENIZER SEARCH BAR =========================
        if event.input.id == "tknizr_sear_input":

            tknizr_list = self.query_one("#tknizr_list", OptionList)
            if tknizr_list.highlighted is None:
                return

            selected_option = tknizr_list.get_option_at_index(tknizr_list.highlighted)
            if not selected_option.id or not self.curr_fb_tknizr:
                return

            selected_tknizr = selected_option.id
            self.fb_tknizr = selected_tknizr
            self.notify(f"Fallback tokenizer set to: {selected_tknizr}")
            self.show_fallback_tokenizer()

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

        # scroll_bar = self.query_one("#status_container", Vertical)

        # # Menu navigation
        # if event.key == "k":
        #     scroll_bar.scroll_up()
        #     self.pending_key = None

        # elif event.key == "j":
        #     scroll_bar.scroll_down()
        #     self.pending_key = None

        # elif event.character == "G":
        #     scroll_bar.scroll_end(animate=False)
        #     self.pending_key = None

        # elif event.character == "g":
        #     if self.pending_key == "g":
        #         scroll_bar.scroll_home(animate=False)
        #         self.pending_key = None
        #     else:
        #         self.pending_key = "g"

        # else:
        #     self.pending_key = None

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
            if self.in_model_search:
                event.prevent_default()
                event.stop()
                self.show_model_roles()

            elif self.in_tknizr_search:
                event.prevent_default()
                event.stop()
                self.show_fallback_tokenizer()

            elif self.in_sess_search:
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

