from difflib import get_close_matches
from typing import Literal
from datetime import datetime
from psycopg2.errors import UniqueViolation

from rich import box
from rich.panel import Panel
from rich.align import Align 

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Static, Input, OptionList, Button
from textual.widgets.option_list import Option
from textual.events import Key

from src.config import models, postgres
from src.agent.models import ollama
from src.agent import tokenizers
from src.app.tui.screens.base_screen import BaseScreen
from assests.icons import app_icon_ascii

from src.app.tui.screens.helpers import date_helpers
from src.app.tui.screens.helpers import tokenizers_helpers
from src.app.tui.screens.helpers import sessions_helpers


MODEL_ROLES = ["chat_model", "memory_model", "web_search_model", "embedding_model"]


def app_icon() -> str:
    """Return app icon as Panel."""
    return app_icon_ascii.RAG


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
        with Horizontal(id="app-banner"):
            yield Static(id="app-icon", classes="box")

            with Vertical(id="app-config", classes="box"):
                # Models
                yield Static(id="model-section-title", classes="models-section")
                with Horizontal(id="model-section-list"):
                    yield Static(id="model-type-list", classes="model-list")
                    yield Static(id="selected-model-list", classes="model-list")

                # Tokenizers
                yield Static(id="tokenizer-section-title", classes="tokenizer-section")
                with Horizontal(id="tokenizer-section-list"):
                    yield Static(id="fallback-tokenizer", classes="tokenizer-list")
                    yield Static(id="selected-tokenizer", classes="tokenizer-list")

        with Vertical(id="sessions-section"):
            # Sessions title
            yield Static(id="session-section-title")

            # Sessions actions
            with Horizontal(id="session-action-bar", classes="action-bar"):
                yield Button("Search session", id="search-session-button", classes="action-item")
                yield Button("New session", id="new-session-button", classes="action-item")
                yield Button("Delete session", id="delete-session-button", classes="action-item")

            yield Input(
                id="session-search-input",
                placeholder="Search session name...",
                classes="hidden",
                select_on_focus=False
            )
            yield Input(
                id="new-session-input",
                placeholder="New session name...",
                classes="hidden",
                select_on_focus=False
            )
            yield Input(
                id="delete-session-input",
                placeholder="Delete session name...",
                classes="hidden",
                select_on_focus=False
            )

            # Sessions list
            yield OptionList(id="session-list", classes="hidden")

        yield from self.compose_command_bar()


    def on_mount(self) -> None:
        """Show app icon, app description and status on startup."""
        self.query_one("#app-icon", Static).update(app_icon())
        self._update_model_status_section()
        self._update_tokenizer_status_section()
        self._show_sessions()
        self.refresh(layout=True)


    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Controls the behaviour when action bar button is pressed."""
        if event.button.id == "search-session-button":
            self._show_session_search_input()

        elif event.button.id == "new-session-button":
            self._show_new_session_input()

        elif event.button.id == "delete-session-button":
            self._show_delete_session_input()


    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """
        Controls the behaviour when option is selected.
        """
        event.prevent_default()
        event.stop()

        list_id = event.option_list.id

        if list_id == "session-list":
            if not event.option.id:
                return

            elif self.in_sess_search or self.in_new_sess_input:
                self._go_to_session(sess_name=event.option.id)

            elif self.in_del_sess_input:
                self._confirm_and_delete_session(sess_name=event.option.id)

            else:
                self._go_to_session(sess_name=event.option.id)


    def on_input_changed(self, event: Input.Changed) -> None:
        """Fuzzy search for models, tokenizers and sessions."""
        super().on_input_changed(event)

        if event.input.id == "session-search-input":
            self._fuzzy_search_behaviour(event=event, id="#session-list")

        if event.input.id == "new-session-input":
            self._fuzzy_search_behaviour(event=event, id="#session-list")

        if event.input.id == "delete-session-input":
            self._fuzzy_search_behaviour(event=event, id="#session-list")


    def on_input_submitted(self, event: Input.Submitted) -> None:
        """
        Behaviour of the search bars for models, tokenizers and sessions
        when input is submitted.
        """
        super().on_input_submitted(event)

        # Session search bar
        if event.input.id == "session-search-input":
            sess_list = self.query_one("#session-list", OptionList)
            if sess_list.highlighted is None:
                return

            selected_option = sess_list.get_option_at_index(sess_list.highlighted)
            if not selected_option.id:
                return

            self._go_to_session(sess_name=selected_option.id)

        # New session naming bar
        if event.input.id == "new-session-input":
            self._create_new_session(sess_name=event.value)
            return

        # Delete session search bar
        if event.input.id == "delete-session-input":
            self._confirm_and_delete_session(sess_name=event.value)
            return


    def on_key(self, event: Key) -> None:
        """Menu and option list keybinds."""
        super().on_key(event)

        sess_act_bar = self.query_one("#session-action-bar", Horizontal)
        sess_list = self.query_one("#session-list", OptionList)

        if event.key == "h":
            self._go_to_first_widget_if_not_focuse(first_wid=sess_act_bar)

            # If focus is in session action bar
            if self.focused and sess_act_bar in self.focused.ancestors:
                buttons = list(sess_act_bar.query(Button))

                if self.focused in buttons:
                    # Go to previous button in session action bar
                    curr_idx = buttons.index(self.focused)
                    prev_idx = (curr_idx - 1) % len(buttons)
                    buttons[prev_idx].focus()
                else:
                    # Focus on the first button if none focused
                    buttons[0].focus()

        elif event.key == "l":
            self._go_to_first_widget_if_not_focuse(first_wid=sess_act_bar)

            # If focus is in session action bar
            if self.focused and sess_act_bar in self.focused.ancestors:
                buttons = list(sess_act_bar.query(Button))

                if self.focused in buttons:
                    # Go to next button in session action bar
                    curr_idx = buttons.index(self.focused)
                    next_idx = (curr_idx + 1) % len(buttons)
                    buttons[next_idx].focus()
                else:
                    # Focus on the first button if none focused
                    buttons[0].focus()

        elif event.key == "k":
            self._go_to_first_widget_if_not_focuse(first_wid=sess_act_bar)

            # If focus is in session action bar
            if self.focused and sess_act_bar in self.focused.ancestors:
                sess_list.focus() # Go to session list

            # If focus is in session list but no option is highlighted
            elif self.focused == sess_list and sess_list.highlighted is None:
                self._focus_first_button_in_bar(wid=sess_act_bar)

            # If focus is in option list and option is highlighted
            elif isinstance(self.focused, OptionList) and self.focused.highlighted is not None:
                event.prevent_default()
                event.stop()
                self.focused.action_cursor_up()

        elif event.key == "j":
            self._go_to_first_widget_if_not_focuse(first_wid=sess_act_bar)

            # If focus is in session action bar
            if self.focused and sess_act_bar in self.focused.ancestors:
                sess_list.focus() # Go to session list

            # If focus is in session list but no option is highlighted
            elif self.focused == sess_list and sess_list.highlighted is None:
                self._focus_first_button_in_bar(wid=sess_act_bar)

            # If focus is in option list and option is highlighted
            elif isinstance(self.focused, OptionList) and self.focused.highlighted is not None:
                event.prevent_default()
                event.stop()
                self.focused.action_cursor_down()

        if event.key == "enter":
            # If focus is on the session list or in session list
            if self.focused and (self.focused == sess_list or sess_list in self.focused.ancestors):
                # If no option is highlighted
                if sess_list.highlighted is None:
                    event.prevent_default()
                    event.stop()
                    if sess_list.option_count > 1:
                        sess_list.highlighted = 1
                    elif sess_list.option_count > 0:
                        sess_list.highlighted = 0

        if event.key == "escape":
            event.prevent_default()
            event.stop()
            self._reset_session_list_layout()
            self._option_list_hightlight_none()

        if event.key == "tab":
            self._reset_session_list_layout()
            self._option_list_hightlight_none()


    def _go_to_first_widget_if_not_focuse(self, first_wid: Widget) -> None:
        if not self.focused:
            buttons = list(first_wid.query(Button))
            buttons[0].focus()


    def _focus_first_button_in_bar(self, wid: Widget) -> None:
        buttons = list(wid.query(Button))
        buttons[0].focus()


    def _reset_session_list_layout(self) -> None:
        """Return to default sessions view if in an input mode."""
        if self.in_sess_search or self.in_new_sess_input or self.in_del_sess_input:
            self.in_new_sess_input = False
            self.in_del_sess_input = False
            self._show_sessions()


    def _option_list_hightlight_none(self) -> None:
        """Exist highlight if any OptionList is highlighted."""
        for opt_list in self.query(OptionList):
            if opt_list.highlighted is not None:
                opt_list.highlighted = None


    def _update_model_status_section(self):
        self.query_one("#model-section-title", Static).update(f"[bold]Models[/] ({len(self.ava_models)} installed)")

        model_list = []
        for model_type in MODEL_ROLES:
            model_names = model_type.replace("_", " ").capitalize()
            model_list.append(f"{model_names}")
        models_str = "\n".join(model_list)

        selected_list = []
        for model in self.selected_models.values():
            selected_list.append(f"{model}")
        selected_str = "\n".join(selected_list)

        self.query_one("#model-type-list", Static).update(models_str)
        self.query_one("#selected-model-list", Static).update(selected_str)


    def _update_tokenizer_status_section(self):
        tknizr_list = tokenizers_helpers.get_tknizr_list()
        tknizr_count = len(tknizr_list) if tknizr_list else 0

        self.query_one("#tokenizer-section-title", Static).update(f"[bold]Tokenizers[/] ({tknizr_count} installed)")
        self.query_one("#fallback-tokenizer", Static).update("Fallback tokenizer:")
        self.query_one("#selected-tokenizer", Static).update(f"{models.FALLBACK_TOKENIZER}")


    def _show_session_action_bar(self) -> None:
        """Option list for search session, new session, delete session."""
        self.in_sess_search = False # User not searching sessions
        self.curr_to_sess = None

        action_bar = self.query_one("#session-action-bar")
        action_bar.remove_class("hidden")
        self.query_one("#search-session-button", Button).focus()


    def _show_session_list(self) -> None:
        """
        Show list of sessions with created time, modified time and session name.
        """
        from src.agent.chat_logs import ChatLogs
        chat_logs = ChatLogs(conn=postgres.conn)

        sess_dict = chat_logs.get_all_existing_sess_metadata()
        if not sess_dict:
            return

        sess_list = self.query_one("#session-list", OptionList)
        sess_list.clear_options()

        list_title = Option("Last modified\t\tCreated at\t\tSession name\n", id="session-option-list-title")
        list_title.disabled = True
        sess_list.add_option(list_title)

        for sess_id in sess_dict:
            sess_name = sess_dict[sess_id]["session_name"]
            sess_with_dates = date_helpers.get_session_with_dates(
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
        self.query_one("#session-section-title", Static).update(title)

        # Session selector
        self._show_session_action_bar()
        self._show_session_list()

        # Hide widgets
        self.query_one("#session-search-input", Input).add_class("hidden")
        self.query_one("#new-session-input", Input).add_class("hidden")
        self.query_one("#delete-session-input", Input).add_class("hidden")


    def _show_session_search_input(self) -> None:
        """
        Consist of the search input bar and available sessions list.
        Session list will be updated according to the search bar.
        """
        self.in_sess_search = True # User is searching sessions

        # Session option list
        sess_list = self.query_one("#session-list", OptionList)

        # Highlight first match
        if sess_list.option_count > 1:
            sess_list.highlighted = 1

        # Fuzzy search
        search_input = self.query_one("#session-search-input", Input)
        search_input.value = ""
        search_input.remove_class("hidden")
        sess_list.remove_class("hidden")
        search_input.focus()

        # Hide session search input trigger
        self.query_one("#session-action-bar").add_class("hidden")


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
        sess_list = self.query_one("#session-list", OptionList)

        # Highlight first match
        if sess_list.option_count > 1:
            sess_list.highlighted = 1

        # Input bar
        new_sess_input = self.query_one("#new-session-input", Input)
        new_sess_input.value = ""
        new_sess_input.remove_class("hidden")
        sess_list.remove_class("hidden")
        new_sess_input.focus()

        # Hide widgets
        self.query_one("#session-action-bar").add_class("hidden")


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
        sess_list = self.query_one("#session-list", OptionList)

        # Input bar
        del_sess_input = self.query_one("#delete-session-input", Input)
        del_sess_input.value = ""
        del_sess_input.remove_class("hidden")
        sess_list.remove_class("hidden")
        del_sess_input.focus()

        # Hide widgets
        self.query_one("#session-action-bar").add_class("hidden")


    def _confirm_and_delete_session(self, sess_name: str) -> None:
        clean_name = sess_name.strip()
        if not clean_name:
            self.notify("Session name cannot be empty.", severity="error")
            return

        from src.app.tui.screens.confirmation_popup_screen import ConfirmationPopupScreen

        def handle_confirmation(is_confirm: bool | None) -> None:
            if is_confirm:
                self._delete_session(sess_name=clean_name)

        title = f"Proceed to delete session '{sess_name}'?"
        positive = "Proceed"
        negative = "Cancel"
        self.app.push_screen(
            ConfirmationPopupScreen(title=title, positive=positive, negative=negative),
            callback=handle_confirmation
        )


    def _delete_session(self, sess_name: str) -> None:
        """Delete session."""
        from src.app.operations import sessions
        result = sessions.del_sess(sess_name=sess_name)

        if result and "fail" in result.lower():
            self.notify(result, severity="error")
            return

        self.notify(f"Session deleted: {sess_name}")
        self._show_sessions()


    def _fuzzy_search_behaviour(
        self,
        event: Input.Changed,
        id: Literal["#session-list"],
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

        sess_list = sessions_helpers.get_session_list()
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

                sess_with_dates = date_helpers.get_session_with_dates(
                    sess_dict=sess_dict, sess_id=sess_id
                )
                opt_list.add_option(Option(sess_with_dates, id=name))

        # Re-highlight first match after every update
        if opt_list.option_count > 1:
            opt_list.highlighted = 1
