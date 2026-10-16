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

from src.config.postgres import CONN
from src.config.config_toml import MODEL, EMBED_MODEL, FALLBACK_TOKENIZER

from .fuzzy_search import fuzzy_search_behaviour, load_sessions, show_session_list, sessions
from .tokenizers import get_tknizr_list
from .status_widgets import update_model_status_section, update_tokenizer_status_section
from .session_selector import (
    _delete_session,
    go_to_session,
    reset_session_list_layout,
    option_list_highlight_none,
    show_sessions,
    show_new_session_input,
    create_new_session,
    show_delete_session_input,
    confirm_and_delete_session,
    show_session_search_input
)

from src.agent.models import ollama
from src.agent import tokenizers
from src.app.tui.screens.base_screen import BaseScreen
from assests.icons import app_icon_ascii

MODEL_ROLES = ["chat_model", "embedding_model"]
SESSION_INPUT_IDS = {"session-search-input", "new-session-input", "delete-session-input"}


def app_icon() -> str:
    """Return app icon as Panel."""
    return app_icon_ascii.RAG


class DashboardScreen(BaseScreen):
    CSS_PATH = ["../tcss/base.tcss", "../tcss/dashboard.tcss"]

    def __init__(self):
        super().__init__()
        self.ava_models = ollama.ollama_models_list()
        self.selected_models = {
            MODEL_ROLES[0]: MODEL,
            MODEL_ROLES[1]: EMBED_MODEL,
        }

        self.fb_tknizr = FALLBACK_TOKENIZER

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
                yield Static(
                    id="model-section-title", classes="models-section"
                )
                with Horizontal(id="model-section-list"):
                    yield Static(
                        id="model-type-list", classes="model-list"
                    )
                    yield Static(
                        id="selected-model-list", classes="model-list"
                    )

                # Tokenizers
                yield Static(
                    id="tokenizer-section-title", classes="tokenizer-section"
                )
                with Horizontal(id="tokenizer-section-list"):
                    yield Static(
                        id="fallback-tokenizer", classes="tokenizer-list"
                    )
                    yield Static(
                        id="selected-tokenizer", classes="tokenizer-list"
                    )

        with Vertical(id="sessions-section"):
            # Sessions title
            yield Static(id="session-section-title")

            # Sessions actions
            with Horizontal(id="session-action-bar", classes="action-bar"):
                yield Button(
                    "Search session",
                    id="search-session-button",
                    classes="action-item",
                    compact=True,
                    flat=True,
                )
                yield Button(
                    "New session",
                    id="new-session-button",
                    classes="action-item",
                    compact=True,
                    flat=True,
                )
                yield Button(
                    "Delete session",
                    id="delete-session-button",
                    classes="action-item",
                    compact=True,
                    flat=True,
                )

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
        update_model_status_section(self)
        update_tokenizer_status_section(self)
        show_sessions(self)
        self.refresh(layout=True)
        self.refresh_status()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Controls the behaviour when action bar button is pressed."""
        if event.button.id == "search-session-button":
            show_session_search_input(self)

        elif event.button.id == "new-session-button":
            show_new_session_input(self)

        elif event.button.id == "delete-session-button":
            show_delete_session_input(self)

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
                go_to_session(self, sess_name=event.option.id)

            elif self.in_del_sess_input:
                confirm_and_delete_session(self, sess_name=event.option.id)

            else:
                go_to_session(self, sess_name=event.option.id)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Fuzzy search for models, tokenizers and sessions."""
        if event.input.id not in SESSION_INPUT_IDS:
            return

        opt_list = self.query_one("#session-list", OptionList)
        if getattr(self, "_fuzzy_timer", None):
            self._fuzzy_timer.stop()
        self._fuzzy_timer = self.set_timer(
            0.1, lambda: fuzzy_search_behaviour(event, opt_list)
        )

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

            go_to_session(self, sess_name=selected_option.id)

        # New session naming bar
        if event.input.id == "new-session-input":
            create_new_session(self, sess_name=event.value)
            return

        # Delete session search bar
        if event.input.id == "delete-session-input":
            sess_list = self.query_one("#session-list", OptionList)
            if sess_list.highlighted is None:
                self.notify("No matching session to delete.", severity="warning")
                return

            selected_option = sess_list.get_option_at_index(sess_list.highlighted)
            if not selected_option.id:
                return

            confirm_and_delete_session(self, sess_name=selected_option.id)
            return

    def on_key(self, event: Key) -> None:
        """Menu and option list keybinds."""
        super().on_key(event)

        sess_act_bar = self.query_one("#session-action-bar", Horizontal)
        sess_list = self.query_one("#session-list", OptionList)

        def _go_to_first_widget_if_not_focuse(first_wid: Widget) -> None:
            if not self.focused:
                buttons = list(first_wid.query(Button))
                buttons[0].focus()

        def _focus_first_button_in_bar(wid: Widget) -> None:
            buttons = list(wid.query(Button))
            buttons[0].focus()

        if event.key == "h":
            _go_to_first_widget_if_not_focuse(first_wid=sess_act_bar)

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
            _go_to_first_widget_if_not_focuse(first_wid=sess_act_bar)

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
            _go_to_first_widget_if_not_focuse(first_wid=sess_act_bar)

            # If focus is in session action bar
            if self.focused and sess_act_bar in self.focused.ancestors:
                sess_list.focus() # Go to session list

            # If focus is in session list but no option is highlighted
            elif self.focused == sess_list and sess_list.highlighted is None:
                _focus_first_button_in_bar(wid=sess_act_bar)

            # If focus is in option list and option is highlighted
            elif isinstance(self.focused, OptionList) and self.focused.highlighted is not None:
                event.prevent_default()
                event.stop()
                self.focused.action_cursor_up()

        elif event.key == "j":
            _go_to_first_widget_if_not_focuse(first_wid=sess_act_bar)

            # If focus is in session action bar
            if self.focused and sess_act_bar in self.focused.ancestors:
                sess_list.focus() # Go to session list

            # If focus is in session list but no option is highlighted
            elif self.focused == sess_list and sess_list.highlighted is None:
                _focus_first_button_in_bar(wid=sess_act_bar)

            # If focus is in option list and option is highlighted
            elif isinstance(self.focused, OptionList) and self.focused.highlighted is not None:
                event.prevent_default()
                event.stop()
                self.focused.action_cursor_down()

        elif event.key == "enter":
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

        elif event.key == "tab":
            event.prevent_default()
            event.stop()

            inputs = [
                self.query_one("#session-search-input", Input),
                self.query_one("#new-session-input", Input),
                self.query_one("#delete-session-input", Input),
            ]

            if self.focused in inputs:
                if sess_list.option_count > 0:
                    if sess_list.highlighted is None:
                        sess_list.highlighted = 1 if sess_list.option_count > 1 else 0
                    else:
                        next_idx = (sess_list.highlighted + 1) % sess_list.option_count

                        if next_idx == 0 and sess_list.option_count > 1:
                            next_idx = 1

                        sess_list.highlighted = next_idx

            # Ensure focus stays on active input bar
            self.focused.focus()

        elif event.key == "escape":
            event.prevent_default()
            event.stop()
            reset_session_list_layout(self)
            option_list_highlight_none(self)
