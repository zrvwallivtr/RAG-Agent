from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, OptionList, Button
from textual.widgets.option_list import Option
from textual.events import Key


class ConfirmationPopupScreen(ModalScreen[None]):
    CSS_PATH = "tcss/confirmation_popup.tcss"


    def __init__(self, title: str, positive: str, negative: str):
        super().__init__()
        self.title = title
        self.positive = positive
        self.negative = negative


    def compose(self) -> ComposeResult:
        with Vertical(id=f"confirmation-popup-modal"):
            yield Static(f"[bold]{self.title}[/]", id="confirmation-modal-title")

            with Horizontal(id=f"confirmation-modal-button-section"):
                yield Button(f"{self.positive}", id="confirmation-modal-true-button")
                yield Button(f"{self.negative}", id="confirmation-modal-false-button")


    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "modal-positive-button":
            self.dismiss(True)
        else:
            self.dismiss(False)


    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            event.prevent_default()
            event.stop()
            self.dismiss(False)
