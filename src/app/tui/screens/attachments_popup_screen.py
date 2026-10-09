import re
from pathlib import Path

from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, OptionList, Button
from textual.widgets.option_list import Option
from textual.events import Key

from src.libs.helpers.string_helpers import clean_id_string


class AttachmentsPopupScreen(ModalScreen[None]):
    CSS_PATH = "tcss/attachments_popup.tcss"

    def __init__(self, paths: list[Path]):
        super().__init__()
        self.paths = list(paths)

    def compose(self) -> ComposeResult:
        with Vertical(id="attachments-modal"):
            yield Static(self._modal_title(), id="attachments-modal-title")

            if self.paths:
                for path in self.paths:
                    with Horizontal(classes="attachment-modal-list"):
                        yield Button("x", id=f"delete-{clean_id_string(path)}", classes="remove-attachment")
                        yield Static(str(path), id=clean_id_string(path), classes="attachment-name")
            else:
                yield Static("[dim]No pending attachments[/]")

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            event.prevent_default()
            event.stop()
            
            self.dismiss(self.paths) # Return updated paths list on dismiss

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id and event.button.id.startswith("delete-"):
            target_clean_id = event.button.id.removeprefix("delete-")

            self.paths = [path for path in self.paths if clean_id_string(path) != target_clean_id]

            # Remove the row container
            if event.button.parent:
                event.button.parent.remove()

            # Update title count
            title = self.query_one("#attachments-modal-title", Static)
            title.update(self._modal_title())

    def _modal_title(self):
        return f"[bold]Pending Attachments[/] ({len(self.paths)} pending)"
