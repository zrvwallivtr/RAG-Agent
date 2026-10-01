import re
from pathlib import Path

from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, OptionList, Button
from textual.widgets.option_list import Option
from textual.events import Key


def _clean_id_string(path: Path):
    clean_str = re.sub(r'[^a-zA-Z0-9]', '-', str(path)).strip('-')
    clean_str = re.sub(r'-+', '-', clean_str)
    return f"path-{clean_str}"


class AttachmentsPopupScreen(ModalScreen[None]):
    CSS_PATH = "tcss/attachments_popup.tcss"


    def __init__(self, paths: list[Path]):
        super().__init__()
        self.paths = list(paths)


    def compose(self) -> ComposeResult:
        with Vertical(id="attachments-modal"):
            yield Static(f"[bold]Pending Attachments[/] ({len(self.paths)} pending)", id="attachments-modal-title")

            if self.paths:
                for path in self.paths:
                    with Horizontal(classes="attachment-modal-list"):
                        yield Button("x", id=f"delete-{_clean_id_string(path)}", classes="remove-attachment")
                        yield Static(str(path), id=_clean_id_string(path), classes="attachment-name")
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

            self.paths = [path for path in self.paths if _clean_id_string(path) != target_clean_id]

            # Remove the row container
            if event.button.parent:
                event.button.parent.remove()

            # Update title count
            title = self.query_one("#attachments-modal-title", Static)
            title.update(f"[bold]Pending Attachments[/] ({len(self.paths)} pending)")
