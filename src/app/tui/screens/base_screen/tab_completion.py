from pathlib import Path

from textual.events import Key
from textual.widgets import Input

from src.config.files_and_directories import UPLOAD_DIR
from src.config.config_toml import ENABLE_PATH_SAFETY


def _start_directory(
    enable_path_safety: bool = ENABLE_PATH_SAFETY,
    upload_dir: Path = UPLOAD_DIR,
) -> Path:
    if enable_path_safety:
        return upload_dir
    return Path("~/")


def tab_completion(screen, cmd_input: Input, event: Key):
    """
    Handles tab completion for attachment commands, scans UPLOAD_DIR for
    matching files/folders and auto-completes the input path.
    """
    event.prevent_default()
    event.stop()

    val = cmd_input.value

    # Check if it's an attach command
    if val.startswith(":a ") or val.startswith(":attach "):
        prefix = ":a " if val.startswith(":a ") else ":attach "
        partial_path = val[len(prefix) :].strip()

        # Resolve directory and search term
        # Support partial paths like subfolders or simple filenames
        search_path = Path(partial_path).expanduser()

        if (
            partial_path.endswith("/")
            or not search_path.suffix
            and not search_path.name
        ):
            base_dir = _start_directory() / search_path
            pattern = "*"
        else:
            base_dir = _start_directory() / search_path.parent
            pattern = f"{search_path.name}*"

        if base_dir.exists() and base_dir.is_dir():
            # Find matching files/folders inside UPLOAD_DIR
            matches = [
                str(p.relative_to(_start_directory())) for p in base_dir.glob(pattern)
            ]

            if len(matches) == 1:
                # Exact single match: auto-complete it
                completed = f"{prefix}{matches[0]}"
                if Path(_start_directory() / matches[0]).is_dir():
                    completed += "/"
                cmd_input.value = completed
                cmd_input.cursor_position = len(completed)
