"""
Session attachments modules, contain functions for the session
pending attachments indicator, and the attachment management
dialog.
"""

import shlex
from pathlib import Path

from textual.widgets import Static

from src.config.files_and_directories import UPLOAD_DIR
from src.config.config_toml import ENABLE_PATH_SAFETY

from src.app.tui.screens.attachments_dialog_screen import AttachmentsDialogScreen
from src.app.tui.screens.base_screen.command_bar import show_command_message


def _add_pending_attachments(screen, paths: list[Path]) -> None:
    screen.pending_attchmnt.extend(paths)
    screen.refresh_status()
    show_command_message(
        screen,
        f"{len(paths)} file(s) attached. "
        f"{len(screen.pending_attchmnt)} file(s) in total",
    )


def _complete_path_helper(path: str) -> Path:
    """
    Helper for automatically adding the preset upload directory at the front
    of the path if path safety is enabled.
    """
    if ENABLE_PATH_SAFETY:
        return (UPLOAD_DIR / Path(path)).expanduser()
    return Path(path)


def _separate_existing_path_from_not_exist(
    all_paths: list[str] | None,
) -> tuple[list, list | None] | None:
    """
    Filter all paths into two list to separate found and not
    found paths.
    """
    if not all_paths:
        return

    found = []
    missing = []

    for p in all_paths:
        path = _complete_path_helper(p)
        if path.exists():
            found.append(path)
        else:
            missing.append(p)

    return found, missing


def process_add_pending_attachments(screen, cmd: str, prefix: str) -> None:
    """
    Retrieve all provided path(s) in the user input command;
    display message for: zero given file path, invalid syntax,
    not found path. Add all found files to the pending
    attachment list.
    """
    paths = cmd[len(prefix) :].strip()
    if not paths:
        show_command_message(screen, "No file path(s) provided")
        return

    # Process paths to string
    try:
        all_paths = shlex.split(paths)
    except ValueError as e:
        show_command_message(screen, f"Invalid path syntax: {e}")
        return

    # Filtering paths
    filtered = _separate_existing_path_from_not_exist(all_paths)
    if not filtered:
        return
    found, missing = filtered
    if missing:
        show_command_message(
            screen, f"{len(missing)} file(s) not found: {', '.join(missing)}"
        )
        return

    # Only add valid paths to pending
    _add_pending_attachments(screen, found)


def reset_pending_attachment_widget(screen) -> None:
    screen.pending_attchmnt = []
    screen.refresh_status()


def to_attachments_popup(screen) -> None:
    """Pushes the manage attachments popup screen to the front."""

    def _on_manage_attachments_close(updated_paths: list[Path] | None) -> None:
        if updated_paths is not None:
            screen.pending_attchmnt = updated_paths
            screen.refresh_status()

    screen.app.push_screen(
        AttachmentsDialogScreen(screen.pending_attchmnt), _on_manage_attachments_close
    )
