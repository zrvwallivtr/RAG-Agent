import shlex
from pathlib import Path

from textual.widgets import Static

from src.config.files_and_directories import UPLOAD_DIR
from src.app.tui.screens.attachments_popup_screen import AttachmentsPopupScreen


def _add_pending_attachments(screen, paths: list[Path]) -> None:
    """
    Appends paths to the pending attachment list, update the
    number of attachments in the attachment indicator widget
    in the user section and show command message for action
    feedback."""
    screen.pending_attchmnt.extend(paths)
    screen.query_one("#pending-attachments-counter", Static).update(
        f"Attachment ({len(screen.pending_attchmnt)} pending)"
    )
    screen.show_command_message(
        f"{len(paths)} file(s) attached. "
        f"{len(screen.pending_attchmnt)} file(s) in total"
    )


def _separate_existing_path_from_not_exist(
    all_paths: list[str] | None
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
        path = (UPLOAD_DIR / Path(p)).expanduser()
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
    paths = cmd[len(prefix):].strip()
    if not paths:
        screen.show_command_message("No file path(s) provided")
        return

    # Process paths to string
    try:
        all_paths = shlex.split(paths)
    except ValueError as e:
        screen.show_command_message(f"Invalid path syntax: {e}")
        return

    # Filtering paths
    filtered = _separate_existing_path_from_not_exist(all_paths)
    if not filtered:
        return
    found, missing = filtered
    if missing:
        screen.show_command_message(
            f"{len(missing)} file(s) not found: {', '.join(missing)}"
        )
        return

    # Only add valid paths to pending
    _add_pending_attachments(screen, found)


def reset_pending_attachment_widget(screen) -> None:
    """
    Clear pending attachment path list, reset attachment
    count in the attachment indicator widget.

    ONLY call this when there is an update on the pending attachment count:
    - User has sent the prompt to the agent.
    - User has deleted attachment(s).
    """
    screen.pending_attchmnt = []
    screen.query_one("#pending-attachments-counter", Static).update(
        f"Attachments ({len(screen.pending_attchmnt)} pending)"
    )


def to_attachments_popup(screen) -> None:
    """Pushes the manage attachments popup screen to the front."""
    def _on_manage_attachments_close(updated_paths: list[Path] | None) -> None:
        if updated_paths is not None:
            screen.pending_attchmnt = updated_paths
            screen.query_one("#pending-attachments-counter", Static).update(
                f"Attachments ({len(screen.pending_attchmnt)} pending)"
            )

    screen.app.push_screen(
        AttachmentsPopupScreen(screen.pending_attchmnt),
        _on_manage_attachments_close
    )
    return
