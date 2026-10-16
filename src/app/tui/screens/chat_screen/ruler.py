from textual.containers import VerticalScroll
from textual.widgets import Input

from src.app.tui.movement_helpers import calculate_current_vertical_scroll_component


def get_location_text(screen) -> str:
    """
    '<message>,<code block>' plus Top / Bot / All / NN% of the chat scroll.
    The code block is '-' when none is selected.
    """
    chat = screen.query_one("#chat-container", VerticalScroll)
    boxes = list(chat.query(".msg-box"))
    if not boxes:
        return ""

    if screen.in_msg_mode and screen.focused_msg_box in boxes:
        current = screen.focused_msg_box
    else:
        current = calculate_current_vertical_scroll_component(
            screen, chat, boxes
        )
    msg_n = boxes.index(current) + 1

    block = "-"
    if screen.in_msg_mode and screen.code_idx is not None:
        block = str(screen.code_idx + 1)

    max_y = chat.max_scroll_y
    if max_y <= 0:
        pct = "All"
    elif chat.scroll_y <= 0:
        pct = "Top"
    elif chat.scroll_y >= max_y - 0.5:
        pct = "Bot"
    else:
        pct = f"{int(chat.scroll_y * 100 / max_y)}%"

    return f"{msg_n},{block}".ljust(14) + pct.rjust(3)


def mode_text(screen) -> str:
    if screen.focused is screen.query_one("#prompt-input", Input):
        return "-- INSERT --"
    if screen.in_msg_mode:
        return "-- CODE BLOCK --" if screen.code_idx is not None else "-- MESSAGE --"
    return ""


def showcmd_text(screen) -> str:
    key = screen.pending_key
    if key == "space":
        return "<Spcae>"
    return key or ""
