"""
Contain the code for how the chat container (user/assistant message box)
is drawn, and the navigation behaviours across messages and code blocks.
"""

from typing import Any, Literal

from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static, Markdown, Input
from textual.widgets.markdown import MarkdownFence

from src.libs.constants import COMPRESS_DB_ENTRY
from src.libs.helpers.file_metadata_helpers import format_size_bytes
from src.libs.constants import AUTO_COMPRESS_SUM_TITLE

from src.app.tui.movement_helpers import (
    calculate_current_vertical_scroll_component,
    calculate_jump_vertical_sroll_component,
    calculate_nth_index,
    calculate_jump_index,
)

from src import logger

CODE_SELECTED_CLASS = "code-selected"


class MessageBox(Vertical, can_focus=True):
    """A focusable container for a single chat message."""


def _unpack_data(msg: dict) -> tuple[str, str, int, Any, Any]:
    """
    Unpacks raw database message dictionary records into normalised tuples
    for UI rendering.
    """
    role = str(msg.get("role"))
    cont = str(msg.get("content", ""))

    p_tkns = msg.get("prompt_tokens") or 0
    o_tkns = msg.get("output_tokens") or 0
    tol_tkns = int(p_tkns + o_tkns)

    attchmnt_metadata = msg.get("attachments", {}) or None

    slash_cmd_metadata = msg.get(COMPRESS_DB_ENTRY)
    if not isinstance(slash_cmd_metadata, dict):
        slash_cmd_metadata = None

    return role, cont, tol_tkns, attchmnt_metadata, slash_cmd_metadata


# --- Chat container interface ------------------------------------------------

def create_user_message_box(
    cont_widget: Static | Markdown,
    attchmnt_metadata: dict | None = None,
    slash_cmd_metadata: dict | None = None,
) -> MessageBox:
    """
    Constructs the user message box, render attached files widget (if present).

    Render the original unformatted message that was stored in the database
    metadata entry (Only useful when re-rendering the chat if a slash
    command is used in that conversation turn).
    """
    widgets = [cont_widget]

    if attchmnt_metadata:
        for filename, a_metadata in attchmnt_metadata.items():
            size = a_metadata.get("size_bytes")
            fmt_size = format_size_bytes(size)
            widgets.append(Static(
                f"{filename} ({fmt_size})", classes="user-attachments"
            ))

    if slash_cmd_metadata:
        unfmt_prompt = slash_cmd_metadata.get("unformatted_prompt")
        if unfmt_prompt:
            # Replace the prompt with unformatted prompt
            widgets[0] = Static(unfmt_prompt)

    return MessageBox(*widgets, classes=f"msg-box user-box")


def create_assistant_message_box(
    cont_widget: Static | Markdown,
    tol_tkns: int,
    slash_cmd_metadata: dict | None = None,
    hide_tkns: bool = False,
) -> tuple[MessageBox, Static]:
    """
    Constructs the assistant message box, handle auto-compression summary
    message box and render message token counts widget.
    """
    # Check if auto summary is triggered
    is_auto_sum = bool(
        slash_cmd_metadata and slash_cmd_metadata.get("method") == "auto"
    )

    # Message token counts widget (Only reveal when the response is completed)
    tkn_widget = Static(
        f"{tol_tkns} token used", classes="msg-token-count"
    )
    if hide_tkns:
        tkn_widget.add_class("hidden")

    # Widget class for labelling whether if it is the regular
    # assistant message box or the auto summary box.
    classes = "msg-box assistant-box"
    if is_auto_sum:
        classes += " auto-summary"

    box = MessageBox(cont_widget, tkn_widget, classes=classes)

    # Add message box title if it is a auto summary box
    if is_auto_sum:
        box.border_title = AUTO_COMPRESS_SUM_TITLE

    return box, tkn_widget


def load_chat_history(screen) -> None:
    """
    Get entire chat history from the database and populates the chat container
    with user and assistant widget.
    """
    chat_container = screen.query_one("#chat-container", VerticalScroll)

    chat_hist = screen.agent.chat_sess_logs.get_chat_history(filter="all")
    if not chat_hist:
        return

    auto_cmp_metadata: dict | None = None

    for msg in chat_hist:
        (
            role, cont, tol_tkns, attchmnt_metadata, slash_cmd_metadata
        ) = _unpack_data(msg)

        if role == "user":
            if (
                slash_cmd_metadata and slash_cmd_metadata.get("method")
            ) == "auto":
                auto_cmp_metadata = slash_cmd_metadata
                continue

            msg_box = create_user_message_box(
                Static(cont), attchmnt_metadata, slash_cmd_metadata
            )
            chat_container.mount(msg_box)

        elif role == "assistant":
            msg_box, _ = create_assistant_message_box(
                Markdown(cont), tol_tkns, slash_cmd_metadata=auto_cmp_metadata
            )
            auto_cmp_metadata = None  # Reset for next iteration
            chat_container.mount(msg_box)

    screen.call_after_refresh(chat_container.scroll_end, animate=False)


# --- Message navigation ------------------------------------------------------

def focus_current_message_box(screen) -> None:
    chat_container = screen.query_one("#chat-container", VerticalScroll)
    msg_boxes = list(chat_container.query(".msg-box"))

    target_box = calculate_current_vertical_scroll_component(
        screen, chat_container, msg_boxes
    )
    if not target_box:
        return

    clear_code_selection(screen)
    screen.in_msg_mode = True
    screen.focused_msg_box = target_box
    target_box.focus()


def jump_messages(
    screen,
    direction: Literal["up", "down"],
    count: int = 1,
) -> None:
    """
    Jump to message box:
    - 'count' determines the number of messages to jump across.

    Only calls if in normal mode.
    """
    chat_container = screen.query_one("#chat-container", VerticalScroll)
    msg_boxes = list(chat_container.query(".msg-box"))
    
    target_box = calculate_jump_vertical_sroll_component(
        screen=screen,
        vert_scroll=chat_container,
        components=msg_boxes,
        direction=direction,
        count=count
    )
    if not target_box:
        return

    chat_container.scroll_to_widget(target_box, top=True, animate=False)

    if screen.in_msg_mode:
        screen.focused_msg_box = target_box
        target_box.focus()


def goto_message(screen, n: int) -> None:
    chat_container = screen.query_one("#chat-container", VerticalScroll)
    msg_boxes = list(chat_container.query(".msg-box"))

    idx = calculate_nth_index(n, len(msg_boxes))
    if idx is None:
        return
    chat_container.scroll_to_widget(msg_boxes[idx], top=True, animate=False)


# --- Code block navigation (message mode) ------------------------------------

def _get_code_blocks(screen) -> list[MarkdownFence]:
    box = screen.focused_msg_box
    if box is None or not box.is_attached:
        return []
    return list(box.query(MarkdownFence))


def clear_code_selection(screen) -> None:
    for block in screen.query(f".{CODE_SELECTED_CLASS}"):
        block.remove_class(CODE_SELECTED_CLASS)
    screen.code_idx = None


def _select_code_block(screen, blocks: list[MarkdownFence], idx: int) -> None:
    for i, block in enumerate(blocks):
        block.set_class(i == idx, CODE_SELECTED_CLASS)
    screen.code_idx = idx

    chat_container = screen.query_one("#chat-container", VerticalScroll)
    chat_container.scroll_to_widget(blocks[idx], top=True, animate=False)


def jump_code_blocks(
    screen,
    direction: Literal["up", "down"],
    count: int = 1,
) -> None:
    """
    Jump to code block:
    - 'count' determines the number of code block to jump across.

    Only calls if in message mode.
    """
    blocks = _get_code_blocks(screen)
    if not blocks:
        return

    curr = screen.code_idx
    if curr is not None:
        curr = min(curr, len(blocks) - 1)

    idx = calculate_jump_index(curr, len(blocks), direction, count)
    if idx is not None:
        _select_code_block(screen, blocks, idx)


def go_to_code_block(screen, n: int | None = None) -> None:
    """
    Go to code block:
    - 'n' determines the code block's index to go to, if None,
      go to the last code block

    Only call in message mode.
    """
    blocks = _get_code_blocks(screen)
    if not blocks:
        return

    idx = len(blocks) - 1 if n is None else calculate_nth_index(n, len(blocks))
    if idx is not None:
        _select_code_block(screen, blocks, idx)


# --- Horizontal scrolling of the selected code block -------------------------

def _selected_code_block(screen) -> MarkdownFence | None:
    blocks = _get_code_blocks(screen)
    if not blocks or screen.code_idx is None:
        return None
    return blocks[min(screen.code_idx, len(blocks) - 1)]


def scroll_code_block(
    screen, direction: Literal["left", "right"], count: int = 1
) -> None:
    block = _selected_code_block(screen)
    if block is None:
        return
    step = count if direction == "right" else -count
    x = max(0, min(block.scroll_x + step, block.max_scroll_x))
    block.scroll_to(x=x, animate=False)


def scroll_code_block_edge(screen, edge: Literal["start", "end"]) -> None:
    block = _selected_code_block(screen)
    if block is None:
        return
    block.scroll_to(
        x=0 if edge == "start" else block.max_scroll_x, animate=False
    )
