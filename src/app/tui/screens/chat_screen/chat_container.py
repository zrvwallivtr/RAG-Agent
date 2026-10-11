from typing import Any

from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static, Markdown

from src.libs.constants import COMPRESS_DB_ENTRY
from src.libs.helpers.tui_helpers.size_bytes_helpers import format_size_bytes
from src.libs.constants import AUTO_COMPRESS_SUM_TITLE

from src import logger


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


def create_user_message_box(
    cont_widget: Static | Markdown,
    attchmnt_metadata: dict | None = None,
    slash_cmd_metadata: dict | None = None,
) -> Vertical:
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

    return Vertical(*widgets, classes=f"msg-box user-box")


def create_assistant_message_box(
    cont_widget: Static | Markdown,
    tol_tkns: int,
    slash_cmd_metadata: dict | None = None,
    hide_tkns: bool = False,
) -> tuple[Vertical, Static]:
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

    box = Vertical(cont_widget, tkn_widget, classes=classes)

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
        role, cont, tol_tkns, attchmnt_metadata, slash_cmd_metadata = _unpack_data(msg)

        if role == "user":
            if slash_cmd_metadata and slash_cmd_metadata.get("method") == "auto":
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

    chat_container.scroll_end(animate=False)
