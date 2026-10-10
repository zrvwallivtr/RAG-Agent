from typing import Any

from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static, Markdown

from src.libs.constants import COMPRESS_DB_ENTRY
from src.libs.helpers.tui_helpers.size_bytes_helpers import format_size_bytes

from src import logger


def _unpack_data(msg: dict) -> tuple[str, str, int, Any, Any]:
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
    Contains the content sent from the user, show widget if attachments was
    attached in that conversation turn.
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
            widgets[0] = Static(unfmt_prompt)  # Replace the prompt with unformatted prompt

    return Vertical(*widgets, classes=f"msg-box user-box")


def create_assistant_message_box(
    cont_widget: Static | Markdown,
    tol_tkns: int,
    slash_cmd_metadata: dict | None = None,
    hide_tkns: bool = False,
) -> tuple[Vertical, Static]:
    """Contains the assistant's response and token count for said response."""
    # Extra widget to indicate auto compression summary
    is_auto_sum = bool(
        slash_cmd_metadata and slash_cmd_metadata.get("method") == "auto"
    )

    tkn_widget = Static(
        f"{tol_tkns} token used", classes="msg-token-count"
    )
    if hide_tkns:
        tkn_widget.add_class("hidden")  # Only reveal when the response is complete

    classes = "msg-box assistant-box"
    if is_auto_sum:
        classes += " auto-summary"

    box = Vertical(cont_widget, tkn_widget, classes=classes)

    if is_auto_sum:
        box.border_title = "AUTO SESSION SUMMARY DUE TO TOKEN LIMIT REACHED"

    return box, tkn_widget


def load_chat_history(screen) -> None:
    """
    Get entire chat history from the database and
    display it on to the interface.
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

