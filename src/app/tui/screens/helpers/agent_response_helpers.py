from pathlib import Path

from textual import work
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static, Input, Markdown, Button

from src.core import Agent
from src.slash_commands import slash_commands_helpers, slash_commands_dictionary
from src.app.tui.screens.helpers import size_bytes_helpers


SLASH_CMD_DICT = slash_commands_dictionary.slash_cmds_dict


def show_loading_indicator(screen, md_widget: Markdown, info: str):
    screen.app.call_from_thread(md_widget.update, info)


def end_loading_and_show_content(
    screen, md_widget: Markdown, full_txt, chat_container: VerticalScroll
):
    screen.app.call_from_thread(md_widget.update, full_txt)
    screen.app.call_from_thread(chat_container.scroll_end, animate=False)


def slash_command_if_called_start_behaviour(screen, prompt: str, md_widget: Markdown):
    cmd, user_prompt = slash_commands_helpers.detect_cmd(prompt)

    if cmd:
        if cmd == SLASH_CMD_DICT["compress"].get("cmd"):
            show_loading_indicator(
                screen=screen,
                md_widget=md_widget,
                info=SLASH_CMD_DICT["compress"].get("a") or ""
            )


def _update_user_message_box(
    screen, attchmnt_metadata: dict | None, user_msg_box: Vertical
):
    def _reveal_user_attachment_widget(
        attchmnt_metadata: dict | None, user_msg_box: Vertical
    ) -> None:
        if attchmnt_metadata:
            for filename, metadata in attchmnt_metadata.items():
                size = metadata.get("size_bytes")
                fmt_size = size_bytes_helpers.format_size_bytes(size)
                user_msg_box.mount(
                    Static(f"{filename} ({fmt_size})", classes="user-attachments")
                )
    screen.app.call_from_thread(_reveal_user_attachment_widget)


def _update_assistant_message_box(screen, tkn_wid: Static, msg_tol_tkns: int):
    def _reveal_token_count_widget() -> None:
        tkn_wid.update(f"{msg_tol_tkns} tokens used")
        tkn_wid.remove_class("hidden")
    screen.app.call_from_thread(_reveal_token_count_widget)


def fetch_agent_response(
    screen,
    prompt: str,
    md_widget: Markdown,
    user_msg_box: Vertical,
    chat_container: VerticalScroll,
    assist_msg_box: tuple[Vertical, Static],
    pending_attchmnt: list[Path],
    agent: Agent
) -> None:
    """
    Stream agent response in Markdown format in the newly created
    assistant message box; update token count and reset pending
    status when done. Call ONCE when the user submits the prompt.
    """
    # Assistant message box
    assistant_box, tkn_wid = assist_msg_box
    tkn_wid.add_class("hidden") # Hide widget box until entire response is generated
    screen.app.call_from_thread(chat_container.mount, assistant_box)

    show_loading_indicator(screen=screen, md_widget=md_widget, info="Thinking...")

    full_txt = ""
    first_tkn_received = False # Checker for when to swap spinner to message box

    def on_token(tkn: str) -> None:
        """Remove loading indicator when first token is received, then start token streaming."""
        nonlocal full_txt, first_tkn_received

        # If checker is still false
        if not first_tkn_received:
            full_txt = ""
            first_tkn_received = True

        # Stream tokens and append to full text varriable
        full_txt += tkn
        end_loading_and_show_content(
            screen=screen, md_widget=md_widget, full_txt=full_txt, chat_container=chat_container
        )

    attchmnts = pending_attchmnt or None
    result = agent.ask(
        prompt=prompt,
        callback=on_token,
        is_attchmnt=bool(attchmnts),
        paths=attchmnts,
        in_tui=True,
        screen=screen,
        md_widget=md_widget
    )

    # Update number of tokens used from current response,
    # reveal widget agent finished responding.
    if result:
        # Format from agent.ask() result
        _, p_tkns, o_tkns, metadata = result
        msg_tol_tkns = p_tkns + o_tkns
        attchmnt_metadata = metadata.get("attachments") if metadata else None

        _update_user_message_box(screen=screen, attchmnt_metadata=attchmnt_metadata, user_msg_box=user_msg_box)
        _update_assistant_message_box(screen=screen, tkn_wid=tkn_wid, msg_tol_tkns=msg_tol_tkns)

    # Reset pending attachment status
    screen._reset_user_section_widgets()
