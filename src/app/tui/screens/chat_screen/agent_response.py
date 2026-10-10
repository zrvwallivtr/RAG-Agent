import time
from pathlib import Path

from textual import work
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static, Markdown, Button

from .interface import show_loading_indicator, end_loading_and_show_content, show_info
from .token_counter import static_update_session_used_tokens_widget
from .pending_attachments import reset_pending_attachment_widget
from .chat_container import create_assistant_message_box

from src.libs.helpers.tui_helpers.size_bytes_helpers import format_size_bytes
from src.libs.constants import COMPRESS_CMD
from src.libs.custom_dataclasses import AgentHooks

from src.core import Agent


def _update_user_message_box(
    screen, attchmnt_metadata: dict | None, user_msg_box: Vertical
):
    if not attchmnt_metadata:
        return

    def _reveal_user_attachment_widget() -> None:
        for filename, metadata in attchmnt_metadata.items():
            size = metadata.get("size_bytes")
            fmt_size = format_size_bytes(size)
            user_msg_box.mount(
                Static(f"{filename} ({fmt_size})", classes="user-attachments")
            )

    screen.app.call_from_thread(_reveal_user_attachment_widget)


def _update_assistant_message_box(screen, tkn_wid: Static, msg_tol_tkns: int):
    def _reveal_token_count_widget() -> None:
        tkn_wid.update(f"{msg_tol_tkns} tokens used")
        tkn_wid.remove_class("hidden")

    screen.app.call_from_thread(_reveal_token_count_widget)


def _fetch_agent_response(
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
    Stream agent response in Markdown format into the assistant message box.
    Updates token counts and resets attachment state upon completion via AgentHooks.
    """
    # Assistant message box
    assistant_box, tkn_wid = assist_msg_box
    tkn_wid.add_class("hidden") # Hide widget box until entire response is generated

    def _mount_initial_box():
        if not assistant_box.is_attached:
            chat_container.mount(assistant_box)
            chat_container.scroll_end(animate=False)

    screen.app.call_from_thread(_mount_initial_box)

    show_loading_indicator(
        screen=screen, md_widget=md_widget, info="Thinking..."
    )

    full_text = ""
    first_tkn_received = False # Checker for when to swap spinner to message box
    last_update_time = 0.0

    def on_first_token() -> None:
        nonlocal first_tkn_received
        first_tkn_received = True

    def on_tkn(tkn: str) -> bool | None:
        """
        Remove loading indicator when first token is received,
        then start token streaming.
        """
        nonlocal full_text, last_update_time

        if screen.stop_event.is_set():
            return False  # Signals backend stream to stop

        full_text += tkn
        now = time.time()

        if now - last_update_time >= 0.05:
            last_update_time = now
            end_loading_and_show_content(
                screen=screen,
                md_widget=md_widget,
                full_text=full_text,
                chat_container=chat_container
            )

        return True

    def on_loading(info: str) -> None:
        show_loading_indicator(
            screen=screen, md_widget=md_widget, info=info
        )

    def on_stat_change(msg: str) -> None:
        show_info(screen, msg)

    def on_error(err: Exception) -> None:
        screen.app.call_from_thread(
            screen.notify, f"Error: {err}", severity="error"
        )

    cmp_md: Markdown | None = None
    cmp_box: Vertical | None = None
    cmp_tkn_wid: Static | None = None
    cmp_text = ""
    cmp_last_update = 0.0
    cmp_mounted = False

    def on_cmp_start() -> None:
        """Build the summary box only. It is mounted when the first summary token arrives."""
        nonlocal cmp_md, cmp_box, cmp_tkn_wid, cmp_text, cmp_last_update, cmp_mounted
        cmp_text = ""
        cmp_last_update = 0.0
        cmp_mounted = False
        cmp_md = Markdown("")
        cmp_box, cmp_tkn_wid = create_assistant_message_box(
            cmp_md, 0, slash_cmd_metadata={"method": "auto"}, hide_tkns=True
        )
        # cmp_md, cmp_tkn_wid = md, new_tkn_wid

        # def _mount():
        #     chat_container.mount(cmp_box, before=user_msg_box)
        #     chat_container.scroll_end(animate=False)

        # screen.app.call_from_thread(_mount)

    def on_cmp_tkn(tkn: str) -> bool | None:
        nonlocal cmp_text, cmp_last_update, cmp_mounted
        md, box = cmp_md, cmp_box
        if md is None or box is None:
            return None
        if screen.stop_event.is_set():
            return False

        cmp_text += tkn

        # First summary token: mount the box under the user message box
        if not cmp_mounted:
            cmp_mounted = True
            text = cmp_text

            async def _mount_summary():
                await chat_container.mount(box, before=assistant_box)
                md.update(text)
                assistant_box.display = False  # Hide assistant box (and loading widget) while summary streams
                chat_container.scroll_end(animate=False)

            screen.app.call_from_thread(_mount_summary)
            return True

        now = time.time()
        if now - cmp_last_update >= 0.05:
            cmp_last_update = now
            screen.app.call_from_thread(md.update, cmp_text)
        return True

    def on_cmp_done(tot_tkns: int) -> None:
        md, wid, text = cmp_md, cmp_tkn_wid, cmp_text
        if not cmp_mounted or md is None:
            return

        def _finish():
            md.update(text)
            if wid is not None:
                wid.update(f"{tot_tkns} tokens used")
                wid.remove_class("hidden")
            # Summary finished: bring the answer box back and show loading widget
            md_widget.update("Thinking...")
            assistant_box.display = True
            chat_container.scroll_end(animate=False)

        screen.app.call_from_thread(_finish)

    hooks = AgentHooks(
        on_tkn=on_tkn,
        on_first_tkn=on_first_token,
        on_loading=on_loading,
        on_stat_change=on_stat_change,
        on_error=on_error,
        on_cmp_start=on_cmp_start,
        on_cmp_tkn=on_cmp_tkn,
        on_cmp_done=on_cmp_done,
    )

    attchmnts = pending_attchmnt
    result = agent.ask(
        prompt=prompt,
        paths=attchmnts,
        is_attchmnt=bool(attchmnts),
        hooks=hooks,
    )
    screen.app.call_from_thread(setattr, assistant_box, "display", True)

    is_stopped = screen.stop_event.is_set()

    if result:
        ans, p_tkns, o_tkns, metadata = result

        if isinstance(ans, str):
            end_loading_and_show_content(
                screen=screen,
                md_widget=md_widget,
                full_text=ans,
                chat_container=chat_container,
            )

        msg_tol_tkns = p_tkns + o_tkns
        attchmnt_metadata = metadata.get("attachments") if metadata else None

        _update_user_message_box(
            screen=screen,
            attchmnt_metadata=attchmnt_metadata,
            user_msg_box=user_msg_box,
        )
        _update_assistant_message_box(
            screen=screen, tkn_wid=tkn_wid, msg_tol_tkns=msg_tol_tkns
        )

    elif is_stopped:
        end_loading_and_show_content(
            screen=screen,
            md_widget=md_widget,
            full_text="*(stopped)*" if is_stopped else "No response",
            chat_container=chat_container,
        )

    else:
        end_loading_and_show_content(
            screen=screen,
            md_widget=md_widget,
            full_text="No response",
            chat_container=chat_container,
        )

    screen.app.call_from_thread(reset_pending_attachment_widget, screen)


def _finish_response_ui(screen) -> None:
    static_update_session_used_tokens_widget(screen)
    screen.query_one("#stop-button", Button).add_class("hidden")


@work(exclusive=True, thread=True)
def run_fetch_agent_response(
    screen,
    prompt: str,
    md_widget: Markdown,
    user_msg_box: Vertical,
    chat_container: VerticalScroll,
    assist_msg_box: tuple[Vertical, Static]
):
    try:
        _fetch_agent_response(
            screen=screen,
            prompt=prompt,
            md_widget=md_widget,
            user_msg_box=user_msg_box,
            chat_container=chat_container,
            assist_msg_box=assist_msg_box,
            pending_attchmnt=screen.pending_attchmnt,
            agent=screen.agent
        )

    finally: # After model done responding
        screen.app.call_from_thread(_finish_response_ui, screen)
