from textual.containers import VerticalScroll
from textual.widgets import Markdown, Button


def show_loading_indicator(
    screen,
    md_widget: Markdown,
    info: str,
    is_scroll_end: bool | None = None,
    chat_container: VerticalScroll | None = None,
):
    """
    Replace the assistant message box for a custom loading string. Only
    call between the user prompt is submitted and model's first output token.
    """
    def _update():
        if md_widget.is_attached:
            md_widget.update(info)
            if is_scroll_end and chat_container is not None:
                chat_container.scroll_end(animate=False)

    screen.app.call_from_thread(_update)


def end_loading_and_show_content(
    screen, md_widget: Markdown, full_text, chat_container: VerticalScroll
):
    """
    Update the assistant message box to show model response. Only call after
    model start streaming tokens.
    """
    def _update():
        md_widget.update(full_text)
        if chat_container is not None:
            chat_container.scroll_end(animate=False)

    screen.app.call_from_thread(_update)


def show_info(screen, info: str):
    """Show TUI app info string."""
    screen.app.call_from_thread(screen.notify, info)


def request_stop(screen) -> None:
    """
    Stop button, on click stop any agent response.
    Only availabe when agent is generating response.
    """
    if not screen._awaiting_response:
        return
    screen.stop_event.set()
    screen.query_one("#stop-button", Button).disabled = True  # No double clicks
    screen.notify("Stopping...")
