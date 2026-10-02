from textual.containers import VerticalScroll
from textual.widgets import Markdown


def show_loading_indicator(
    screen,
    md_widget: Markdown,
    info: str,
    is_scroll_end: bool | None = None,
    chat_container: VerticalScroll | None = None,
):
    screen.app.call_from_thread(md_widget.update, info)

    # Go to bottom (optional)
    if is_scroll_end and chat_container:
        chat_container.scroll_end(animate=False)


def show_info(screen, info: str):
    screen.notify(info)


def end_loading_and_show_content(
    screen, md_widget: Markdown, full_txt, chat_container: VerticalScroll
):
    screen.app.call_from_thread(md_widget.update, full_txt)
    screen.app.call_from_thread(chat_container.scroll_end, animate=False)
