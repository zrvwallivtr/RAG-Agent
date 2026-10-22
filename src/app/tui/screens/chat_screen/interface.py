"""Chat screen helpers for interface rendering and behaviours."""

# from textual.containers import VerticalScroll
# from textual.widgets import Markdown, Button, Input, Static
#
# from src.app.tui.screens.base_screen.command_bar import show_command_message
#
#
# def show_loading_indicator(
#     screen,
#     md_widget: Markdown,
#     info: str,
#     is_scroll_end: bool | None = None,
#     chat_container: VerticalScroll | None = None,
# ):
#     """
#     Shown in the chat container, it replaces the assistant message box
#     for a custom loading indicator.
#
#     Only call before the token starts streaming.
#     """
#     def _update():
#         if md_widget.is_attached:
#             md_widget.update(info)
#             if is_scroll_end and chat_container is not None:
#                 chat_container.scroll_end(animate=False)
#
#     screen.app.call_from_thread(_update)
#
#
# def show_markdown_content(
#     screen, md_widget: Markdown, full_text, chat_container: VerticalScroll
# ):
#     """
#     Update the assistant message box to show model response.
#
#     Only call on/after the first token is streamed.
#     """
#     def _update():
#         md_widget.update(full_text)
#         if chat_container is not None:
#             chat_container.scroll_end(animate=False)
#
#     screen.app.call_from_thread(_update)
#
#
# def show_information_in_command_bar(screen, info: str):
#     """Show infromation on the command bar with custom string."""
#     screen.app.call_from_thread(show_command_message, screen, info)


# def request_stop(screen) -> None:
#     """
#     Sets stop flags and disables the stop button to prevent double-click
#     race conditions.
#
#     Only call when agent is generating/streaming response.
#     """
#     if not screen._awaiting_response:
#         return
#     screen.stop_event.set()
#     screen.query_one("#stop-button", Button).disabled = True  # No double clicks
#     show_command_message(screen, "Stopping agent response...")
