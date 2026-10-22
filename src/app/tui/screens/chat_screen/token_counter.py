"""Contains the functions for the session used token indicator."""

from textual import work


def set_tokens_text(screen, text: str) -> None:
    """Store the token text and redraw the status bar. Main thread only."""
    screen.tokens_text = text
    screen.refresh_status()


# def create_session_used_tokens_widget(screen) -> str | None:
#     """
#     Returns session tokens used over the max token limit of the model.
#     """
#     if not screen.local_tknizr.model_max_tkns:
#         return
#
#     # Used tokens throughout the entire session chat history
#     used_tkns = screen.local_tknizr.count_history_tokens(
#         screen.chat_sess_logs.get_active_conversations()
#     )
#     if not used_tkns:
#         return
#
#     return f"{used_tkns}/{screen.local_tknizr.model_max_tkns} tokens"  # Widget string


def refresh_session_token_cache(screen) -> None:
    """
    Refresh session token used.

    Call only when chat history changes:
    - On mount.
    - After sending a message.
    - Model finished responding.
    """
    result = screen.chat_sess_logs.get_session_data()
    if not result:
        return
    _, _, screen._cached_sess_tkns = result


@work(exclusive=True, thread=True, group="token_count")
def _session_used_tokens_and_prompt_tokens_count(screen, prompt: str):
    if screen._awaiting_response or not screen.local_tknizr.model_max_tkns:
        return

    pending_tkns = screen.local_tknizr.count_string_tokens(text=prompt) or 0
    curr_tkns = screen._cached_sess_tkns + pending_tkns

    screen.app.call_from_thread(
        set_tokens_text,
        screen,
        f"{curr_tkns}/{screen.local_tknizr.model_max_tkns} session tokens",
    )


def debounced_token_count(screen, prompt: str) -> None:
    """
    Implements a tokenization debounce timer to prevent excessive calculation
    during rapid typing.
    """
    if hasattr(screen, "_token_timer") and screen._token_timer:
        screen._token_timer.stop()

    screen._token_timer = screen.set_timer(
        0.15, lambda: _session_used_tokens_and_prompt_tokens_count(screen, prompt)
    )


def static_update_session_used_tokens_widget(screen) -> None:
    screen._awaiting_response = False
    refresh_session_token_cache(screen)  # Execute cache refresh on the main thread
    set_tokens_text(
        screen,
        f"{screen._cached_sess_tkns}/{screen.local_tknizr.model_max_tkns} session tokens",
    )
