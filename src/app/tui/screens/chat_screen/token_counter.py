from textual import work
from textual.widgets import Static


# Features a live token count for showing the how many tokens will be
# in the current session if the prompt is submitted
# (session used tokens + prompt tokens).

def create_session_used_tokens_widget(screen) -> str | None:
    """
    Returns the string displaying the number of tokens used throughout
    the session against the max tokens limit of the active chat model.
    """
    if not screen.local_tknizr.model_max_tkns:
        return

    # Used tokens throughout the entire session chat history
    used_tkns = screen.local_tknizr.count_history_tokens(
        screen.chat_sess_logs.get_active_conversations()
    )
    if not used_tkns:
        return

    return f"{used_tkns}/{screen.local_tknizr.model_max_tkns} tokens"  # Widget string


def refresh_session_token_cache(screen) -> None:
    """
    Refresh the value for tokens used throughout session.
    Call ONLY WHEN CHAT HISTORY CHANGES:
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
    """
    Update of the session used tokens widget:
    - Session used tokens + current tokens count in prompt input.
    """
    if screen._awaiting_response or not screen.local_tknizr.model_max_tkns:
        return

    pending_tkns = screen.local_tknizr.count_string_tokens(text=prompt) or 0
    curr_tkns = screen._cached_sess_tkns + pending_tkns

    screen.app.call_from_thread(
        screen.query_one("#tokens-counter", Static).update,
        f"{curr_tkns}/{screen.local_tknizr.model_max_tkns} tokens"
    )

def debounced_token_count(screen, prompt: str) -> None:
    """
    Reset a short timer on every keystroke; only the LAST keystroke in
    a burst actually trigger tokenization (so rapid typing does not run
    it dozens of times per second).
    """
    if hasattr(screen, "_token_timer") and screen._token_timer:
        screen._token_timer.stop()

    screen._token_timer = screen.set_timer(
        0.15, lambda: _session_used_tokens_and_prompt_tokens_count(screen, prompt)
    )


def static_update_session_used_tokens_widget(screen) -> None:
    """
    Refresh session used token cache value, and update
    tokens counter widget in the user section.

    ONLY call this when:
    - On mount.
    - Model done responding.
    """
    screen._awaiting_response = False
    refresh_session_token_cache(screen)  # Execute cache refresh on the main thread

    # Update widget using cached value
    screen.query_one("#tokens-counter", Static).update(
        f"{screen._cached_sess_tkns}/{screen.local_tknizr.model_max_tkns} tokens"
    )

