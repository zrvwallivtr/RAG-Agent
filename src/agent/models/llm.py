import ollama
from typing import Callable

from src.agent.models.ollama import ollama_clt
from src.logger import app_logger, prompt_logger

app_log     = app_logger(f"{__name__}.app")
prompt_log  = prompt_logger(f"{__name__}.prompt")

# Type hints
ModelResult = tuple[str, int, int, bool]
TokenCallback = Callable[[str], bool | None]


# Roles:
# - Structure string appropriate for said role

def system_prompt(content: str) -> dict:
    system_entry = {"role": "system", "content": content}
    return system_entry


def assistant_message(content: str) -> dict:
    assistant_entry = {"role": "assistant", "content": content}
    return assistant_entry


def user_message(content: str) -> dict:
    user_entry = {"role": "user", "content": content}
    return user_entry


def _build_messages(
    sys_prompt: str,
    msgs: list[dict] | None,
    prompt: str | None
) -> list[dict]:
    """System prompt (old system prompts removed) + msgs + new user prompt."""
    new_msgs = [system_prompt(sys_prompt)]

    if msgs:
        # Add existing message entries (OTHER THAN "system" role)
        # into the new list
        new_msgs += [msg for msg in msgs if msg["role"] != "system"]

    if prompt:
        new_msgs.append(user_message(prompt))

    return new_msgs


def _stream_tokens(
    model: str,
    msgs: list[dict],
    callback: TokenCallback | None = None
) -> tuple[str, int, int, bool]:
    """
    Sends messages to the model and calls callback with each piece of text
    as it arrives.
    Returns full text, prompt tokens, output tokens, stop
    """
    text = ""
    p_tkns = 0
    o_tkns = 0
    is_stopped = False

    stream = ollama_clt.chat(model=model, messages=msgs, stream=True)
    try:
        for chnk in stream:
            piece = chnk.message.content or ""
            if piece:
                if callback:
                    # Caller request to stop
                    if callback(piece) is False:
                        is_stopped = True
                        break
                text += piece

            # The last chunk carries the token counts
            if chnk.done:
                p_tkns = chnk.prompt_eval_count or 0
                o_tkns = chnk.eval_count or 0

    finally:
        close = getattr(stream, "close", None)  # Drops connection to Ollama
        if close:
            close()

    return text, p_tkns, o_tkns, is_stopped


def _print_token(tkn: str) -> None:
    print(tkn, end="", flush=True)


def _ignore_token(tkn: str) -> None:
    pass


# Response methods

def model_response(
    model: str,
    msgs: list[dict],
    callback: TokenCallback | None = None
) -> ModelResult | None:
    """Response using specified model, streams output."""
    if msgs:
        prompt_log.info(msgs[-1].get("content", ""))

    if callback:
        text, p_tkns, o_tkns, is_stopped = _stream_tokens(model, msgs, callback)
    else:
        text, p_tkns, o_tkns, is_stopped = _stream_tokens(model, msgs, _print_token)
        print()

    if not text:
        app_log.warning("Model failed to generate response from the provided messages")
        return None
    return text, p_tkns, o_tkns, is_stopped


def model_response_with_new_system_prompt_and_context(
    model: str,
    sys_prompt: str,
    prompt: str | None,
    context: list[dict] | None = None,
    callback: TokenCallback | None = None
) -> ModelResult | None:
    """Combines a system prompt, context and question with model response."""
    msgs = _build_messages(sys_prompt, context, prompt)

    prompt_log.info(sys_prompt)
    prompt_log.info(msgs[-1].get("content", ""))

    try:
        text, p_tkns, o_tkns, is_stopped = _stream_tokens(model, msgs, callback or _ignore_token)
    except Exception as e:
        app_log.warning(
            "Model failed to generate response from the provided messages: %s",
            e, exc_info=True
        )
        return None

    return text, p_tkns, o_tkns, is_stopped


def model_response_memory_recall_format(
    model: str,
    sys_prompt: str,
    prompt: str,
    context: list[dict] | None = None,
    callback: Callable[[str], None] | None = None
) -> ModelResult | None:
    """Model reponse with memory recall system prompt and format."""
    msgs = _build_messages(sys_prompt, context, prompt)

    prompt_log.info(sys_prompt)
    return model_response(model, msgs, callback)
