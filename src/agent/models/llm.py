"""Contains all functions for how the models behaves, and read context."""

import ollama
from typing import Callable

from src.libs.constants import ModelResult
from src.libs.custom_dataclasses import AgentHooks

from src.agent.models.ollama import ollama_clt
from src.logger import app_logger, prompt_logger

app_log = app_logger(f"{__name__}.app")
prompt_log = prompt_logger(f"{__name__}.prompt")


def system_prompt(cont: str) -> dict:
    """
    Parameters
    ----------
        cont: System prompt string.

    Returns
    -------
        - LLM friendly format.
    """
    system_entry = {"role": "system", "content": cont}
    return system_entry


def assistant_message(cont: str) -> dict:
    """
    Parameters
    ----------
        cont: Assistant response string.

    Returns
    -------
        - LLM friendly format.
    """
    assistant_entry = {"role": "assistant", "content": cont}
    return assistant_entry


def user_message(cont: str) -> dict:
    """
    Parameters
    ----------
        cont: User's prompt string.

    Returns
    -------
        - LLM friendly format.
    """
    user_entry = {"role": "user", "content": cont}
    return user_entry


def model_response(
    model: str,
    msgs: list[dict],
    hooks: AgentHooks | None = None,
) -> ModelResult | None:
    """
    Response using specified model, streams output via AgentHooks.

    Parameters
    ----------
        model: Model name.
        msgs:  Messages for context.
        hooks: Function hooks for cli/tui integration.

    Returns
    -------
        - Model's response.
        - Prompt tokens.
        - Output tokens.
        - Stop.
    """
    if msgs:
        prompt_log.info(msgs[-1].get("content", ""))

    text, p_tkns, o_tkns, is_stopped = _stream_tokens(model, msgs, hooks=hooks)

    if not text:
        warn_str = "Model failed to generate response from the provided messages"
        app_log.warning(warn_str)
        if hooks and hooks.on_stat_change:
            hooks.on_stat_change(warn_str)
        if hooks and hooks.on_error:
            hooks.on_error(RuntimeError(warn_str))
        return None

    if hooks and hooks.on_complete:
        hooks.on_complete({"prompt_tokens": p_tkns, "output_tokens": o_tkns})

    return text, p_tkns, o_tkns, is_stopped


def model_response_with_new_system_prompt_and_context(
    model: str,
    sys_prompt: str,
    prompt: str | None,
    context: list[dict] | None = None,
    hooks: AgentHooks | None = None,
) -> ModelResult | None:
    """
    Combines a system prompt, context and question with model response.

    Parameters
    ----------
        model:      Model name.
        sys_prompt: System prompt.
        prompt:     User's prompt.
        context:    Context messages.
        hooks:      Function hooks for cli/tui integration.

    Returns
    -------
        - Model's response.
        - Prompt tokens.
        - Output tokens.
        - Stop.
    """
    msgs = _build_messages(sys_prompt, context, prompt)

    prompt_log.info(sys_prompt)
    prompt_log.info(msgs[-1].get("content", ""))

    try:
        text, p_tkns, o_tkns, is_stopped = _stream_tokens(model, msgs, hooks=hooks)
    except Exception as e:
        app_log.warning(
            "Model failed to generate response from the provided messages: %s",
            e,
            exc_info=True,
        )
        warn_str = "Model failed to generate response from the provided messages"
        if hooks and hooks.on_stat_change:
            hooks.on_stat_change(warn_str)
        if hooks and hooks.on_error:
            hooks.on_error(e)
        return None

    if hooks and hooks.on_complete:
        hooks.on_complete({"prompt_tokens": p_tkns, "output_tokens": o_tkns})

    return text, p_tkns, o_tkns, is_stopped


def model_response_memory_recall_format(
    model: str,
    sys_prompt: str,
    prompt: str,
    context: list[dict] | None = None,
    hooks: AgentHooks | None = None,
) -> ModelResult | None:
    """
    Model reponse with memory recall system prompt and format.

    Parameters
    ----------
        model:      Model name.
        sys_prompt: System prompt.
        prompt:     User's prompt.
        context:    Context messages.
        hooks:      Function hooks for cli/tui integration.

    Returns
    -------
        - Model's response.
        - Prompt tokens.
        - Output tokens.
        - Stop.
    """
    msgs = _build_messages(sys_prompt, context, prompt)
    prompt_log.info(sys_prompt)
    return model_response(model, msgs, hooks=hooks)


def _build_messages(
    sys_prompt: str, msgs: list[dict] | None, prompt: str | None
) -> list[dict]:
    """
    System prompt (old system prompts removed) + msgs + new user prompt.

    Parameters
    ----------
        sys_prompt: System prompt.
        msg:        Messages for context.
        prompt:     User's prompt.

    Returns
    -------
        - Messages list with the new system prompt.
    """
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
    hooks: AgentHooks | None = None,
) -> tuple[str, int, int, bool]:
    """
    Sends messages to the model and calls callback with each piece of text
    as it arrives.

    Parameters
    ----------
        model: Model name.
        msgs:  Messages for context.
        hooks: Function hooks for cli/tui integration.

    Returns
    -------
        - Full model response text.
        - Prompt tokens.
        - Output tokens.
        - Stop.
    """
    text = ""
    p_tkns = 0
    o_tkns = 0
    is_stopped = False
    first_tkn_fired = False

    stream = ollama_clt.chat(model=model, messages=msgs, stream=True)
    try:
        for chnk in stream:
            piece = chnk.message.content or ""
            if piece:
                # Trigger on_first_tkn callback on the first non-empty chunk
                if not first_tkn_fired:
                    first_tkn_fired = True
                    if hooks and hooks.on_first_tkn:
                        hooks.on_first_tkn()

                # Trigger on_tkn callback for incoming piece
                if hooks and hooks.on_tkn:
                    if hooks.on_tkn(piece) is False:
                        is_stopped = True
                        break
                elif not hooks or not hooks.on_tkn:
                    _print_token(piece)

                text += piece

            # The last chunk carries the token counts
            if chnk.done:
                p_tkns = chnk.prompt_eval_count or 0
                o_tkns = chnk.eval_count or 0

        if not hooks or not hooks.on_tkn:
            print()

    finally:
        close = getattr(stream, "close", None)  # Drops connection to Ollama
        if close:
            close()

    return text, p_tkns, o_tkns, is_stopped


def _print_token(tkn: str) -> None:
    """
    Parameters
    ----------
        tkn: Tokens
    """
    print(tkn, end="", flush=True)


def _ignore_token(tkn: str) -> None:
    """
    Parameters
    ----------
        tkn: Tokens
    """
    pass
