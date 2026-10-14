import sys
import os
from typing import Any
from pathlib import Path

from src.libs.models_database import MODEL_MAX
from src.config.files_and_directories import CONFIG_FILE, _cfg


def _model_not_found(
    config_file: Path,
    model_type: str,
    model_max_tkns: str,
):
    """
    Error message for when the model specified in 'config.toml'
    does not match any of models on MODEL_MAX list in
    'src/agent/tokens_handler.py'.
    """
    err_msg = (
        "Error in '%s':\n"
        "Model '%s' does not match any in the existing database. "
        "Please change the model name or add '%s' value."
        % (config_file, model_type, model_max_tkns)
    )
    print(err_msg)
    sys.exit(1)


def _check_if_value_is_valid(config_file: Path, field_name: str, value: Any):
    """Check if the value is a number in the field."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        err_msg = (
            "Error in '%s':\n"
            "'%s' must be a number, got '%s' instead."
            % (config_file, field_name, value)
        )
        print(err_msg)
        sys.exit(1)


def _normalise_to_integer(value: Any) -> int | str | None:
    """Converts valid values to int."""
    if value is None:
        return None

    try:
        return int(value)
    except (ValueError, TypeError):
        return (
            "Error: Invalid value '%s'"
            % value
        )


# --- Set model provider ------------------------------------------------------

OLLAMA_HOST = _cfg["models"]["ollama_host"]  # URL for ollama docker service


# --- Model values ------------------------------------------------------------

def _set_model_tokens(
    field: str, field_name: str, config_file: Path = CONFIG_FILE, 
) -> tuple[str, Any]:
    """Verify if 'model_name' is on the list or specified in 'config.toml'."""
    model               = _cfg["models"][field]
    max_tokens          = _cfg["models"].get(field_name, None)
    model_max_tokens    = _normalise_to_integer(max_tokens)

    if model not in MODEL_MAX:
        if model_max_tokens is None:
            _model_not_found(config_file, field, field_name)
        _check_if_value_is_valid(config_file, field_name, model_max_tokens)

    return model, model_max_tokens


MODEL, MODEL_MAX_TOKENS = _set_model_tokens(
    field="model", field_name="model_max_tokens"
)

EMBED_MODEL, EMBED_MAX_TOKENS = _set_model_tokens(
    field="embedding", field_name="embedding_max_tokens"
)


# --- Tokens ------------------------------------------------------------------

# Set fallback tokenizer
DEFAULT_FALLBACK_TOKENIZER = "gpt2"
FALLBACK_TOKENIZER = (
    _cfg.get("models", {}).get("fallback_tokenizer")
    or DEFAULT_FALLBACK_TOKENIZER
)

# Set token reserve for agent
DEFAULT_TOKEN_RESERVE = 1000
TOKEN_RESERVE = (
    _cfg.get("models", {}).get("token_reserve")
    or DEFAULT_TOKEN_RESERVE
)
