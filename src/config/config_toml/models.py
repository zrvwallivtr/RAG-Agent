import sys
import os
from typing import Any
from pathlib import Path
from src.config.files_and_directories import CONFIG_FILE, CONFIG


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
        err_msg = "Error in '%s':\n" "'%s' must be a number, got '%s' instead." % (
            config_file,
            field_name,
            value,
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
        return "Error: Invalid value '%s'" % value


def _set_model_tokens(
    field: str,
    field_name: str,
    config_file: Path = CONFIG_FILE,
) -> tuple[str, Any]:
    """Verify if 'model_name' is on the list or specified in 'config.toml'."""
    model = CONFIG["models"][field]
    max_tokens = CONFIG["models"].get(field_name, None)
    model_max_tokens = _normalise_to_integer(max_tokens)

    return model, model_max_tokens


# URL for Ollama docker service
OLLAMA_HOST: str = CONFIG["models"]["ollama_host"]

# Language model
# _MODEL_RESULT = _set_model_tokens(field="model", field_name="model_max_tokens")
# MODEL: str = _MODEL_RESULT[0]
MODEL: str = CONFIG.get("models", {}).get("model", None)
if not MODEL:
    raise ValueError("Error: Model not set in 'config.toml'")

# Embedding model
# _EMB_MODEL_RESULT = _set_model_tokens(
#     field="embedding", field_name="embedding_max_tokens"
# )
# EMBED_MODEL: str = _EMB_MODEL_RESULT[0]
EMBED_MODEL: str = CONFIG.get("models", {}).get("embedding", None)
if not EMBED_MODEL:
    raise ValueError("Error: Embedding model not set in 'config.toml'")

# Fallback tokenizer
_DEFAULT_FALLBACK_TOKENIZER: str = "gpt2"
FALLBACK_TOKENIZER: str = CONFIG.get("models", {}).get(
    "fallback_tokenizer", _DEFAULT_FALLBACK_TOKENIZER
)

# Token reserve for agent
_DEFAULT_TOKEN_RESERVE: int = 1000
TOKEN_RESERVE: int = CONFIG.get("models", {}).get("token_reserve", _DEFAULT_TOKEN_RESERVE)
