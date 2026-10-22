from pathlib import Path
from tokenizers import Tokenizer
from datetime import datetime
from functools import lru_cache
from typing import Any, Literal

from src.config.config_toml import FALLBACK_TOKENIZER, MODEL_MAX_TOKENS
from src.config.files_and_directories import TOKENIZERS_DIR

from src.libs.helpers.internet_helpers import is_connected
from src.libs.models_database import MODEL_MAX
from src.libs.custom_dataclasses import AgentHooks
from src.libs.huggingface_repo import MODEL_NAME_TO_HF_TOKENIZER

from src.agent.models import ollama
from src.logger import app_logger

app_log = app_logger(__name__)


@lru_cache(maxsize=None)
def _load_tokenizer(
    model: str,
) -> Tokenizer | None:
    """Load local installed tokenizer."""
    file = TOKENIZERS_DIR / f"{model}.json"

    try:
        # Load pre-installed tokenizer
        return Tokenizer.from_file(str(file))

    except Exception as e:
        # Load pre-installed fallback tokenizer
        try:
            tknizr = Tokenizer.from_file(
                str(TOKENIZERS_DIR / f"{FALLBACK_TOKENIZER}.json")
            )
            app_log.warning(
                "Failed to load model '%s' tokenizer: %s. "
                "Falling back to model '%s' tokenizer" % (model, e, FALLBACK_TOKENIZER)
            )
            return tknizr

        # Disable tokenizer feature
        except Exception as fallback_err:
            app_log.warning(
                "Failed to load model '%s' fallback tokenizer: %s. "
                "Tokenizer feature disabled" % (FALLBACK_TOKENIZER, fallback_err)
            )
            return


class LocalTokenizer:
    """Load installed tokenizer."""

    def __init__(self, model: str):
        self.model = model
        self.load_tknizr = _load_tokenizer(model)

        # User specified max tokens
        if MODEL_MAX_TOKENS:
            self.model_max_tkns = MODEL_MAX_TOKENS
            app_log.debug(
                "Imported user specified max token limit " "for model '%s' tokenizer",
                self.model,
            )

        # Fallback to preset list
        elif self.model in MODEL_MAX:
            self.model_max_tkns = MODEL_MAX[self.model]
            app_log.debug(
                "Max token limit not specified by user. "
                "Loaded preset limits for model '%s' tokenizer",
                self.model,
            )

        # Neither option is available
        else:
            app_log.warning(
                "Model '%s' not in 'MODEL_MAX' and no max tokens provided. "
                "Add it to 'MODEL_MAX' or set 'chat_max tokens' in "
                "'~/.agent_app/config.toml'.",
                self.model,
            )
            self.model_max_tkns = None

    # --- Count tokens --------------------------------------------------------

    def count_string_tokens(self, text: str) -> int | None:
        """Counts tokens in string."""
        if not self.load_tknizr:
            app_log.warning(
                "Unable to count tokens: Tokenizer feature has been disabled"
            )
            return
        app_log.debug("Counting tokens from the provided text")
        return len(self.load_tknizr.encode(text))

    def count_history_tokens(self, msgs: list[dict]) -> int | None:
        """Calculates total token count in the message list."""
        if not self.load_tknizr:
            app_log.warning(
                "Unable to count tokens: Tokenizer feature has been disabled"
            )
            return None

        texts = [msg.get("content") or "" for msg in msgs]
        encodings = self.load_tknizr.encode_batch(texts)
        return sum(len(e.ids) for e in encodings) + 4 * len(msgs) + 3

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough estimate (about 4 characters per tokens)."""
        return max(1, len(text) // 4)

    def count_or_estimate_tokens(self, text: str) -> int:
        if not self.load_tknizr:
            return self.estimate_tokens(text)
        return len(self.load_tknizr.encode(text))

    # --- Encode --------------------------------------------------------------

    def encode_text(
        self, txt: str, hooks: AgentHooks | None = None
    ) -> list[int] | None:
        """Encode text using the loaded tokenizer."""
        if not self.load_tknizr:
            warn_str = "Unable to encode text: Tokenizer feature has been disabled"
            app_log.warning(warn_str)
            if hooks and hooks.on_stat_change:
                hooks.on_stat_change(warn_str)
            return

        if hooks and hooks.on_loading:
            info_str = "Encoding from provided text"
            hooks.on_loading(info_str)
        return self.load_tknizr.encode(txt)
