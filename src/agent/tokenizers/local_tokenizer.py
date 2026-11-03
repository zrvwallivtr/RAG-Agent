from pathlib import Path
from tokenizers import Tokenizer
from datetime import datetime
from functools import lru_cache
from typing import Any, Literal

from src.config.config_toml import FALLBACK_TOKENIZER
from src.config.files_and_directories import TOKENIZERS_DIR, MODEL_DICT, ModelDict

from src.libs.helpers.internet_helpers import is_connected
from src.libs.custom_dataclasses import AgentHooks
from src.libs.helpers.model_dictionary_helpers import (
    tokenizer_file,
    model_context_window,
    model_huggingface_repo,
)

from src.agent.models import ollama
from src.logger import app_logger

app_log = app_logger(__name__)


class LocalTokenizer:
    """
    Load installed tokenizer.

    Parameters
    ----------
        model:          Model name.
        model_max_tkns: Model max tokens (context window).
        model_max:      User define model max tokens, required if
                        MODEL_MAX_TOKENS is not defined.
    """

    def __init__(
        self,
        model: str,
    ):
        self.model = model
        self.load_tknizr = load_tokenizer(model)
        self.model_max_tkns = _load_model_max_tokens(model)

    def count_string_tokens(self, text: str) -> int | None:
        """
        Counts tokens in string.

        Parameters
        ----------
            text: Text string for token count.

        Returns
        -------
            - Token used from the string.
        """
        if not self.load_tknizr:
            app_log.warning(
                "Unable to count tokens: Tokenizer feature has been disabled"
            )
            return
        app_log.debug("Counting tokens from the provided text")
        return len(self.load_tknizr.encode(text))

    def count_history_tokens(self, msgs: list[dict]) -> int | None:
        """
        Calculates total token count in the message list.

        Parameters
        ----------
            msgs: List of dictionary contain messages.

        Returns
        -------
            - Token used across the messages.
        """
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
        """
        Fallback, rough estimate (about 4 characters per tokens).

        Parameters
        ----------
            text: Text string for token extimation.

        Returns
        -------
            - Estimated token used from the string.
        """
        return max(1, len(text) // 4)

    def count_or_estimate_tokens(self, text: str) -> int:
        """
        Fallback to estimate tokens if tokenizer unavailable for
        critical features.

        Parameters
        ----------
            text: Text string for token count.

        Returns
        -------
            - Token used from the string.
        """
        if not self.load_tknizr:
            return self.estimate_tokens(text)
        return len(self.load_tknizr.encode(text))

    def encode_text(
        self, text: str, hooks: AgentHooks | None = None
    ) -> list[int] | None:
        """
        Encode text using the loaded tokenizer.

        Parameters
        ----------
            text:  Text string for encoding.
            hooks: Function hooks for cli/tui integration.

        Returns
        -------
            - Encodings.
        """
        if not self.load_tknizr:
            warn_str = "Unable to encode text: Tokenizer feature has been disabled"
            app_log.warning(warn_str)
            if hooks and hooks.on_stat_change:
                hooks.on_stat_change(warn_str)
            return

        if hooks and hooks.on_loading:
            info_str = "Encoding from provided text"
            hooks.on_loading(info_str)
        return self.load_tknizr.encode(text)


# @lru_cache(maxsize=None)
# def load_tokenizer(
#     model: str,
#     tknizr_dir: Path = TOKENIZERS_DIR,
#     fb_tknizr: str | None = FALLBACK_TOKENIZER,
# ) -> Tokenizer | None:
#     """
#     Load local installed tokenizer.
#
#     Parameters
#     ----------
#         model:      Model name.
#         tknizr_dir: The directory storing all the installed tokenizers.
#         fb_tknizr:  Fallback tokenizer's name.
#
#     Returns
#     -------
#         - The tokenizer class.
#     """
#     file = tknizr_dir / f"{model}.json"
#
#     try:
#         # Load pre-installed tokenizer
#         return Tokenizer.from_file(str(file))
#
#     except Exception as e:
#         # Load pre-installed fallback tokenizer
#         try:
#             tknizr = Tokenizer.from_file(str(tknizr_dir / f"{fb_tknizr}.json"))
#             app_log.warning(
#                 "Failed to load model '%s' tokenizer: %s. "
#                 "Falling back to model '%s' tokenizer" % (model, e, fb_tknizr)
#             )
#             return tknizr
#
#         # Disable tokenizer feature
#         except Exception as fallback_err:
#             app_log.warning(
#                 "Failed to load model '%s' fallback tokenizer: %s. "
#                 "Tokenizer feature disabled" % (fb_tknizr, fallback_err)
#             )
#             return


@lru_cache(maxsize=None)
def load_tokenizer(
    model: str,
    model_dict: ModelDict = MODEL_DICT,
    tknizr_dir: Path = TOKENIZERS_DIR,
    fb_tknizr: str | None = FALLBACK_TOKENIZER,
) -> Tokenizer | None:
    """
    Load local installed tokenizer with matching huggingface repo name.

    Parameters
    ----------
        model:      Model name.
        model_dict: Model's dictionary.
        tknizr_dir: The directory storing all the installed tokenizers.
        fb_tknizr:  Fallback tokenizer's name.

    Returns
    -------
        - The tokenizer class.
    """
    hf_repo = model_huggingface_repo(model)

    try:
        if not hf_repo:
            raise FileNotFoundError("No huggingface_repo set for model '%s'" % model)
        # Load tokenizer with matching huggingface repo name
        return Tokenizer.from_file(str(tokenizer_file(hf_repo)))
    except Exception as e:
        try:
            # Load fallback tokenizer
            tknizr = Tokenizer.from_file(str(tknizr_dir / f"{fb_tknizr}.json"))
            app_log.warning(
                "Failed to load model '%s' tokenizer: %s. "
                "Falling back to model '%s' tokenizer" % (model, e, fb_tknizr)
            )
            return tknizr

        except Exception as fallback_err:
            # Disable tokenizer feature
            app_log.warning(
                "Failed to load model '%s' fallback tokenizer: %s. "
                "Tokenizer feature disabled" % (fb_tknizr, fallback_err)
            )
            return None


def _load_model_max_tokens(
    model: str, model_dict: ModelDict = MODEL_DICT
) -> int | None:
    """
    Parameters
    ----------
        model:      Model name
        model_dict: Model's dictionary.

    Returns
    -------
        - Model's max tokens
    """
    model_max_tkns = model_context_window(model)

    if not model_max_tkns:
        app_log.warning(
            "Model '%s' not in 'MODEL_MAX' and no max tokens provided. "
            "Add it to 'MODEL_MAX' or set 'chat_max tokens' in "
            "'~/.agent_app/config.toml'.",
            model,
        )
    return model_max_tkns
