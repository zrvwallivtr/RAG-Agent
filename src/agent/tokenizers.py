from pathlib import Path
from tokenizers import Tokenizer
from datetime import datetime
from functools import lru_cache

from src.config.config_toml import FALLBACK_TOKENIZER, MODEL_MAX_TOKENS
from src.config.files_and_directories import TOKENIZERS_DIR

from src.libs.helpers.internet_helpers import is_connected
from src.libs.models_database import MODEL_MAX
from src.libs.custom_dataclasses import AgentHooks

from src.agent.models import ollama
from src.logger import app_logger

MODEL_NAME_TO_HF_TOKENIZER = {
    # Mistral
    "mistral":              "mistralai/Mistral-7B-Instruct-v0.2",
    "dolphin-mistral":      "mistralai/Mistral-7B-Instruct-v0.2",
    "ministral_3b":         "ministral/Ministral-3b-instruct",

    # Llama
    "llama3.1":             "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.2":             "meta-llama/Llama-3.2-3B-Instruct",

    # Gemma
    "gemma3":               "google/gemma-3-4b-it",

    # Phi
    "dolphin-phi":          "microsoft/phi-2",

    # Nomic
    "nomic-embed-text":     "nomic-ai/nomic-embed-text-v1.5",
}

app_log = app_logger(__name__)

ollama_clt = ollama.ollama_clt


def _install_fallback_tokenizers(
    tknizr_dir: Path = TOKENIZERS_DIR,
    fb_tknizr: str = FALLBACK_TOKENIZER,
    hooks: AgentHooks | None = None
) -> None:
    """Install fallback tokenizer."""
    fb_tknizr_path = tknizr_dir / f"{fb_tknizr}.json"

    # Skip: Tokenizer already installed
    if fb_tknizr_path.exists():
        info_str = (
            "Model '%s' tokenizer already saved in '%s' as fallback tokenizer"
            % (fb_tknizr, fb_tknizr_path)
        )
        app_log.debug(info_str)
        if hooks and hooks.on_stat_change:
            hooks.on_stat_change(info_str)

    info_str = ("Installing fallback tokenizer '%s'" % fb_tknizr)
    if hooks and hooks.on_loading:
        hooks.on_loading(info_str)
    try:
        fb_tknizr = Tokenizer.from_pretrained(fb_tknizr)
        fb_tknizr.save(str(fb_tknizr_path))
        info_str = (
            "Model '%s' tokenizer has been installed and "
            "saved to '%s' as fallback tokenizer"
            % (FALLBACK_TOKENIZER, fb_tknizr_path)
        )
        app_log.info(info_str)
        if hooks and hooks.on_stat_change:
            hooks.on_stat_change(info_str)
    except Exception as e:
        err_msg = (
            "Failed to install model '%s' tokenizer as fallback tokenizer: "
            "%s. Skipping"
            % (FALLBACK_TOKENIZER, e)
        )
        app_log.warning(err_msg)
        if hooks and hooks.on_stat_change:
            hooks.on_stat_change(err_msg)


def _install_tokenizer_from_huggingface(
    count: int,
    model_name: str,
    installed_models: list[str],
    hooks: AgentHooks | None = None
) -> None:
    """Install tokenizers from huggingface model list."""
    base_name = model_name.split(":")[0]
    hf_repo = MODEL_NAME_TO_HF_TOKENIZER.get(base_name)
    tknizr_path = TOKENIZERS_DIR / f"{model_name}.json"

    # Skip: Huggingface repo not found
    if hf_repo is None:
        if hooks and hooks.on_stat_change:
            warn_str = (
                "No known tokenizer mapping for model '%s' (base '%s'). "
                "Skipping..."
                % (model_name, base_name)
            )
            hooks.on_stat_change(warn_str)
        return

    # Skip: Tokenizer already installed
    if tknizr_path.exists():
        if hooks and hooks.on_stat_change:
            info_str = (
                "Model '%s' tokenizer already installed. Skipping" 
                % model_name
            )
            hooks.on_stat_change(info_str)
        return

    try:
        if hooks and hooks.on_loading:
            info_str = (
                "[%d/%d] Installing model '%s' tokenizer"
                % (count, len(installed_models), model_name)
            )
            hooks.on_loading(info_str)

        tknizr = Tokenizer.from_pretrained(hf_repo)
        tknizr.save(str(tknizr_path))  # Writes local file
        info_str = (
            "Model '%s' tokenizer saved to '%s'" 
            % (model_name, tknizr_path)
        )
        app_log.info(info_str)
        if hooks and hooks.on_stat_change:
            hooks.on_stat_change(info_str)

    except Exception as e:
        err_msg = Exception(
            "Failed to install '%s' (repo=%s) tokenizer: %s. Skipping"
            % (model_name, hf_repo, e)
        )
        app_log.warning(err_msg)
        if hooks and hooks.on_error:
            hooks.on_error(err_msg)


def install_tokenizers(hooks: AgentHooks | None = None) -> None:
    """
    Install the fallback tokenizer and tokenizers from model list,
    internet connection required.
    """
    if not is_connected:
        warn_str = (
            "Failed to install tokenizers: No internet connection"
        )
        app_log.warning(warn_str)
        if hooks and hooks.on_stat_change:
            hooks.on_stat_change(warn_str)
        return

    _install_fallback_tokenizers(hooks=hooks)

    count = 0
    installed_models = ollama_clt.list().models
    for installed_model in installed_models:
        count += 1
        model_name = installed_model.model
        _install_tokenizer_from_huggingface(
            count=count,
            model_name=model_name,
            installed_models=installed_models,
            hooks=hooks,
        )


def fetch_all_installed_tokenizers() -> list[dict]:
    """
    Return the name and modified time of all installed tokenizers
    in the tokenizer directory.
    """
    return [
        {
            "name": file.stem,
            "modified_at": datetime.fromtimestamp(file.stat().st_mtime),
        }
        for file in TOKENIZERS_DIR.iterdir()
        if file.is_file()
    ]


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
                "Falling back to model '%s' tokenizer"
                % (model, e, FALLBACK_TOKENIZER)
            )
            return tknizr

        # Disable tokenizer feature
        except Exception as fallback_err:
            app_log.warning(
                "Failed to load model '%s' fallback tokenizer: %s. "
                "Tokenizer feature disabled"
                % (FALLBACK_TOKENIZER, fallback_err)
            )
            return


class LocalTokenizer:
    """Load installed tokenizer."""

    def __init__(self, model: str):
        self.model          = model
        self.load_tknizr    = _load_tokenizer(model)

        # User specified max tokens
        if MODEL_MAX_TOKENS:
            self.model_max_tkns = MODEL_MAX_TOKENS
            app_log.debug(
                "Imported user specified max token limit "
                "for model '%s' tokenizer",
                self.model
            )

        # Fallback to preset list
        elif self.model in MODEL_MAX:
            self.model_max_tkns = MODEL_MAX[self.model]
            app_log.debug(
                "Max token limit not specified by user. "
                "Loaded preset limits for model '%s' tokenizer",
                self.model
            )

        # Neither option is available
        else:
            app_log.warning(
                "Model '%s' not in 'MODEL_MAX' and no max tokens provided. "
                "Add it to 'MODEL_MAX' or set 'chat_max tokens' in "
                "'~/.agent_app/config.toml'.",
                self.model
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

    # --- Encode --------------------------------------------------------------

    def encode_text(
        self,
        txt: str,
        hooks: AgentHooks | None = None
    ) -> list[int] | None:
        """Encode text using the loaded tokenizer."""
        if not self.load_tknizr:
            warn_str = (
                "Unable to encode text: Tokenizer feature has been disabled"
            )
            app_log.warning(warn_str)
            if hooks and hooks.on_stat_change:
                hooks.on_stat_change(warn_str)
            return

        if hooks and hooks.on_loading:
            info_str = ("Encoding from provided text")
            hooks.on_loading(info_str)
        return self.load_tknizr.encode(txt)
