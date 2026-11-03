from pathlib import Path
from tokenizers import Tokenizer
from datetime import datetime
from functools import lru_cache
from typing import Any

from src.config.config_toml import FALLBACK_TOKENIZER
from src.config.files_and_directories import TOKENIZERS_DIR, MODEL_DICT, ModelDict

from src.libs.helpers.internet_helpers import is_connected
from src.libs.custom_dataclasses import AgentHooks
from src.libs.helpers.model_dictionary_helpers import (
    tokenizer_file,
    model_huggingface_repo,
)

from src.agent.models import ollama
from src.logger import app_logger

app_log = app_logger(__name__)

ollama_clt = ollama.ollama_clt


def install_tokenizers(hooks: AgentHooks | None = None) -> None:
    """
    Install the fallback tokenizer and tokenizers from model list,
    internet connection required.

    Parameters
    ----------
        hooks: Function hooks for cli/tui integration.
    """
    if not is_connected():
        warn_str = "Failed to install tokenizers: No internet connection"
        app_log.warning(warn_str)
        if hooks and hooks.on_stat_change:
            hooks.on_stat_change(warn_str)
        return

    _install_fallback_tokenizer(hooks=hooks)

    count = 0
    installed_models = ollama_clt.list().models
    for installed_model in installed_models:
        count += 1
        model = installed_model.model
        _install_tokenizers_from_huggingface(
            count=count,
            model=model,
            installed_models=installed_models,
            hooks=hooks,
        )


def fetch_all_installed_tokenizers(tknizr_dir: Path = TOKENIZERS_DIR) -> list[dict]:
    """
    Return the name and modified time of all installed tokenizers
    in the tokenizer directory.

    Parameters
    ----------
        tknizr_dir: The directory storing all the installed tokenizers.

    Returns
    -------
        - A list of dictionary about the installed tokenizers.
    """
    return [
        {
            "name": file.stem,
            "modified_at": datetime.fromtimestamp(file.stat().st_mtime),
        }
        for file in tknizr_dir.iterdir()
        if file.is_file()
    ]


def _install_fallback_tokenizer(
    fb_tknizr: str = FALLBACK_TOKENIZER,
    hooks: AgentHooks | None = None,
) -> None:
    """
    Install fallback tokenizer.

    Parameters
    ----------
        tknizr_dir: The directory storing all the installed tokenizers.
        fb_tknizr:  Name of the fallback tokenizer.
        hooks:      Function hooks for cli/tui integration.
    """
    fb_tknizr_path = tokenizer_file(fb_tknizr)

    # Skip: Tokenizer already installed
    if fb_tknizr_path.exists():
        info_str = (
            "Model '%s' tokenizer already saved in '%s' as fallback tokenizer"
            % (fb_tknizr, fb_tknizr_path)
        )
        app_log.debug(info_str)
        if hooks and hooks.on_stat_change:
            hooks.on_stat_change(info_str)
        return

    info_str = "Installing fallback tokenizer '%s'" % fb_tknizr
    if hooks and hooks.on_loading:
        hooks.on_loading(info_str)

    try:
        tknizr = Tokenizer.from_pretrained(fb_tknizr)
        tknizr.save(str(fb_tknizr_path))
        info_str = (
            "Model '%s' tokenizer has been installed and "
            "saved to '%s' as fallback tokenizer" % (fb_tknizr, fb_tknizr_path)
        )
        app_log.info(info_str)
        if hooks and hooks.on_stat_change:
            hooks.on_stat_change(info_str)
    except Exception as e:
        err_msg = (
            "Failed to install model '%s' tokenizer as fallback tokenizer: "
            "%s. Skipping" % (fb_tknizr, e)
        )
        app_log.warning(err_msg)
        if hooks and hooks.on_stat_change:
            hooks.on_stat_change(err_msg)


def _install_tokenizers_from_huggingface(
    count: int,
    model: str,
    installed_models: list[str],
    model_dict: ModelDict = MODEL_DICT,
    tknizr_dir: Path = TOKENIZERS_DIR,
    hooks: AgentHooks | None = None,
) -> None:
    """
    Install tokenizers from huggingface model list.

    Parameters
    ----------
        count:             Installing tokenizer counter.
        model:             Model name for the tokenizer.
        installed_models:  All installed models.
        model_dict:        Model's dictionary.
        model_tknizr_dict: Dictionary paring the model name to the huggingface
                           repo name.
        tknizr_dir:        The directory storing all the installed tokenizers.
        hooks:             Function hooks for cli/tui integration.
    """
    hf_repo = model_huggingface_repo(model)

    # Skip: Huggingface repo not found
    if hf_repo is None:
        if hooks and hooks.on_stat_change:
            warn_msg = (
                "No Hugging Face repo set for model '%s' "
                "(add it to the model dictionary). Skipping..." % model
            )
            app_log.warning(warn_msg)
            hooks.on_stat_change(warn_msg)
        return None

    tknizr_path = tokenizer_file(hf_repo)

    # Skip: Tokenizer already installed
    if tknizr_path.exists():
        if hooks and hooks.on_stat_change:
            info_str = "Model '%s' tokenizer already installed. Skipping" % model
            hooks.on_stat_change(info_str)
        return None

    try:
        if hooks and hooks.on_loading:
            info_str = "[%d/%d] Installing model '%s' tokenizer" % (
                count,
                len(installed_models),
                model,
            )
            hooks.on_loading(info_str)

        tknizr = Tokenizer.from_pretrained(hf_repo)
        tknizr.save(str(tknizr_path))  # Writes local file
        info_str = "'%s' tokenizer saved to '%s'" % (hf_repo, tknizr_path)
        app_log.info(info_str)
        if hooks and hooks.on_stat_change:
            hooks.on_stat_change(info_str)

    except Exception as e:
        err_msg = Exception(
            "Failed to install '%s' tokenizer: %s. Skipping" % (hf_repo, e)
        )
        app_log.warning(err_msg)
        if hooks and hooks.on_error:
            hooks.on_error(err_msg)
