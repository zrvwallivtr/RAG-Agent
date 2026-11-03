"""
All the path constants stores here, which will be called by other
config modules.

NEVER IMPORT ANY OTHER MODULES INTO THIS FILE.
"""

import sys
import shutil
import tomllib
import ast
from pathlib import Path
from typing import Any

APP_NAME = "agent"

# Static assets
MAIN_DIR = (
    Path(sys._MEIPASS)
    if hasattr(sys, "_MEIPASS")
    else Path(__file__).resolve().parent.parent.parent
)
PROMPT_DIR = MAIN_DIR / "src" / "prompts"
DEFAULT_CONFIG_FILE = MAIN_DIR / "config.toml"
DEFAULT_MODEL_DICT_FILE = MAIN_DIR / "src" / "libs" / "model_dictionary"

# Application directory
APP_DIR = Path.home() / f".{APP_NAME}"
UPLOAD_DIR = APP_DIR / "uploads"

# Config
CONFIG_FILE = APP_DIR / "config.toml"
ENV_PATH = APP_DIR / ".env"

# Data
APP_DATA_DIR = APP_DIR / "data"
CUS_PROMPT_DIR = APP_DATA_DIR / "prompts"

# Model
TOKENIZERS_DIR = APP_DIR / "tokenizers"
MODEL_DICT_FILE = APP_DIR / "model_dictionary"

# Logs
LOG_DIR = APP_DIR / "logs"
APP_LOG_FILE = LOG_DIR / "app.log"
PROMPT_LOG_FILE = LOG_DIR / "prompts.log"


ModelDict = dict[str, dict[str, str | int]]


def _create_app_directories(dir_list: list[Path]):
    """Create app directories from the list."""
    for folder in dir_list:
        folder.mkdir(parents=True, exist_ok=True)


def _load_configs(
    config_file: Path = CONFIG_FILE, default_config: Path = DEFAULT_CONFIG_FILE
) -> dict[str, Any]:
    """
    Check if 'config.toml' exists in the app directory, if not
    copy the default config to the app directory.

    Fallback to the default config.

    Parameters
    ----------
        config_file:    User's custom 'config.toml'.
        default_config: The default 'config.toml'.

    Returns
    -------
        - The configurations.
    """
    if not config_file.exists() and default_config.exists():
        # Copy default 'config.toml' to '~/.agent_app/'
        shutil.copy(default_config, config_file)

    elif not config_file.exists():
        config_file = default_config  # Fallback to default

    return tomllib.loads(config_file.read_text(encoding="utf-8"))


def _load_models_dictionary(
    name: str = "MODELS",
    model_dict: Path = MODEL_DICT_FILE,
    default_model_dict: Path = DEFAULT_MODEL_DICT_FILE,
) -> ModelDict:
    """
    Read named dictionary from the user's model dictionary, creating
    the file from the default one if it doesn't exist yet.

    Raises KeyError if the file doesn't define 'name'.

    Parameters
    ----------
        name:               Name of the dictionary.
        model_dict:         User's custom model dictionary.
        default_model_dict: The default model dictionary.

    Returns
    -------
        - Models' dictionary.
    """
    if not model_dict.exists():
        if not default_model_dict.exists():
            raise FileNotFoundError("No model dictionary at '%s'" % default_model_dict)

        model_dict.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(default_model_dict, model_dict)

    tree = ast.parse(model_dict.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)

    raise KeyError("'%s' is not defined in '%s'" % (name, model_dict))


# Run on every app start up:
# - Create app directories.
# - Copy `config.toml` to app directory if not exist.

_create_app_directories(
    [APP_DIR, APP_DATA_DIR, CUS_PROMPT_DIR, TOKENIZERS_DIR, LOG_DIR]
)

CONFIG = _load_configs()

MODEL_DICT = _load_models_dictionary()
