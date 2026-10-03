import sys
from pathlib import Path

from src.config.files_and_directories import PROMPT_DIR, CUS_PROMPT_DIR, CONFIG_FILE


def _prompt_path_handler(
    filename: str, config_file: Path, dir: Path
) -> tuple[Path, str]:
    """
    Check if file exist, then read file contents and return it's path
    and contents as string.
    """
    path = dir / filename
    if not path.exists():
        print(f"Error in {config_file}:")
        print(f"'{filename}' prompt file not found: '{path}'")
        sys.exit(1)

    prompt = (path).read_text(encoding="utf-8").strip()
    return path, prompt


def _prioritise_custom_prompt(
    filename: str, config_file: Path, custom_prompt_dir: Path, default_prompt_dir: Path
) -> tuple[Path, str]:
    """
    Returns active directory and prompt if exists. Else, return
    default directory and prompt.
    """
    custom_prompt_path = custom_prompt_dir / filename
    path = Path(custom_prompt_path)

    if path.exists():
        return _prompt_path_handler(filename, config_file, custom_prompt_dir)

    return _prompt_path_handler(filename, config_file, default_prompt_dir)


# System prompt directories
SYS_PROMPT_DIR      = PROMPT_DIR / "system"
CUS_SYS_PROMPT_DIR  = CUS_PROMPT_DIR / "system"

# Memory prompt directories
MEM_PROMPT_DIR      = PROMPT_DIR / "memory"
CUS_MEM_PROMPT_DIR  = CUS_PROMPT_DIR / "memory"

# Compression prompt directories
COMPRESS_PROMPT_DIR     = PROMPT_DIR / "chat_compression"
CUS_COMPRESS_PROMPT_DIR = CUS_PROMPT_DIR / "chat_compression"


# Base agent system prompt

SYS_PROMPT_PATH, SYS_PROMPT = _prioritise_custom_prompt(
    filename="standard",
    config_file=CONFIG_FILE,
    custom_prompt_dir=CUS_SYS_PROMPT_DIR,
    default_prompt_dir=SYS_PROMPT_DIR
)


# Compression agent system prompt

COMPRESS_PROMPT_PATH, COMPRESS_PROMPT = _prioritise_custom_prompt(
    filename="instructions",
    config_file=CONFIG_FILE,
    custom_prompt_dir=CUS_COMPRESS_PROMPT_DIR,
    default_prompt_dir=COMPRESS_PROMPT_DIR
)


# Memory agent system prompts

# Auto memory extraction
MEM_PROMPT_PATH, MEM_PROMPT = _prioritise_custom_prompt(
    filename="memory_agent",
    config_file=CONFIG_FILE,
    custom_prompt_dir=CUS_MEM_PROMPT_DIR,
    default_prompt_dir=MEM_PROMPT_DIR
)

# Manual memory extraction
MEM_MANUAL_PROMPT_PATH, MEM_MANUAL_PROMPT = _prioritise_custom_prompt(
    filename="manual_memory_extraction",
    config_file=CONFIG_FILE,
    custom_prompt_dir=CUS_MEM_PROMPT_DIR,
    default_prompt_dir=MEM_PROMPT_DIR
)

# Interpret retrieved memory entries
MEM_RECALL_INTERPRET_PATH, MEM_RECALL_INTERPRET_PROMPT = _prioritise_custom_prompt(
    filename="memory_recall_interpreter",
    config_file=CONFIG_FILE,
    custom_prompt_dir=CUS_MEM_PROMPT_DIR,
    default_prompt_dir=MEM_PROMPT_DIR
)
