import json
from pathlib import Path

from .interface import safe_print

from src.config.files_and_directories import APP_LOG_FILE, PROMPT_LOG_FILE


def structured_app_log(path: Path = APP_LOG_FILE) -> None:
    """Print structured app logs to the terminal."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                log = json.loads(line)
            except json.JSONDecodeError:
                continue

            timestamp = log.get("timestamp", "")
            level = log.get("level", "INFO")
            message = log.get("message", "")

            level_color = {
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
            }.get(level, "white")

            fmtted_line = (
                f"[dim]{timestamp}[/] | "
                f"[{level_color} bold]{level}[/] | "
                f"{message}"
            )
            safe_print(fmtted_line)

def structured_prompt_log(path: Path = PROMPT_LOG_FILE) -> None:
    """Print structured prompt logs to the terminal."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                log = json.loads(line)
            except json.JSONDecodeError:
                continue

            timestamp = log.get("timestamp", "")
            level = log.get("level", "INFO")
            message = log.get("message", "")

            level_color = {
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
            }.get(level, "white")

            fmtted_line = (
                f"[dim]{timestamp}[/] | "
                f"[{level_color} bold]{level}[/] | "
                f"{message}"
            )
            safe_print(fmtted_line)
