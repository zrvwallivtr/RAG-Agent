from src.logger import app_logger


app_log = app_logger(f"{__name__}.app")


def detect_cmd(prompt: str) -> tuple[str | None, str]:
    """Extracts shortcut if detected."""
    question_trimmed = prompt.strip()

    if question_trimmed.startswith(f"/"):
        parts           = question_trimmed.split(" ", 1)
        cmd             = parts[0]
        cleaned_text    = parts[1].strip() if len(parts) > 1 else ""
        app_log.debug("Command detected: %s", cmd)
        return cmd, cleaned_text

    return None, prompt
