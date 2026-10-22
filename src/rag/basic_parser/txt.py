from pathlib import Path

from src.libs.constants import ParserResult
from src.logger import app_logger

app_log = app_logger(f"{__name__}.app")


def parse_txt(path: Path, file_type: str = "txt") -> ParserResult | None:
    """Reads plain text files using UTF-8 validation."""
    try:
        cont = path.read_text(encoding="utf-8", errors="ignore")
        clean_cont = "".join(c for c in cont if c.isprintable() or c in "\n\r\t")
        app_log.info(
            "Extracted '%s' file content in '%s' as plain text", file_type, path
        )
        return clean_cont, "txt"

    except Exception as e:
        app_log.warning("Failed to read file '%s': %s", path, e)
        return
