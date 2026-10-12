import mimetypes
from pathlib import Path

from src.config.files_and_directories import UPLOAD_DIR

from src.logger import app_logger

app_log = app_logger(f"{__name__}.app")


def format_size_bytes(size_bytes: int) -> str:
    """Converts size bytes into a human readable string (KB, MB, GB)."""
    if size_bytes is None or size_bytes < 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    unit_idx = 0

    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1

    return f"{size:.2f} {units[unit_idx]}"


def unpack_file_metadata(path: Path) -> tuple[str, str, int] | None:
    """Return name, mime type and size bytes from given path."""
    if not path.exists():
        app_log.warning(
            "File '%s' is not in the upload directory '%s'",
            path.name,
            UPLOAD_DIR
        )
        return

    # Prevent duplicated name in the metadata
    name = path.name
    counter = 1

    mime, _ = mimetypes.guess_type(path)
    mime = mime or "unknown"
    size = path.stat().st_size
    app_log.debug("Metadata extracted from %s", path.name)
    return name, mime, size
