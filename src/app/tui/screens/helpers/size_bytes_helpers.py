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
