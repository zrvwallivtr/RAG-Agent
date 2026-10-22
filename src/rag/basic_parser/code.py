from pathlib import Path

from .txt import parse_txt

from src.libs.constants import ParserResult


def parse_code(path: Path) -> ParserResult | None:
    """Reads code files and wraps them in markdown code fences."""
    lang = path.suffix.lstrip(".")

    result = parse_txt(path, f"{lang}")
    if not result:
        return
    content, _ = result

    md_code_block = f"```{lang}\n{content}\n```"
    return md_code_block, "txt"
