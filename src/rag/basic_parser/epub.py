from pathlib import Path
from bs4 import BeautifulSoup

from src.libs.constants import ParserResult
from src.logger import app_logger

app_log = app_logger(f"{__name__}.app")


def parse_epub(path: Path) -> ParserResult | None:
    """Extracts plain text blocks from internal EPUB XHTML document payloads."""
    try:
        from ebooklib import epub

        book = epub.read_epub(path)
        chapters_text = []

        import ebooklib

        for item in book.get_items():
            # EPUB structural content is split into internal document items
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "lxml")

                # Remove visual layout nodes that shouldn't feed context windows
                for junk in soup(["script", "style"]):
                    junk.decompose()

                plain_text = soup.get_text(separator="\n").strip()
                if plain_text:
                    chapters_text.append(plain_text)

        app_log.info(
            "Plain text blocks extracted from EPUB file '%s': Chapters=%d",
            path,
            len(chapters_text),
        )
        return "\n\n".join(chapters_text), "txt"

    except Exception as e:
        app_log.warning("Failed to read EPUB file '%s': %s", path, e)
        return
