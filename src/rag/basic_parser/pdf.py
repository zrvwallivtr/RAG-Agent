from pathlib import Path

from src.libs.constants import ParserResult
from src.logger import app_logger

app_log = app_logger(f"{__name__}.app")


def parse_pdf(path: Path) -> ParserResult | None:
    """Extracts pdf text content layout page by page."""
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            pages = [page.extract_text() for page in pdf.pages]
            pages = [p for p in pages if p]
            app_log.info(
                "Text content extracted from PDF file '%s' page by page: Pages=%d",
                path,
                len(pdf.pages),
            )
            return "\n\n".join(pages), "txt"

    except Exception as e:
        app_log.warning("Failed to read PDF file '%s': %s", path, e)
        return
