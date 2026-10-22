"""
Contains all the code for the basic parser behaviour, do not import this
module directly, all parser tools route through the 'file_parser' package.
"""

from pathlib import Path
from bs4 import BeautifulSoup
from bs4.builder import register_treebuilders_from

from src.config.files_and_directories import UPLOAD_DIR

from .office import parse_docx, parse_xlsx
from .pdf import parse_pdf
from .epub import parse_epub
from .code import parse_code
from .txt import parse_txt

from src.libs.constants import ParserResult
from src.logger import app_logger

app_log = app_logger(f"{__name__}.app")


class BasicParser:
    def __init__(self):
        # Map formats to their parsing methods
        self.formats = {
            # PLAIN TEXT
            ".txt": self.read_txt,
            # Libreoffice
            # ".odt": self._read_odt,
            # ".ods": self._read_ods,
            # ".odp": self._read_odp,
            # Microsoft office
            ".docx": self._read_docx,
            ".xlsx": self._read_xlsx,
            # ".pptx": self._read_pptx,
            # Data & configuration formats
            ".csv": self._read_csv,
            ".yaml": self._read_yaml,
            ".yml": self._read_yml,
            ".toml": self._read_toml,
            ".xml": self._read_xml,
            # Text documents
            ".pdf": self._read_pdf,
            ".epub": self._read_epub,
            # Programming
            ".py": self._read_code,
            ".js": self._read_code,
            ".ts": self._read_code,
            ".tsx": self._read_code,
            ".json": self._read_code,
            ".md": self._read_code,
            ".sh": self._read_code,
            ".html": self._read_code,
            ".css": self._read_code,
            ".rs": self._read_code,
            ".go": self._read_code,
        }

    # --- Libreoffice ---------------------------------------------------------

    # --- Microsoft office ----------------------------------------------------

    def _read_docx(self, path: Path) -> ParserResult | None:
        return parse_docx(path)

    def _read_xlsx(self, path: Path) -> ParserResult | None:
        return parse_xlsx(path)

    # --- Plain text ----------------------------------------------------------

    def read_txt(self, path: Path, file_type: str = "txt") -> ParserResult | None:
        return parse_txt(path, file_type)

    # --- Data & configuration formats ----------------------------------------

    def _read_csv(self, path: Path) -> ParserResult | None:
        """Reads csv as plain text files."""
        return self.read_txt(path, "csv")

    def _read_yaml(self, path: Path) -> ParserResult | None:
        return self.read_txt(path, "yaml")

    def _read_yml(self, path: Path) -> ParserResult | None:
        return self.read_txt(path, "yml")

    def _read_toml(self, path: Path) -> ParserResult | None:
        return self.read_txt(path, "toml")

    def _read_xml(self, path: Path) -> ParserResult | None:
        return self.read_txt(path, "xml")

    # --- Text documents ------------------------------------------------------

    def _read_pdf(self, path: Path) -> ParserResult | None:
        return parse_pdf(path)

    def _read_epub(self, path: Path) -> ParserResult | None:
        return parse_epub(path)

    # --- Programming ---------------------------------------------------------

    def _read_code(self, path: Path) -> ParserResult | None:
        return parse_code(path)

    # --- Read ----------------------------------------------------------------

    def read_document(self, path: Path) -> ParserResult | None:
        """Return file content as a string using mapped parsers."""
        ext = path.suffix.lower()
        parser = self.formats.get(ext, self.read_txt)
        return parser(path)
