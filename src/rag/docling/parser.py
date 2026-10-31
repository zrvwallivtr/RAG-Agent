"""
Contains all the code for the docling parser behaviour, do not import this
module directly, all parser tools route through the 'file_parser' package.
"""

from pathlib import Path

from src.libs.custom_dataclasses import AgentHooks
from src.libs.constants import ParserResult

from src.logger import app_logger

app_log = app_logger(f"{__name__}.app")


class DoclingParser:
    """Uses Docling to convert all listed file format to Markdown."""

    def __init__(
        self,
    ):
        # Map formats to their parsing methods
        self.formats = {
            # Libreoffice
            ".odt": self._read_odt,
            ".ods": self._read_ods,
            ".odp": self._read_odp,
            # Microsoft office
            ".docx": self._read_docx,
            ".xlsx": self._read_xlsx,
            ".pptx": self._read_pptx,
            # Data & configuration formats
            ".csv": self._read_csv,
            ".xml": self._read_xml,
            # Text documents
            ".pdf": self._read_pdf,
            ".epub": self._read_epub,
        }

        self.converter = None

    def _get_converter(self):
        """Lazy import docling converter."""
        if self.converter is None:
            from docling.document_converter import DocumentConverter

            self.converter = DocumentConverter()
        return self.converter

    def _docling_convert_to_md(
        self, path: Path, hooks: AgentHooks | None = None
    ) -> ParserResult | None:
        """
        Convert any file to markdown via docling.

        Parameters
        ----------
            path:  File's path.
            hooks: Function's hooks for cli/tui integration.

        Returns
        -------
            - The parsed file contents in markdown format.
            - Format string.
        """
        string = "Docling converting %s file '%s'..." % (
            path.suffix.lstrip(".").upper(),
            path,
        )
        if hooks and hooks.on_loading:
            hooks.on_loading(string)
        app_log.debug(string)

        try:
            result = self._get_converter().convert(path)
            app_log.info(
                "%s file '%s' converted to MD format",
                path.suffix.lstrip(".").upper(),
                path,
            )
            return result.document.export_to_markdown(), "md"

        except Exception as e:
            err_string = Exception(
                "Docling failed to convert %s file '%s': %s"
                % (path.suffix.lstrip(".").upper(), path, e)
            )
            if hooks and hooks.on_error:
                hooks.on_error(err_string)
            app_log.warning(err_string)
            return None

    # --- Libreoffice ---------------------------------------------------------

    def _read_odt(self, path: Path) -> ParserResult | None:
        return self._docling_convert_to_md(path)

    def _read_ods(self, path: Path) -> ParserResult | None:
        return self._docling_convert_to_md(path)

    def _read_odp(self, path: Path) -> ParserResult | None:
        return self._docling_convert_to_md(path)

    # --- Microsoft office ----------------------------------------------------

    def _read_docx(self, path: Path) -> ParserResult | None:
        return self._docling_convert_to_md(path)

    def _read_xlsx(self, path: Path) -> ParserResult | None:
        return self._docling_convert_to_md(path)

    def _read_pptx(self, path: Path) -> ParserResult | None:
        return self._docling_convert_to_md(path)

    # --- Data & configuration formats ----------------------------------------

    def _read_csv(self, path: Path) -> ParserResult | None:
        return self._docling_convert_to_md(path)

    def _read_xml(self, path: Path) -> ParserResult | None:
        return self._docling_convert_to_md(path)

    # --- Text documents ------------------------------------------------------

    def _read_pdf(self, path: Path) -> ParserResult | None:
        return self._docling_convert_to_md(path)

    def _read_epub(self, path: Path) -> ParserResult | None:
        return self._docling_convert_to_md(path)
