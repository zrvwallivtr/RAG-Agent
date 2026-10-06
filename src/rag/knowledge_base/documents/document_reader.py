from pathlib import Path

from src.config.files_and_directories import UPLOAD_DIR
from src.config.documents import ENABLE_DOCLING

from src.libs.parsers.basic_parsers import BasicParsers
from src.libs.parsers.docling_parsers import DoclingParsers
from src.logger import app_logger

app_log = app_logger(f"{__name__}.app")


class PathTraversalError(Exception):
    pass


class DocumentReader:
    def __init__(self):
        self.bs_prsrs       = BasicParsers()
        self.docling_prsrs  = DoclingParsers()

    def _validate_path(self, path: Path) -> Path:
        """Resolve 'path' to its real, absolute form."""
        resolved = path.resolve()

        try:
            resolved.relative_to(UPLOAD_DIR)

        except ValueError:
            raise PathTraversalError(
                f"Path '{path}' resolves to '{resolved}', which is outside "
                f"the allowed directory '{UPLOAD_DIR}'"
            )

        return resolved

    def read_document(self, path: Path) -> tuple[str, str] | None:
        """Return file content as a string using mapped parsers and converted file format."""
        safe_path = self._validate_path(path)

        if not safe_path.exists():
            raise FileNotFoundError(f"'{safe_path}' does not exist")
        if not safe_path.is_file():
            raise ValueError(f"'{safe_path}' is not a regular file")

        # Match with the correct parser
        ext = safe_path.suffix.lower()

        # === PRIMARY: DOCLING -> FALLBACK: BASIC =========================================
        if ENABLE_DOCLING:
            try:
                app_log.info("Running Docling parser for '%s'", path)
                parser = self.docling_prsrs.formats.get(ext, self.bs_prsrs.read_txt)
                out = parser(safe_path)
                if not out:
                    return
                cont, format = out
                return cont, format

            # Fallback: Basic
            except Exception as e:
                app_log.warning(
                    "Failed to run Docling for '%s': %s. Falling back to basic parser", path, e
                )
                try:
                    parser = self.bs_prsrs.formats.get(ext, self.bs_prsrs.read_txt)
                    return parser(safe_path), "txt"

                except Exception as fallback_err:
                    app_log.warning(
                        "Both parser methods failed for '%s': %s. Skipping", path, fallback_err
                    )
                    return

        # === PRIMARY: BASIC -> FALLBACK: DOCLING =========================================
        try:
            app_log.info("Running basic parser for '%s'", path)
            parser = self.bs_prsrs.formats.get(ext, self.bs_prsrs.read_txt)
            return parser(safe_path), "txt"

        except Exception as e:
            if not ENABLE_DOCLING:
                app_log.warning("Failed to run basic parser for '%s': Skipping", path)
                return

            app_log.warning(
                "Failed to run basic parser for '%s': %s. Falling back to Docling parser", path, e
            )
            try:
                parser = self.docling_prsrs.formats.get(ext, self.bs_prsrs.read_txt)
                out = parser(safe_path)
                if not out:
                    return
                cont, format = out
                return cont, format

            except Exception as fallback_err:
                app_log.warning(
                    "Failed to run Docling parser for '%s': %s. Skipping", path, fallback_err
                )
                return

    def read_content(
        self,
        is_attchmnt: bool,
        attch_paths: list[Path] | None
    ) -> dict[Path, dict[str, str]] | None:
        """Return content in attachment(s)."""
        if not is_attchmnt:
            return

        if not attch_paths:
            return

        attchmnt_dict = {}
        for path in attch_paths:

            out = self.read_document(path)
            if not out:
                continue
            cont, format = out

            app_log.info("Content extracted from attachment '%s'", path)
            attchmnt_dict[path] = {"content": cont, "format": format}

        return attchmnt_dict
