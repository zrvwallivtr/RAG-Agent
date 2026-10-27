"""
Routing to the correct parser based on if user enabled docling in the config.
"""

from pathlib import Path

from src.config.files_and_directories import UPLOAD_DIR
from src.config.config_toml import ENABLE_DOCLING, ENABLE_PATH_SAFETY

from src.libs.custom_dataclasses import AgentHooks
from src.libs.constants import ParserResult

from src.rag.docling import DoclingParser
from src.rag.basic_parser import BasicParser

from src import logger

app_log = logger.app_logger(f"{__name__}.app")


class PathTraversalError(Exception):
    """
    Raised when a file path attempts to escape outside
    the permitted upload directory.
    """

    pass


class FileReader:
    """
    Manages secure file path resolution, verification, and text extraction via
    Docling or basic parsers.
    """

    def __init__(self, basic_parser: BasicParser, docling_parser: DoclingParser):
        self.basic_parser = basic_parser
        self.docling_parser = docling_parser

    def _docling_parser_default(
        self, path: Path, safe_path: Path, ext: str, hooks: AgentHooks | None = None
    ) -> ParserResult | None:
        """
        If user enabled docling parser:

        Attempts to parse a document using Docling first, falling back to
        the basic parsers upon failure.
        """
        # Docling parser
        try:
            if hooks and hooks.on_loading:
                hooks.on_loading("Running Docling parser for '%s'..." % path)

            parser = self.docling_parser.formats.get(ext, self.basic_parser.read_txt)
            return parser(safe_path)

        # Basic parsers (fallback)
        except Exception as e:
            if hooks and hooks.on_loading:
                hooks.on_loading(
                    "Failed to run Docling for '%s'. "
                    "Falling back to basic parser..." % path
                )
            try:
                parser = self.basic_parser.formats.get(ext, self.basic_parser.read_txt)
                return parser(safe_path)

            except Exception as fallback_err:
                if hooks and hooks.on_error:
                    hooks.on_error(
                        Exception(
                            "Both parser methods failed for '%s': %s. Skipping..."
                            % (path, fallback_err)
                        )
                    )
                return None

    def _basic_parser_default(
        self, path: Path, safe_path: Path, ext: str, hooks: AgentHooks | None = None
    ) -> ParserResult | None:
        """
        Use the basic parser if user disabled docling parser.
        """
        try:
            if hooks and hooks.on_loading:
                hooks.on_loading("Running basic parser for '%s'" % path)

            parser = self.basic_parser.formats.get(ext, self.basic_parser.read_txt)
            return parser(safe_path)

        except Exception as e:
            if hooks and hooks.on_loading:
                hooks.on_loading("Failed to run basic parser for '%s': %s" % (path, e))
            return None

    def read_document(
        self,
        path: Path,
        enable_docling: bool = ENABLE_DOCLING,
        hooks: AgentHooks | None = None,
    ) -> ParserResult | None:
        """
        Return file content as a string using mapped parsers
        and converted file format.
        """
        safe_path = _validate_path(path)

        if not safe_path.exists():
            raise FileNotFoundError("'%s' does not exist" % safe_path)
        if not safe_path.is_file():
            raise ValueError("'%s' is not a regular file" % safe_path)

        # Match with the correct parser
        ext = safe_path.suffix.lower()

        # Docling as parser by default, fallback to basic parser if failed
        if enable_docling:
            return self._docling_parser_default(
                path=path, safe_path=safe_path, ext=ext, hooks=hooks
            )

        # Docling parser disabled, basic parser by default
        return self._basic_parser_default(
            path=path, safe_path=safe_path, ext=ext, hooks=hooks
        )

    def read_content(
        self, attch_paths: list[Path] | None
    ) -> dict[Path, dict[str, str]] | None:
        """Return content in attachment(s)."""
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


def _validate_path(
    path: Path,
    upload_dir: Path = UPLOAD_DIR,
    enable_path_safety: bool = ENABLE_PATH_SAFETY,
) -> Path:
    """
    If path safety is enabled:
    This function resolves the given path and ensures it remains strictly
    contained within the permitted upload directory.
    """
    if enable_path_safety:
        info_str = "Path safety feature is enabled: Validating upload path '%s'" % (
            path
        )
        app_log.debug(info_str)
        resolved = path.resolve()

        try:
            resolved.relative_to(upload_dir)
        except ValueError:
            raise PathTraversalError(
                "Path '%s' resolves to '%s', "
                "which is outside the allowed directory '%s'"
                % (path, resolved, upload_dir)
            )

        return resolved

    info_str = (
        "Path safety feature is disabled: Upload path '%s' validation skipped" % (path)
    )
    app_log.debug(info_str)
    return path
