"""
Routing to the correct parser based on if user enabled docling in the config.
"""

from pathlib import Path

from src.config.files_and_directories import UPLOAD_DIR
from src.config.config_toml import ENABLE_DOCLING, ENABLE_PATH_SAFETY

from .basic_parser import BasicParser

from src.libs.custom_dataclasses import AgentHooks

from src.rag.docling import DoclingParser
from src import logger

app_log = logger.app_logger(f"{__name__}.app")


class PathTraversalError(Exception):
    """
    Raised when a file path attempts to escape outside
    the permitted upload directory.
    """
    pass


def _validate_path(
    path: Path,
    enable_path_safety: bool = ENABLE_PATH_SAFETY
) -> Path:
    """
    If path safety is enabled:
    This function resolves the given path and ensures it remains strictly
    contained within the permitted upload directory.
    """
    if enable_path_safety:
        info_str = (
            "Path safety feature is enabled: Validating upload path '%s'"
            % (path)
        )
        app_log.debug(info_str)
        resolved = path.resolve()

        try:
            resolved.relative_to(UPLOAD_DIR)
        except ValueError:
            raise PathTraversalError(
                "Path '%s' resolves to '%s', "
                "which is outside the allowed directory '%s'"
                % (path, resolved, UPLOAD_DIR)
            )

        return resolved

    info_str = (
        "Path safety feature is disabled: Upload path '%s' validation skipped"
        % (path)
    )
    app_log.debug(info_str)
    return path


class FileParser:
    """
    Manages secure file path resolution, verification, and text extraction via
    Docling or basic parsers.
    """

    def __init__(self):
        self.basic_parser  = BasicParser()
        self.docling_prsrs = DoclingParser()

    def _docling_parser_default(
        self,
        path: Path,
        safe_path: Path,
        ext: str,
        hooks: AgentHooks | None = None
    ) -> tuple[str, str] | None:
        """
        Attempts to parse a document using Docling first, falling back to
        the basic parsers upon failure.
        """
        # Docling parser
        try:
            if hooks and hooks.on_loading:
                hooks.on_loading("Running Docling parser for '%s'..." % path)

            parser = self.docling_prsrs.formats.get(ext, self.basic_parser.read_txt)
            out = parser(safe_path)
            if not out:
                return
            cont, format = out
            return cont, format

        # Basic parsers (fallback)
        except Exception as e:
            if hooks and hooks.on_loading:
                hooks.on_loading(
                    "Failed to run Docling for '%s'. "
                    "Falling back to basic parser..."
                    % path
                )
            try:
                parser = self.basic_parser.formats.get(
                    ext, self.basic_parser.read_txt
                )
                return parser(safe_path), "txt"

            except Exception as fallback_err:
                if hooks and hooks.on_error:
                    hooks.on_error(
                        Exception(
                            "Both parser methods failed for '%s': %s. Skipping"
                            % (path, fallback_err)
                        )
                    )
                return None

    def _basic_parser_default(
        self,
        path: Path,
        safe_path: Path,
        ext: str,
        hooks: AgentHooks | None = None
    ) -> tuple[str, str] | None:
        try:
            if hooks and hooks.on_loading:
                hooks.on_loading("Running basic parser for '%s'" % path)

            parser = self.basic_parser.formats.get(ext, self.basic_parser.read_txt)
            return parser(safe_path), "txt"

        except Exception as e:
            if not ENABLE_DOCLING:
                if hooks and hooks.on_error:
                    hooks.on_error(
                        Exception(
                            "Failed to run basic parser for '%s': %s. Skipping"
                            % (path, e)
                        )
                    )
                return None

            if hooks and hooks.on_loading:
                hooks.on_loading(
                    "Failed to run basic parser for '%s': %s. "
                    "Falling back to Docling parser..."
                    % (path, e)
                )

            try:
                parser = self.docling_prsrs.formats.get(
                    ext, self.basic_parser.read_txt
                )
                out = parser(safe_path)
                if not out:
                    return
                cont, format = out
                return cont, format

            except Exception as fallback_err:
                if hooks and hooks.on_error:
                    hooks.on_error(
                        Exception(
                            "Failed to run Docling parser for '%s': %s. Skipping"
                            % (path, fallback_err)
                        )
                    )
                return None

    def read_document(
        self,
        path: Path,
        hooks: AgentHooks | None = None,
    ) -> tuple[str, str] | None:
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
        if ENABLE_DOCLING:
            result = self._docling_parser_default(
                path=path, safe_path=safe_path, ext=ext, hooks=hooks
            )
            if result:
                return result

        # Basic parser by default, fallback to docling if failed
        result = self._basic_parser_default(
            path=path, safe_path=safe_path, ext=ext, hooks=hooks
        )
        return result

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
