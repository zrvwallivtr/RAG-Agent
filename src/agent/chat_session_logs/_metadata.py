import mimetypes
from pathlib import Path
from typing import Any, TYPE_CHECKING, Literal

from src import logger

app_log = logger.app_logger(f"{__name__}.app")


class ToolCallsMetadata:
    """
    Contains all the functions for building the metadata dictionaries.
    Called in 'add_turn()' in 'write_chat_logs.py'.
    """

    @staticmethod
    def all_tool_calls(
        attchmnts: list[Path] | None = None,
        unfmt_cmp_prompt: str | None = None,
        cmp_method: Literal["default", "instruct", "auto"] | None = None,
    ) -> dict[str, Any]:
        """
        Parameters
        ----------
            attchmnts:    List of attachment paths of a turn.
            unfmt_prompt: The original unedited prompt including the slash command.
            method:       Compression method.

        Returns
        -------
            - A dictionary contain all tool calls data.
        """
        tool_entries = {}

        if attchmnts:
            tool_entries["attachments"] = ToolCallsMetadata._attachments_tool_call(
                attchmnts
            )

        if cmp_method is not None:
            tool_entries["compress"] = ToolCallsMetadata._compress_tool_call(
                unfmt_cmp_prompt, cmp_method
            )

        return tool_entries

    @staticmethod
    def _attachments_tool_call(attchmnts: list[Path] | None) -> dict[str, Any] | None:
        """
        Parameters
        ----------
            attchmnts: List of attachment paths of a turn.

        Returns
        -------
            - A dictionary containing the attachment's data with
              attachment name as the keys.
        """
        if attchmnts is None:
            return

        attchmnts_dict = {}
        count = 0
        for attchmnt in attchmnts:
            count += 1
            mime_type, _ = mimetypes.guess_type(attchmnt)
            attchmnts_dict[attchmnt.name] = {
                "mime_type": mime_type,
                "size_bytes": attchmnt.stat().st_size if attchmnt.exists() else 0,
            }
        app_log.debug("Added attachments metadata into the database")
        return attchmnts_dict

    @staticmethod
    def _compress_tool_call(
        unfmt_prompt: str | None = None,
        method: Literal["default", "instruct", "auto"] | None = None,
    ) -> dict[str, str | None] | None:
        """
        "unfmt_prompt" stores the exact input the user submitted in the chat,
        it will be used to reconstruct the exact chat conversation history as
        '/compress' won't be stored in the user prompt.

        Parameters
        ----------
            unfmt_prompt: The original unedited prompt including the slash command.
            method:       Compression method.

        Returns
        -------
            - A dictionary about the compression tool call data.
        """
        if method is None:
            return

        app_log.debug(
            "Added '/compress' too call metadata into the database: '%s' method", method
        )
        return {
            "unformatted_prompt": unfmt_prompt,
            "method": method,
        }
