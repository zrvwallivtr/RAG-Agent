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

    # --- Attachments dictionaries ------------------------------

    @staticmethod
    def _attachments_tool_call(attchmnts: list[Path] | None) -> dict | None:
        """
        Returns the dictionary for uploaded attachments storing its metadata.

        {
            "filename": {
                "mime_type": ,
                "size_bytes": ,
            }
        }
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
                "size_bytes": attchmnt.stat().st_size if attchmnt.exists() else 0
            }
        app_log.debug("Added attachments metadata into the database")
        return attchmnts_dict

    # --- Slash commands metadata -------------------------------

    @staticmethod
    def _memories_tool_call(unfmt_prompt: str | None = None) -> dict[str, str] | None:
        """
        Returns the dictionary for storing metadata for the '/memories' slash command.

        {
            "unformatted_prompt": ,
        }

        "unformatted_prompt" stores the exact input the user submitted in the chat,
        it will be used to reconstruct the exact chat conversation history as
        '/memories' won't be stored in the user prompt.
        """
        if unfmt_prompt is None:
            return None
        app_log.debug("Added '/memories' too call metadata into the database")
        return {"unformatted_prompt": unfmt_prompt}

    @staticmethod
    def _recall_tool_call(unfmt_prompt: str | None = None) -> dict[str, str] | None:
        """
        Returns the dictionary for storing metadata for the '/recall' slash command.

        {
            "unformatted_prompt": ,
        }

        "unformatted_prompt" stores the exact input the user submitted in the chat,
        it will be used to reconstruct the exact chat conversation history as
        '/recall' won't be stored in the user prompt.
        """
        if unfmt_prompt is None:
            return None
        app_log.debug("Added '/recall' too call metadata into the database")
        return {"unformatted_prompt": unfmt_prompt}

    @staticmethod
    def _compress_tool_call(
        unfmt_prompt: str | None = None,
        method: Literal["default", "instruct", "auto"] | None = None
    ) -> dict | None:
        """
        Returns the dictionary for storing metadata for the '/compress' slash command.

        {
            "unformatted_prompt": ,
            "method": ,
        }

        "unformatted_prompt" stores the exact input the user submitted in the chat,
        it will be used to reconstruct the exact chat conversation history as
        '/compress' won't be stored in the user prompt.
        """
        if method is None:
            return

        app_log.info(
            "Added '/compress' too call metadata into the database: '%s' method",
            method
        )
        return {
            "unformatted_prompt": unfmt_prompt,
            "method": method,
        }

    # --- All metadata dictionaries combined --------------------

    @staticmethod
    def all_tool_calls(
        attchmnts: list[Path] | None = None,
        unfmt_mem_prompt: str | None = None,
        unfmt_recall_prompt: str | None = None,
        unfmt_cmp_prompt: str | None = None,
        cmp_method: Literal["default", "instruct", "auto"] | None = None,
    ) -> dict:
        """
        Returns a dictionary of all tool calls metadata:

        {
            "attachments": ,
            "memories": ,
            "recall": ,
            "compress": ,
        }

        Return an empty dictionary if no tool calls.
        """
        tool_entries = {}

        if attchmnts:
            tool_entries["attachments"] = ToolCallsMetadata._attachments_tool_call(attchmnts)

        if unfmt_mem_prompt:
            tool_entries["memorise"] = ToolCallsMetadata._memories_tool_call(unfmt_mem_prompt)

        if unfmt_recall_prompt:
            tool_entries["recall"] = ToolCallsMetadata._recall_tool_call(unfmt_recall_prompt)

        if cmp_method is not None:
            tool_entries["compress"] = ToolCallsMetadata._compress_tool_call(
                unfmt_cmp_prompt, cmp_method
            )

        return tool_entries
