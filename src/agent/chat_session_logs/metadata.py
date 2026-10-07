import mimetypes
from pathlib import Path
from typing import Any, TYPE_CHECKING, Literal

from src import logger


class ToolCallsMetadata:
    @staticmethod
    def _attachments_tool_call(attchmnts: list[Path] | None) -> dict | None:
        """
        Add metadata to every filename in the list of filenames:

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
        return attchmnts_dict

    # Slash command tools

    @staticmethod
    def _memories_tool_call(unfmt_prompt: str | None = None) -> dict[str, str] | None:
        """
        Metadata for the memories slash command:
        {
            "unformatted_prompt": ,
        }

        "unformatted_prompt" stores the exact input the user submitted in the chat,
        it will be used to reconstruct the exact chat conversation history as
        '/memories' won't be stored in the user prompt.
        """
        if unfmt_prompt is None:
            return None
        return {"unformatted_prompt": unfmt_prompt}

    @staticmethod
    def _recall_tool_call(unfmt_prompt: str | None = None) -> dict[str, str] | None:
        """
        Metadata for the recall slash command:
        {
            "unformatted_prompt": ,
        }

        "unformatted_prompt" stores the exact input the user submitted in the chat,
        it will be used to reconstruct the exact chat conversation history as
        '/recall' won't be stored in the user prompt.
        """
        if unfmt_prompt is None:
            return None
        return {"unformatted_prompt": unfmt_prompt}

    @staticmethod
    def _compress_tool_call(
        unfmt_prompt: str | None = None, method: Literal["default", "instruct"] | None = None
    ) -> dict | None:
        """
        Metadata for compression method:

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

        return {
            "unformatted_prompt": unfmt_prompt,
            "method": method,
        }

    # All tools

    @staticmethod
    def all_tool_calls(
        attchmnts: list[Path] | None = None,
        unfmt_mem_prompt: str | None = None,
        unfmt_recall_prompt: str | None = None,
        unfmt_cmp_prompt: str | None = None,
        cmp_method: Literal["default", "instruct"] | None = None,
    ) -> dict:
        """
        Return a dictionary of all tool calls:

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
            tool_entries["memories"] = ToolCallsMetadata._memories_tool_call(unfmt_mem_prompt)

        if unfmt_recall_prompt:
            tool_entries["recall"] = ToolCallsMetadata._recall_tool_call(unfmt_recall_prompt)

        if unfmt_cmp_prompt and cmp_method:
            tool_entries["compress"] = ToolCallsMetadata._compress_tool_call(unfmt_cmp_prompt, cmp_method)

        return tool_entries
