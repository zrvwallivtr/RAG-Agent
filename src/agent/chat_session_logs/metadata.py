import mimetypes
from pathlib import Path
from typing import Any, TYPE_CHECKING

from src import logger


ToolCallsResult = dict[str, dict[str, Any]]


class ToolCallsMetadata:
    @staticmethod
    def _add_attachments_metadata(attchmnts: list[Path] | None) -> ToolCallsResult | None:
        """
        Add metadata to every filename in the list of filenames.

        {
            "filename": {
                "mime_type": "type",
                "size_bytes": size,
            }
        }
        """
        if not attchmnts:
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


    @staticmethod
    def all_tool_calls(attchmnts: list[Path] | None = None) -> ToolCallsResult:
        """
        Return a dictionary of all tool calls.

        {
            "attachments": ,
        }
        """
        tool_entries = {}

        if attchmnts:
            tool_entries["attachments"] = ToolCallsMetadata._add_attachments_metadata(attchmnts)

        return tool_entries
