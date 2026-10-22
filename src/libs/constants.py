"""
Contains constant for the agent code.

DO NOT IMPORT ANY OTHER MODULES HERE.
"""

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

CompressionResult = tuple[str, int, int, dict[str, dict[str, Any]]]

ModelResult = tuple[str, int, int, bool]

ParserResult = tuple[str, str]

TokenCallback = Callable[[str], bool | None]
EventCallback = Callable[[str], None]

# Compression slash command
COMPRESS_DB_ENTRY = "compress"
COMPRESS_CMD = "/compress"
DEFAULT_PROMPT = "Summarise all previous conversations."
NO_PROMPT_WARN = (
    "Command '/compress' aborted: Prompt must be provided if attachment is uploaded"
)
COMPRESS_METHOD = Literal["default", "instruct", "auto"]

# Loading indicator status strings
THINKING = "Thinking..."
AUTO_COMPRESS = ""
RESPONSE_STOPPED = "*(stopped)*"
NO_RESPONSE = "No Response"

# Widget strings
AUTO_COMPRESS_SUM_TITLE = "Auto session summary due to token limit reached"
