from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

COMPRESS_DB_ENTRY = "compress"
COMPRESS_CMD   = "/compress"
DEFAULT_PROMPT = "Summarise all previous conversations."
NO_PROMPT_WARN = "Command '/compress' aborted: Prompt must be provided if attachment is uploaded"
COMPRESS_METHOD = Literal["default", "instruct", "auto"]

CompressionResult = tuple[str, int, int, dict[str, dict[str, Any]]]
GenericEventHook = Callable[[str], None]

ModelResult = tuple[str, int, int, bool]
TokenCallback = Callable[[str], bool | None]
EventCallback = Callable[[str], None]
