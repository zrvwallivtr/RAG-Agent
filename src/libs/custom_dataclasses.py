from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

from src.libs.constants import TokenCallback, EventCallback


@dataclass
class AgentHooks:
    # Model token streaming
    on_tkn: Optional[TokenCallback] = None  # Whenver new token arrives
    on_first_tkn: Optional[Callable[[], None]] = None  # Fires on the very first token

    # UI state / Feedback
    on_loading: Optional[EventCallback] = (
        None  # Call when long-running operation begins
    )
    on_stat_change: Optional[EventCallback] = None  # For non-blocking notifications

    # Execution lifecycle
    on_complete: Optional[Callable[[dict[str, Any]], None]] = (
        None  # Triggered when full response is done
    )
    on_error: Optional[Callable[[Exception], None]] = (
        None  # Triggered on failure / Render error banners
    )

    # Auto compression summary (shown in its own message box)
    on_cmp_start: Optional[Callable[[], None]] = None
    on_cmp_tkn: Optional[TokenCallback] = None
    on_cmp_done: Optional[Callable[[int], None]] = None
