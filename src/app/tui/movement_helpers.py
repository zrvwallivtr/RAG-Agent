"""Movement helper utilities for the TUI screens."""

from typing import Literal

from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.events import Key


def pending_key_is_digit_counter(screen) -> int | None:
    """
    Return the counter as an integer

    Fires only when the pending key sequence consists entriely of digits.
    """
    if screen.pending_key and screen.pending_key.isdigit():
        return int(screen.pending_key)
    return None


def calculate_jump_index(
    curr_idx: int | None,
    total: int,
    direction: Literal["up", "down"],
    count: int = 1,
) -> int | None:
    """
    Return the clamped target index or 'None' if there is nothing to jump to.

    - 'down' lands on the count-th item from the start.
    - 'up' lands on the count-th item from the end.
    """
    if total <= 0:
        return None

    # Handle uninitialised selection state
    if curr_idx is None:
        if direction == "down":
            curr_idx = -1
        else:
            curr_idx = total

    if direction == "down":
        target = curr_idx + count
    else:
        target = curr_idx - count

    # Clamp index within valid list boundaries [0, total - 1]
    return max(0, min(target, total - 1))


def calculate_nth_index(n: int, total: int) -> int | None:
    """Return the 0-indexed position for the nth item."""
    if total <= 0:
        return None
    return max(0, min(n - 1, total - 1))


# --- Vertical Scroll behaviour -----------------------------------------------


def _bottom_is_under_viewport_top(
    component: Widget,
    vert_scroll: VerticalScroll,
) -> bool:
    """
    Check if the bottom edge of a component extends under
    the top of the viewport.
    """
    return component.virtual_region.bottom > vert_scroll.scroll_offset.y


def calculate_current_vertical_scroll_component(
    screen,
    vert_scroll: VerticalScroll,
    components: list[Widget],
) -> Widget | None:
    """
    Determine and return the component currently visible in the scroll
    viewport.
    """
    if not components:
        return None

    # Find the component currently visible in the viewport
    target = None
    for component in components:
        if _bottom_is_under_viewport_top(component, vert_scroll):
            target = component
            break

    # Fallback to the last component if none matched
    if not target:
        target = components[-1]

    return target


def calculate_jump_vertical_sroll_component(
    screen,
    vert_scroll: VerticalScroll,
    components: list[Widget],
    direction: Literal["up", "down"],
    count: int = 1,
) -> Widget | None:
    """Calculate the target index and return its widget."""
    if not components:
        return None

    curr_idx = 0
    found = False

    # Check if currently in message mode
    if getattr(screen, "in_msg_mode", False):
        focused_wid = screen.app.focused
        if focused_wid is not None:
            for i, component in enumerate(components):
                if focused_wid is component or component in focused_wid.ancestors:
                    curr_idx = i
                    found = True
                    break

    # Fallback to viewport visibility if no active message focus is found
    if not found:
        for i, component in enumerate(components):
            if _bottom_is_under_viewport_top(component, vert_scroll):
                curr_idx = i
                break
        else:
            curr_idx = len(components) - 1

    # Compute target index using safe boundary clamping
    target_idx = calculate_jump_index(curr_idx, len(components), direction, count)
    return components[target_idx]
