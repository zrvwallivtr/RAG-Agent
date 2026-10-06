import re
from typing import Literal, get_args

CATEGORY_TYPES = Literal["preference", "stack", "fact", "project", "instruction", "correction"]
CATEGORIES     = list(get_args(CATEGORY_TYPES))


def format_coupled_memory_string(
    ext_out: str, extraction: Literal["manual", "auto"]
) -> dict[str, str]:
    """
    Format every other memory entry to the next line, trims out unnecessary
    spaces and symbols.

    Group 1 = category name
    Group 2 = everything after the brackets
    """
    mem_dict = {}

    # Slice model output into line-by-line format
    for line in ext_out.split("\n"):
        new_line = line.strip().lstrip("-*• ")

        match = re.search(r"\[([a-zA-Z\s_/]+)\]\s*(.*)", new_line)
        if match:
            ctgry_tag   = match.group(1).strip().lower()
            cont        = match.group(2).strip()

            if not cont:
                continue

            ctgry = ctgry_tag if ctgry_tag in CATEGORIES else "fact"
            mem_dict[cont] = ctgry

    return mem_dict
