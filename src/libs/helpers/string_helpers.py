import hashlib
import re
from typing import Any


def hash_string(string: str) -> str:
    """
    Parameters
    ----------
        string: String for hashing.

    Returns
    -------
        - SHA-256 hash of raw text string, used for dedupe before embedding.
    """
    return hashlib.sha256(string.encode("utf-8")).hexdigest()


def clean_id_string(any: Any) -> str:
    """
    Parameters
    ----------
        any: Any type of ids.

    Returns
    -------
        - clean string: 'path-{id}'.
    """
    clean_str = re.sub(r"[^a-zA-Z0-9]", "-", str(any)).strip("-")
    clean_str = re.sub(r"-+", "-", clean_str)
    return f"path-{clean_str}"
