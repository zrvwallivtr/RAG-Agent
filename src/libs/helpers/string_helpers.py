import hashlib


def hash_string(string: str) -> str:
    """
    Return a SHA-256 hash of raw text string, used for dedupe before embedding.
    """
    return hashlib.sha256(string.encode("utf-8")).hexdigest()
