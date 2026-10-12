from typing import Sequence


def to_vector_literal(values: Sequence[float]) -> str:
    """pgvector text literal, used in SQL as '%s::vector'."""
    return "[" + ",".join(f"{v:.8g}" for v in values) + "]"
