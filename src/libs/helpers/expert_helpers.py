from typing import Sequence


def to_vector_literal(values: Sequence[float]) -> str:
    """
    Turns the embeddings (a list of floats) into the text format,
    so it can be passed as an ordinary string parameter.
    """
    return "[" + ",".join(f"{v:.8g}" for v in values) + "]"
