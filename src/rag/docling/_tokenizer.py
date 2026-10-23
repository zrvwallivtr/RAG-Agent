"""Tokenizer function for the HybridChunker in 'chunker.py'."""

from typing import Any

from src.agent.tokenizers import LocalTokenizer


def docling_tokenizer(
    local_tknizr: LocalTokenizer,
    max_tkns: int,
):
    """
    Wrap this tokenizer for Docling's HybridChunker (it accepts any
    BaseTokenizer, uses installed tokenizer so that it works offline).
    """
    from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer

    class _DoclingTokenizer(BaseTokenizer):
        local_tknizr: Any
        max_tkns: int

        def count_tokens(self, text: str) -> int:
            return self.local_tknizr.count_or_estimate_tokens(text)

        def get_max_tokens(self) -> int:
            return self.max_tkns

        def get_tokenizer(self) -> Any:
            return self.count_tokens

    return _DoclingTokenizer(local_tknizr=local_tknizr, max_tkns=max_tkns)
