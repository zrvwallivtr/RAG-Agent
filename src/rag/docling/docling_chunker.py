"""
Split parsed documents into chunks sized for embedding. Returns plain
list[str].

The chunk size is measured with the EMBEDDING model's tokenizer.
"""

import re
import hashlib
from typing import Callable

from src.config.config_toml import DOCLING_CHUNK_TKNS

from src.agent.tokenizers import LocalTokenizer
from src import logger

app_log = logger.app_logger(f"{__name__}.app")

SAFETY_MARGIN = 0.8

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class DoclingChunker:
    def __init__(
        self,
        local_tknizr: LocalTokenizer,
        max_tkns: int = DOCLING_CHUNK_TKNS,
        embed_max_tkns: int | None = None,
    ):
        """
        local_tknizr:   For the embedding model
        max_tkns:       Ceiling for a chunk, in tokens.
        embed_max_tkns: The embedding model's input limit, if known. max_tokens
                        is lowered to SAFETY_MARGIN of it when it is larger.
        """
        if max_tkns < 1:
            raise ValueError("max_tkns must be at least 1.")

        if embed_max_tkns:
            ceiling = max(1, int(embed_max_tkns * SAFETY_MARGIN))
            if max_tkns > ceiling:
                app_log.warning(
                    "Chunk size %d tokens is too close to "
                    "the embedding limit (%d). Using %d instead",
                    max_tkns, embed_max_tkns, ceiling,
                )
                max_tkns = ceiling

        self.max_tkns = max_tkns
        self.local_tknizr = local_tknizr
        self._docling_chunker = None

        self.chunker = None

    def _get_chunker(self):
        """Lazy import docling chunker."""
        if self.chunker is None:
            from docling.chunking import HybridChunker
            self.chunker = HybridChunker(
                tokenizer=self.local_tknizr,
                merge_peers=True,  # optional, defaults to True
            )
        return self.chunker


    # --- Token counting ------------------------------------------------------

    def count(self, text: str) -> int:
        tkn_count = self.local_tknizr.count_string_tokens(text=text)
        return tkn_count if tkn_count is not None else _estimate_tokens(text)

    # --- Docling documents ---------------------------------------------------

    def chunking(self, cont) -> list[str]:
        """
        Chunk a Docling object into plain text strings.
        """
        chunk_iter = self.chunker.chunk(dl_doc=cont)
        return list(chunk_iter)
