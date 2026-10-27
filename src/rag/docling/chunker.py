"""
Split parsed documents into chunks sized for embedding. Returns plain
list[str].

The chunk size is measured with the EMBEDDING model's tokenizer.
"""

from io import BytesIO

from src.config.config_toml import (
    DOCLING_CHUNK_TKNS,
    CHUNKER_SAFETY_MARGIN,
    EMBED_MODEL,
)

from ._tokenizer import docling_tokenizer

from src.libs.custom_dataclasses import AgentHooks

from src.agent.tokenizers import LocalTokenizer, load_tokenizer
from src import logger

app_log = logger.app_logger(f"{__name__}.app")


class DoclingChunker:
    def __init__(
        self,
        max_tkns: int = DOCLING_CHUNK_TKNS,
    ):
        """
        emb_tknizr:   For the embedding model
        max_tkns:       Ceiling for a chunk, in tokens.
        """
        if max_tkns < 1:
            raise ValueError("max_tkns must be at least 1.")

        self.emb_tknizr = load_tokenizer(EMBED_MODEL)
        self.max_tkns = max_tkns

        self._chunker = None
        self._converter = None

    def _get_chunker(self):
        """Lazy import docling chunker."""
        if self._chunker is None:
            from docling.chunking import HybridChunker

            self._chunker = HybridChunker(
                tokenizer=docling_tokenizer(self.emb_tknizr, self.max_tkns),
                merge_peers=True,  # optional, defaults to True
            )
        return self._chunker

    def _parse_to_docling_document(
        self, cont: str, hooks: AgentHooks | None = None
    ) -> None:
        """Turn content into a DoclingDocument for the HybridChunker."""
        from docling.datamodel.base_models import DocumentStream
        from docling.document_converter import DocumentConverter

        if self._converter is None:
            self._converter = DocumentConverter()

        stream = DocumentStream(
            name="document.md", stream=BytesIO(cont.encode("utf-8"))
        )

        if hooks and hooks.on_loading:
            hooks.on_loading("Docling chunking content...")
        return self._converter.convert(stream).document

    def chunking(self, cont, hooks: AgentHooks | None = None) -> list[dict]:
        """Chunk DoclingDocument into [{"text"}]."""
        chunker = self._get_chunker()
        doc = (
            self._parse_to_docling_document(cont, hooks=hooks)
            if isinstance(cont, str)
            else cont
        )

        chunks = []
        for chunk in chunker.chunk(dl_doc=doc):
            chunks.append({"text": chunker.contextualize(chunk=chunk)})

        return chunks
