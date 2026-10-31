from src.config.files_and_directories import _cfg

ENABLE_DOCLING: bool = _cfg.get("rag", {}).get("enable_docling", True)

RETRIEVE_DOC_ENTRY_LIMIT: int = _cfg.get("rag", {}).get("retrieve_entry_limit", 3)

MIN_SIMILARITY_SCORE: float = _cfg.get("rag", {}).get("min_similarity_score", 0.65)

CHUNKER_SAFETY_MARGIN: float = _cfg.get("rag", {}).get("chunker_safety_margin", 0.8)
