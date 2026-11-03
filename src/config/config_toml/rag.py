from src.config.files_and_directories import CONFIG

ENABLE_DOCLING: bool = CONFIG.get("rag", {}).get("enable_docling", True)

RETRIEVE_DOC_ENTRY_LIMIT: int = CONFIG.get("rag", {}).get("retrieve_entry_limit", 3)

MIN_SIMILARITY_SCORE: float = CONFIG.get("rag", {}).get("min_similarity_score", 0.65)

CHUNKER_SAFETY_MARGIN: float = CONFIG.get("rag", {}).get("chunker_safety_margin", 0.8)
