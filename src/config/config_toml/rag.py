from src.config.files_and_directories import _cfg

ENABLE_DOCLING  = (
    _cfg.get("rag", {}).get("enable_docling")
    or True
)

RETRIEVE_DOC_ENTRY_LIMIT = (
    _cfg.get("rag", {}).get("retrieve_entry_limit")
    or 3
)

MIN_SIMILARITY_SCORE = (
    _cfg.get("rag", {}).get("min_similarity_score")
    or 0.65
)
