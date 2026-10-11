from src.config.files_and_directories import _cfg

ENABLE_DOCLING  = _cfg["rag"]["enable_docling"] or True
RETRIEVE_DOC_ENTRY_LIMIT = _cfg["rag"]["retrieve_entry_limit"] or 3
