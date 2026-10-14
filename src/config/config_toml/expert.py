from src.config.files_and_directories import _cfg

# Set exploration factor
DEFALUT_EXPLORATION_FACTOR = 100
EXPLORATION_FACTOR = (
    _cfg.get("expert", {}).get("exploration_factor")
    or DEFALUT_EXPLORATION_FACTOR
)

# Set expert retrieval limit
DEFAULT_EXPERT_RETRIEVAL_LIMIT = 8
EXPERT_RETRIEVE_LIMIT = (
    _cfg.get("expert", {}).get("retrieval_limit")
    or DEFAULT_EXPERT_RETRIEVAL_LIMIT
)

# Set per expert retrieval limit
DEFAULT_PER_EXPERT_RETRIEVAL_LIMIT = 3
PER_EXPERT_RETRIEVAL_LIMIT = (
    _cfg.get("expert", {}).get("per_expert_retrieval_limit")
    or DEFAULT_PER_EXPERT_RETRIEVAL_LIMIT
)

# Set docling chunks tokens
DEFAULT_DOCLING_CHUNK_TKNS = 300
DOCLING_CHUNK_TKNS = (
    _cfg.get("expert", {}).get("docling_chunk_tokens")
    or DEFAULT_DOCLING_CHUNK_TKNS
)
