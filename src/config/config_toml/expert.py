from src.config.files_and_directories import _cfg

# Set exploration factor
DEFALUT_EXPLORATION_FACTOR = 100
EXPLORATION_FACTOR = (
    _cfg.get("expert", {}).get("exploration_factor") or DEFALUT_EXPLORATION_FACTOR
)

# Set expert retrieval limit
DEFAULT_EXPERT_RETRIEVAL_LIMIT = 8
EXPERT_RETRIEVE_LIMIT = (
    _cfg.get("expert", {}).get("retrieval_limit") or DEFAULT_EXPERT_RETRIEVAL_LIMIT
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
    _cfg.get("expert", {}).get("docling_chunk_tokens") or DEFAULT_DOCLING_CHUNK_TKNS
)

# Set minimum similar score for embedding retrieval
DEFAULT_EXPERT_MIN_SIM_SCORE = 0.65
EXP_MIN_SIM_SCORE = (
    _cfg.get("expert", {}).get("minimum_similar_score") or DEFAULT_EXPERT_MIN_SIM_SCORE
)

# Set reserve tokens for when the expert replies
DEFAULT_EXPERT_RESERVE_TKNS = 10000
EXPERT_RESERVE_TKNS = (
    _cfg.get("expert", {}).get("reply_reserve_tokens") or DEFAULT_EXPERT_RESERVE_TKNS
)

# Fallback maximum character count (for when the model's tokenizer fails to
# calculate character limit) to sent to the model when generating the document
# description to avoid exceeding the model's token limits.
DEFAULT_EXPERT_DOC_FALLBACK_CHARS = 12000
EXPERT_DOC_FALLBACK_CHARS = (
    _cfg.get("expert", {}).get("doc_fallback_chars")
    or DEFAULT_EXPERT_DOC_FALLBACK_CHARS
)
