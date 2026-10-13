from src.config.files_and_directories import _cfg

EXPLORATION_FACTOR = _cfg["expert"]["exploration_factor"] or 100
EXPERT_RETRIEVE_LIMIT = _cfg["expert"]["retrieval_limit"] or 8
PER_EXPERT_RETRIEVAL_LIMIT = _cfg["expert"]["per_expert_retrieval_limit"] or 3
