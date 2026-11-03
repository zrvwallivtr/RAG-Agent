from pathlib import Path

from src.config.files_and_directories import TOKENIZERS_DIR, MODEL_DICT, ModelDict
from src.config.config_toml import MODEL, EMBED_MODEL


def tokenizer_file(hf_repo: str, tknizr_dir: Path = TOKENIZERS_DIR) -> Path:
    """
    Parameters
    ----------
        hf_repo:    Huggingface repo name 'org/name'
        tknizr_dir: The directory storing all the installed tokenizers.

    Returns
    -------
        - '<tknizr_dir'/'org--name.json'
    """
    return tknizr_dir / f"{hf_repo.replace('/', '--')}.json"


def model_context_window(model: str, model_dict: ModelDict = MODEL_DICT) -> int | None:
    """
    Parameters
    ----------
        model:      Model name.
        model_dict: Model dictionary.

    Returns
    -------
        - Context window of the model.
    """
    entry = model_dict.get(model) or model_dict.get(model.removesuffix(":latest"), {})
    model_max_tkns = entry.get("context_window")
    return int(model_max_tkns) if model_max_tkns else None


def model_huggingface_repo(
    model: str, model_dict: ModelDict = MODEL_DICT
) -> str | None:
    """
    Parameters
    ----------
        model:      Model name.
        model_dict: Model dictionary.

    Returns
    -------
        - Hugging Face repo name of the model.
    """
    entry = model_dict.get(model) or model_dict.get(model.removesuffix(":latest"), {})
    hf_repo = entry.get("huggingface_repo")
    return str(hf_repo) if hf_repo else None


def embedding_model_dimension(
    model: str, model_dict: ModelDict = MODEL_DICT
) -> int | None:
    """
    Parameters
    ----------
        model:      Model name.
        model_dict: Model dictionary.

    Returns
    -------
        - Embedding model's dimension.
    """
    entry = model_dict.get(model) or model_dict.get(model.removesuffix(":latest"), {})
    dim = entry.get("dimension")
    return int(dim) if dim else None

# The current language model
MODEL_MAX_TOKENS = model_context_window(MODEL)
MODEL_HF_REPO = model_huggingface_repo(MODEL)

# The current embedding model
EMB_MODEL_MAX_TOKENS = model_context_window(EMBED_MODEL)
EMB_MODEL_HF_REPO = model_huggingface_repo(EMBED_MODEL)
EMB_DIM = embedding_model_dimension(EMBED_MODEL)
