from textual.widgets import Static

from src.config.models import FALLBACK_TOKENIZER

from .tokenizers import get_tknizr_list

MODEL_ROLES = ["chat_model", "embedding_model"]


def update_model_status_section(screen):
    screen.query_one("#model-section-title", Static).update(
        f"[bold]Models[/] ({len(screen.ava_models)} installed)"
    )

    model_list = []
    for model_type in MODEL_ROLES:
        model_names = model_type.replace("_", " ").capitalize()
        model_list.append(f"{model_names}")
    models_str = "\n".join(model_list)

    selected_list = []
    for model in screen.selected_models.values():
        selected_list.append(f"{model}")
    selected_str = "\n".join(selected_list)

    screen.query_one("#model-type-list", Static).update(models_str)
    screen.query_one("#selected-model-list", Static).update(selected_str)


def update_tokenizer_status_section(screen):
    tknizr_list = get_tknizr_list()
    tknizr_count = len(tknizr_list) if tknizr_list else 0

    screen.query_one("#tokenizer-section-title", Static).update(
        f"[bold]Tokenizers[/] ({tknizr_count} installed)"
    )
    screen.query_one("#fallback-tokenizer", Static).update("Fallback tokenizer:")
    screen.query_one("#selected-tokenizer", Static).update(f"{FALLBACK_TOKENIZER}")
