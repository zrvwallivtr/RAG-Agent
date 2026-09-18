from rich.panel import Panel
from rich import box

from src.config import models, postgres
from src.agent.chat_logs import ChatLogs
from src.agent import tokenizers
from assests.icons import app_icon_ascii


def current_fallback_tokenizer() -> str:
    tknizr_dict = tokenizers.fetch_all_installed_tokenizers()

    if not tknizr_dict:
        return f"[bold]Tokenizers[/bold] (0 installed)\n"
    return (
        f"[bold]Tokenizers[/bold] ({len(tknizr_dict)} installed)\n"
        f"[bright_black]│[/bright_black]\tFallback tokenizer: [yellow]{models.FALLBACK_TOKENIZER}[/yellow]\n\n"
    )


def latest_session() -> str:
    chat_logs = ChatLogs(conn=postgres.conn)
    sess_dict = chat_logs.get_all_existing_sess_metadata()
    latest = chat_logs.latest_modified_chat_session()
    sess_name, sess_dt = latest if latest else (None, None)

    if not sess_dict:
        return f"[bold]Sessions[/bold] (0 created)\n"

    return (
        f"[bold]Sessions[/bold] ({len(sess_dict)} created)\n"
        f"[bright_black]│[/bright_black]\tLatest: [blue]{sess_dt}[/blue] [yellow]{sess_name}[/yellow]"
    )


def app_status() -> Panel:
    status_panel = current_fallback_tokenizer() + latest_session()

    return Panel(
            (status_panel),
            title=f"[bright_cyan bold not italic]STATUS",
            box=box.SQUARE
    )


def selected_model_section(ava_models: dict) -> str:
    return (
        f"[bold]Models[/bold] ({len(ava_models)} installed)\n"
        f"[bright_black]│[/bright_black]\tChat model:\t\t[yellow]{models.MODEL}[/yellow]\n"
        f"[bright_black]│[/bright_black]\tMemory model:\t\t[yellow]{models.MEM_MODEL}[/yellow]\n"
        f"[bright_black]│[/bright_black]\tWeb search model:\t[yellow]{models.SEAR_MODEL}[/yellow]\n"
        f"[bright_black]│[/bright_black]\tEmbedding model:\t[yellow]{models.EMBED_MODEL}[/yellow]\n\n"
    )
