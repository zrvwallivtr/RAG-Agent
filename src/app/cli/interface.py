import argparse
import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.columns import Columns
from rich import box

from src.config import models
from assests.icons import app_icon_ascii


console: Console | None = None
console_handler: RichHandler | None = None


def build_console(args: argparse.Namespace) -> Console:
    return Console(
        no_color=args.no_color,
        width=args.width,
        quiet=args.quiet,
        force_terminal=None,
    )


def init_logger(args: argparse.Namespace) -> None:
    """Call once right after argparse.parse_args(), before anything else."""
    global console, console_handler

    console = build_console(args)

    console_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_path=False,
        show_time=True,
        markup=True,
    )
    console_handler.setFormatter(
        logging.Formatter("%(message)s",)
    )
    console_handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)


def safe_print(*args, **kwargs) -> None:
    """Print via shared console if initialised, else fallback to normal print()."""
    if console:
        console.print(*args, **kwargs)
    else:
        print(*args)


def sessions_table(sess_dict: dict | None) -> None:
    """List all user created sessions."""
    if not sess_dict:
        safe_print("No session in the database")
        return

    table = Table(title="[bright_cyan bold not italic]AVAILABLE SESSIONS", box=box.SIMPLE)
    table.add_column("[bright_cyan not bold]Created at", justify="center", no_wrap=True, vertical="middle", style="blue")
    table.add_column("[bright_cyan not bold]Session name", justify="center", vertical="middle")

    for sess in sess_dict:
        table.add_row(f"{sess_dict[sess]['created_at']}", f"{sess_dict[sess]['session_name']}")
    safe_print(table)


def installed_models_table(model_dict: dict | None) -> None:
    """List all installed models."""
    if not model_dict:
        safe_print("No model installed locally")
        return

    table = Table(title="[bright_cyan bold not italic]INSTALLED MODELS", box=box.SIMPLE)
    table.add_column("[bright_cyan not bold]Modified at", justify="center", vertical="middle", style="blue")
    table.add_column("[bright_cyan not bold]Model name", justify="center", vertical="middle")
    table.add_column("[bright_cyan not bold]Family", justify="center", vertical="middle")
    table.add_column("[bright_cyan not bold]Parameter size", justify="center", vertical="middle")
    table.add_column("[bright_cyan not bold]Quantization level", justify="center", vertical="middle")

    for model in model_dict.models:
        table.add_row(f"{model.modified_at}", f"{model.model}", f"{model.details.family}", f"{model.details.parameter_size}", f"{model.details.quantization_level}")
    safe_print(table)
