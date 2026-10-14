import json
import argparse
import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.columns import Columns
from rich import box

from src.config.files_and_directories import APP_LOG_FILE, PROMPT_LOG_FILE

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
    """
    Print via shared console if initialised, else fallback
    to normal print().
    """
    if console:
        console.print(*args, **kwargs)
    else:
        print(*args)
