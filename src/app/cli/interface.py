import json
import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.columns import Columns
from rich.markup import escape
from rich import box

from src.config.files_and_directories import APP_LOG_FILE, PROMPT_LOG_FILE

from src.libs.custom_dataclasses import AgentHooks

from assests.icons import app_icon_ascii

console = Console()
err_console = Console(stderr=True)
console_handler: RichHandler | None = None


def _print_cli_status(msg: str) -> None:
    err_console.print(escape(msg))


def _print_cli_loading(msg: str) -> None:
    err_console.print(escape(msg))


def _print_cli_error(exc: Exception) -> None:
    err_console.print(f"[red]Error:[/] {escape(str(exc))}")


def build_console(args: argparse.Namespace) -> tuple[Console, Console]:
    common = dict(no_color=args.no_color, width=args.width)
    out = Console(quiet=args.quiet, **common)
    err = Console(stderr=True, **common)
    return out, err


def init_logger(args: argparse.Namespace) -> None:
    """Call once right after argparse.parse_args(), before anything else."""
    global console, err_console, console_handler

    console, err_console = build_console(args)

    console_handler = RichHandler(
        console=err_console,
        rich_tracebacks=True,
        show_path=False,
        show_time=True,
        markup=True,
    )
    console_handler.setFormatter(
        logging.Formatter(
            "%(message)s",
        )
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


def print_cli_hooks():
    return AgentHooks(
        on_loading=_print_cli_loading,
        on_stat_change=_print_cli_status,
        on_error=_print_cli_error,
    )
