import json
from typing import Any

from rich.table import Table
from rich import box

from src.config.postgres import CONN

from .interface import safe_print

from src.libs.helpers.date_helpers import trimmed_date

from src.rag.expert import Expert


def load_existing_expert(name: str) -> Expert | None:
    """Validate and return the expert if exists in the database."""
    if not Expert.exists(CONN, name):
        print("Error: Expert '%s' not found. Use --create-expert first" % name)
        return None
    return Expert(conn=CONN, exp_name=name)


def show_all_experts_table(exp_dict: dict[str, dict[str, Any]] | None) -> None:
    if not exp_dict:
        safe_print("No available experts in the database")
        return

    table = Table(title="AVAILABLE EXPERTS", box=box.SIMPLE)

    table.add_column(
        "Created at",
        justify="center",
        no_wrap=True,
        vertical="middle",
        style="blue",
    )

    table.add_column(
        "Updated at",
        justify="center",
        no_wrap=True,
        vertical="middle",
        style="blue",
    )

    table.add_column(
        "Expert name",
        justify="center",
        vertical="middle",
    )

    table.add_column(
        "Document count",
        justify="center",
        vertical="middle",
    )

    for exp in exp_dict:
        table.add_row(
            trimmed_date(str(exp_dict[exp]["created_at"])),
            trimmed_date(str(exp_dict[exp]["updated_at"])),
            str(exp),
            str(exp_dict[exp]["document_count"]),
        )

    safe_print(table)


def show_expert_documents_table(doc_dict: dict[int, dict[str, Any]]) -> None:
    if not doc_dict:
        safe_print("No document stored in the database")
        return

    table = Table(title="AVAILABLE EXPERTS", box=box.SIMPLE)

    table.add_column(
        "Created at",
        justify="center",
        no_wrap=True,
        vertical="middle",
        style="blue",
    )

    table.add_column(
        "Type",
        justify="center",
        vertical="middle",
    )

    table.add_column(
        "Title",
        justify="center",
        vertical="middle",
    )

    for doc in doc_dict:
        table.add_row(
            trimmed_date(str(doc_dict[doc]["created_at"])),
            str(doc_dict[doc]["type"]),
            str(doc_dict[doc]["title"]),
        )

    safe_print(table)


def show_all_documents_table(doc_dict: dict[int, dict[str, Any]]) -> None:
    if not doc_dict:
        safe_print("No document stored in the database")
        return

    table = Table(title="AVAILABLE EXPERTS", box=box.SIMPLE)

    table.add_column(
        "Created at",
        justify="center",
        no_wrap=True,
        vertical="middle",
        style="blue",
    )

    table.add_column(
        "Type",
        justify="center",
        vertical="middle",
    )

    table.add_column(
        "Expert(s)",
        justify="center",
        vertical="middle",
    )

    table.add_column(
        "Title",
        justify="center",
        vertical="middle",
    )

    for doc in doc_dict:
        table.add_row(
            trimmed_date(str(doc_dict[doc]["created_at"])),
            str(doc_dict[doc]["type"]),
            str(doc_dict[doc]["experts"]),
            str(doc_dict[doc]["title"]),
        )

    safe_print(table)
