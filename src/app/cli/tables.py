import json

from rich.table import Table
from rich import box

from .interface import safe_print


def sessions_table(sess_dict: dict | None) -> None:
    """List all user created sessions."""
    if not sess_dict:
        safe_print("No session in the database")
        return

    table = Table(
        title="[bright_cyan bold not italic]AVAILABLE SESSIONS",
        box=box.SIMPLE
    )

    table.add_column(
        "[bright_cyan not bold]Created at",
        justify="center",
        no_wrap=True,
        vertical="middle",
        style="blue"
    )
    table.add_column(
        "[bright_cyan not bold]Session name",
        justify="center",
        vertical="middle"
    )

    for sess in sess_dict:
        table.add_row(
            f"{sess_dict[sess]['created_at']}",
            f"{sess_dict[sess]['session_name']}"
        )
    safe_print(table)


def installed_models_table(model_dict: dict | None) -> None:
    """List all installed models."""
    if not model_dict:
        safe_print("No model installed locally")
        return

    table = Table(
        title="[bright_cyan bold not italic]Installed Models",
        box=box.SIMPLE
    )

    table.add_column(
        "[bright_cyan not bold]Modified at",
        justify="center",
        vertical="middle",
        style="blue"
    )
    table.add_column(
        "[bright_cyan not bold]Model name",
        justify="center",
        vertical="middle"
    )
    table.add_column(
        "[bright_cyan not bold]Family",
        justify="center",
        vertical="middle"
    )
    table.add_column(
        "[bright_cyan not bold]Parameter size",
        justify="center",
        vertical="middle"
    )
    table.add_column(
        "[bright_cyan not bold]Quantization level",
        justify="center",
        vertical="middle"
    )

    for model in model_dict.models:
        table.add_row(
            str(model.modified_at),
            str(model.model),
            str(model.details.family),
            str(model.details.parameter_size),
            str(model.details.quantization_level),
        )
    safe_print(table)


def upload_documents_table(rows) -> str:
    lines = [
        f"UPLOADED DOCUMENT(S)\n"
        f"====================\n"
        f"UPLOADED AT\t\t\t\tNAME"
    ]

    for uploaded_at, metadata in rows:
        data = (
            json.loads(metadata) 
            if isinstance(metadata, str) else metadata
        )
        doc_name = data.get("document_name", "Unknown")
        lines.append(f"{str(uploaded_at)}\t{doc_name}")

    return "\n".join(lines)
