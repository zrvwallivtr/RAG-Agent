import json
import argparse
from pathlib import Path

from rich_argparse import RichHelpFormatter

from src.config.postgres import CONN
from src.config.models import MODEL
from src.config.files_and_directories import UPLOAD_DIR

from .interface import init_logger, installed_models_table, sessions_table

from src.libs.helpers import sessions_helpers

from src.core import Agent
from src.agent.models.ollama import ollama_pull_model, ollama_clt
from src.agent.chat_session_logs import ChatSessionLogs
from src.rag.knowledge_base.documents import DocumentKnowledgeBase
from src.rag.expert import Expert

from src import logger


def _format_upload_documents(rows) -> str:
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


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=RichHelpFormatter, description="Local RAG agent"
    )

    # Promptting with not arguments
    parser.add_argument("prompt", nargs="?", help="Prompt agent")

    # Settings arguments
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-essential output",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show debug-level logs",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Force console width (useful for non-TTY output)",
    )

    # Models arguments
    parser.add_argument(
        "--model",
        "-m",
        default=MODEL,
        metavar=("MODEL_NAME"),
        help=f"Select model (defualt: {MODEL})",
    )
    parser.add_argument(
        "--list-models",
        "-lm",
        action="store_true",
        help="List all installed ollama models",
    )
    parser.add_argument(
        "--install-model",
        metavar=("MODEL_NAME"),
        help=f"Install specified model",
    )
    parser.add_argument(
        "--install-tokenizer",
        action="store_true",
        help="Install tokenizers for current installed models",
    )

    # Session arguments
    parser.add_argument(
        "--session",
        "-s",
        default=None,
        metavar=("SESSION_NAME"),
        help="Continue a selected session",
    )
    parser.add_argument(
        "--list-session",
        "-ls",
        action="store_true",
        help="List all existing session",
    )
    parser.add_argument(
        "--new-session",
        default=None,
        metavar=("SESSION_NAME"),
        help="Create a new session and prompt agent (optional)",
    )
    parser.add_argument(
        "--delete-default-session",
        action="store_true",
        help="Reset default session",
    )
    parser.add_argument(
        "--delete-session",
        default=None,
        metavar=("SESSION_NAME"),
        help="Delete a specified session",
    )

    # Attachments arguments
    parser.add_argument(
        "--attachments",
        "-a",
        nargs="+",
        type=Path,
        default=None,
        metavar=("FILE_PATH"),
        help=(
            "Read specified attachments "
            f"(It must be uploaded to {UPLOAD_DIR})"
        ),
    )
    parser.add_argument(
        "--list-attachments",
        "-la",
        action="store_true",
        help="List all uploaded attachments in the database",
    )

    # Expert
    parser.add_argument(
        "--create-expert",
        nargs="+",
        default=None,
        metavar=("EXPERT_NAME", "DESCRIPTION"),
        help="Create an expert, with an optional description",
    )

    return parser


def build_parser() -> None:
    args = _create_parser().parse_args()
    init_logger(args)
    logger.set_verbose(verbose=args.verbose)

    # --- Promptting ----------------------------------------------------------

    # When prompt is provided
    if args.prompt:
        agent = Agent(sess_name=args.session)
        answer = agent.ask(prompt=args.prompt)

    # --- Models --------------------------------------------------------------

    # Install model flag
    if args.install_model:
        ollama_pull_model(args.install_model)
        return

    # List installed model flag
    if args.list_models:
        model_dict = ollama_clt.list()
        installed_models_table(model_dict)
        return

    # Install tokenizer flag
    if args.install_tokenizer:
        from src.agent.tokenizers import install_tokenizers
        install_tokenizers()

    # --- Session -------------------------------------------------------------

    # List available sessions flag
    if args.list_session:
        chat_logs = ChatSessionLogs(conn=CONN)
        sess_dict = chat_logs.get_all_existing_session_metadata()
        sessions_table(sess_dict)
        return

    # Create new session flag
    if args.new_session:
        chat_logs = ChatSessionLogs(conn=CONN, sess_name=args.new_session)
        chat_logs.create_session()

        if not args.prompt:
            return

        agent = Agent(sess_name=args.new_session)
        answer = agent.ask(prompt=args.prompt)
        return

    # Delete default session flag
    if args.delete_default_session:
        sessions_helpers.del_sess()
        return

    # Delete session flag
    if args.delete_session:
        response = sessions_helpers.del_sess(args.delete_session)
        if response:
            print(response)
        return

    # --- Attachments ---------------------------------------------------------

    # Attach file(s) flag
    if args.attachments:
        if not args.prompt:
            print("Error: Prompt required")
            return

        agent = Agent(sess_name=args.session)
        answer = agent.ask(
            prompt=args.prompt, is_attchmnt=True, paths=args.attachments
        )
        return

    # List all uploaded attachment(s) in session
    if args.list_attachments:
        chat_sess_logs = ChatSessionLogs(
            conn=CONN, sess_name=args.session
        )
        doc_kw_bs = DocumentKnowledgeBase(
            conn=CONN, chat_sess_logs=chat_sess_logs
        )

        rows = doc_kw_bs.query_docs.all_documents_metadata()
        print(
            _format_upload_documents(rows)
            if rows else "No documents uploaded."
        )
        return

    # --- Expert --------------------------------------------------------------

    # Create expert
    if args.create_expert:
        exp_name, des = (args.create_expert + [None])[:2]
        Expert(conn=CONN, exp_name=exp_name, des=des)

    # --- TUI -----------------------------------------------------------------

    # Start TUI (No arguments and question)
    if not args.prompt and not any([args.attachments]):
        from src.app.tui.main_app import MainApp
        MainApp().run()
        return
