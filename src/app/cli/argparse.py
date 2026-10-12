import json
import argparse
from pathlib import Path

from rich.console import Console
from rich_argparse import RichHelpFormatter

from src.config.postgres import CONN
from src.config.models import MODEL
from src.config.files_and_directories import UPLOAD_DIR

from src.app.cli import interface
from src import logger


def build_parser() -> None:
    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter, description="Local RAG agent")

    # --- General ------------------------------------------------------

    parser.add_argument("prompt", nargs="?", help="Prompt agent")

    parser.add_argument(
        "--delete-default-session", action="store_true", help="Reset default session"
    )

    # --- Settings -----------------------------------------------------

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress non-essential output"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show debug-level logs"
    )

    parser.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )

    parser.add_argument(
        "--width", type=int, default=None, help="Force console width (useful for non-TTY output)"
    )

    # --- Models -------------------------------------------------------

    parser.add_argument(
        "--model",
        "-m",
        default=MODEL,
        metavar=("MODEL_NAME"),
        help=f"Select model (defualt: {MODEL})"
    )

    parser.add_argument(
        "--list-models", "-lm", action="store_true", help="List all installed ollama models"
    )

    parser.add_argument(
        "--install-model", metavar=("MODEL_NAME"), help=f"Install specified model"
    )

    # --- Session flags ------------------------------------------------

    parser.add_argument(
        "--session",
        "-s",
        default=None,
        metavar=("SESSION_NAME"),
        help="Continue a selected session"
    )

    parser.add_argument(
        "--list-session", "-ls", action="store_true", help="List all existing session")

    parser.add_argument(
        "--new-session",
        default=None,
        metavar=("SESSION_NAME"),
        help="Create a new session and prompt agent (optional)"
    )

    parser.add_argument(
        "--delete-session",
        default=None,
        metavar=("SESSION_NAME"),
        help="Delete a specified session"
    )

    # --- Attachments --------------------------------------------------

    parser.add_argument(
        "--attachments",
        "-a",
        nargs="+",
        type=Path,
        default=None,
        metavar=("FILE_PATH"),
        help=f"Read specified attachments (It must be uploaded to {UPLOAD_DIR})"
    )

    parser.add_argument(
        "--list-attachments",
        "-la",
        action="store_true",
        help="List all uploaded attachments in the database"
    )

    # --- Initialise tokenizers ----------------------------------------

    parser.add_argument(
        "--install-tokenizer",
        action="store_true",
        help="Install tokenizers for current installed models"
    )

    # --- Arguments ----------------------------------------------------

    args = parser.parse_args()
    interface.init_logger(args)
    logger.set_verbose(verbose=args.verbose)

    from src.core import Agent
    from src.agent.models import ollama
    from src.agent.chat_session_logs import ChatSessionLogs
    from src.agent import tokenizers
    from src.libs.helpers import sessions_helpers

    # Delete default session flag
    if args.delete_default_session:
        sessions_helpers.del_sess()
        return

    # Install model flag
    if args.install_model:
        ollama.ollama_pull_model(args.install_model)
        return

    # List installed model flag
    if args.list_models:
        model_dict = ollama.ollama_clt.list()
        interface.installed_models_table(model_dict)
        return

    # Create new session flag
    if args.new_session:
        chat_logs = ChatSessionLogs(conn=CONN, sess_name=args.new_session)
        chat_logs.create_session()
        if not args.prompt:
            return
        agent = Agent(sess_name=args.session)
        answer = agent.ask(prompt=args.prompt)
        return

    # Delete session flag
    if args.delete_session:
        response = sessions_helpers.del_sess(args.delete_session)
        if response:
            print(response)
        return

    # List available sessions flag
    if args.list_session:
        chat_logs = ChatSessionLogs(conn=CONN)
        sess_dict = chat_logs.get_all_existing_session_metadata()
        interface.sessions_table(sess_dict)
        return

    # Attach file(s) flag
    if args.attachments:
        if not args.prompt:
            print("Error: Prompt required")
            return

        agent = Agent(sess_name=args.session)
        answer = agent.ask(prompt=args.prompt, is_attchmnt=True, paths=args.attachments)
        return

    # List all uploaded attachment(s) in session
    if args.list_attachments:
        from src.rag.knowledge_base.documents import DocumentKnowledgeBase
        chat_sess_logs = ChatSessionLogs(conn=CONN, sess_name=args.session)
        doc_kw_bs = DocumentKnowledgeBase(conn=CONN, chat_sess_logs=chat_sess_logs)

        def list_all_uploaded_documents() -> str | None:
            """Return a list of all document(s) in the database."""
            rows = doc_kw_bs.query_docs.all_documents_metadata()
            if not rows:
                return

            # Unpack data in metadata
            lines = [
                f"UPLOADED DOCUMENT(S)\n"
                f"====================\n"
                f"UPLOADED AT\t\t\t\tNAME"
            ]
            for time, metadata in rows:
                data_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
                doc_name = data_dict.get("document_name", "Unknown")
                lines.append(f"{str(time)}\t{doc_name}")
            return "\n".join(lines)

        print(list_all_uploaded_documents())
        return

    # Install tokenizer flag
    if args.install_tokenizer:
        from src.agent.tokenizers import install_tokenizers
        install_tokenizers()

    # When prompt is provided
    if args.prompt:
        agent = Agent(sess_name=args.session)
        answer = agent.ask(prompt=args.prompt)

    # Start TUI (No arguments and question)
    if not args.prompt and not any([args.attachments]):
        from src.app.tui.main_app import MainApp
        MainApp().run()
        return
