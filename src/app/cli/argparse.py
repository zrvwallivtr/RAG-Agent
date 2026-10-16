import argparse
from pathlib import Path

from rich_argparse import RichHelpFormatter

from src.config.postgres import CONN
from src.config.files_and_directories import UPLOAD_DIR
from src.config.config_toml import MODEL

from .interface import init_logger
from .tables import (
    installed_models_table,
    sessions_table,
    upload_documents_table,
)
from .logs import structured_app_log, structured_prompt_log

from src.libs.helpers.sessions_helpers import delete_session

from src.core import Agent
from src.agent.models.ollama import ollama_pull_model, ollama_clt
from src.agent.chat_session_logs import ChatSessionLogs
from src.rag.knowledge_base.documents import DocumentKnowledgeBase
from src.rag.expert import Expert

from src.app.tui.screens.dashboard_screen.session_selector import go_to_session
from src.app.tui.screens.chat_screen import ChatScreen

from src import logger


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
    parser.add_argument(
        "--delete-expert",
        action="store_true",
        help="Delete an expert (no documents will be deleted)",
    )
    parser.add_argument(
        "--expert-upload",
        nargs="+",
        metavar=("DOCUMENT_PATH"),
        help="Embed and chunk documents to database",
    )
    parser.add_argument(
        "--expert-document-description",
        nargs=1,
        default=None,
        metavar=("DOCUMENT_NAME"),
        help="Call model to generate a description for the embedded document",
    )
    parser.add_argument(
        "--expert-learn",
        nargs="+",
        metavar=("DOCUMENT_NAME"),
        help="Link documents to expert",
    )
    parser.add_argument(
        "--expert-forget",
        nargs="+",
        metavar=("DOCUMENT_NAME"),
        help="Unlink documents to expert",
    )
    parser.add_argument(
        "--expert-qualifications",
        nargs=1,
        default=None,
        metavar=("EXPERT_NAME"),
        help="Show all documents that is accessable by this expert",
    )
    parser.add_argument(
        "--expert-all-documents",
        action="store_true",
        help="Sholl all expert documents across all experts",
    )

    # Show logs
    parser.add_argument(
        "--app-logs",
        action="store_true",
        help="Print out the application logs into the terminal",
    )
    parser.add_argument(
        "--prompt-logs",
        action="store_true",
        help="Print out the prompt sent to the model into the terminal",
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

    # Install model
    if args.install_model:
        ollama_pull_model(args.install_model)
        return

    # List installed model
    if args.list_models:
        model_dict = ollama_clt.list()
        installed_models_table(model_dict)
        return

    # Install tokenizer
    if args.install_tokenizer:
        from src.agent.tokenizers import install_tokenizers
        install_tokenizers()

    # --- Session -------------------------------------------------------------

    # List available sessions
    if args.list_session:
        chat_logs = ChatSessionLogs(conn=CONN)
        sess_dict = chat_logs.get_all_existing_session_metadata()
        sessions_table(sess_dict)
        return

    # Create new session
    if args.new_session:
        chat_logs = ChatSessionLogs(conn=CONN, sess_name=args.new_session)
        chat_logs.create_session()

        if not args.prompt:
            return

        agent = Agent(sess_name=args.new_session)
        answer = agent.ask(prompt=args.prompt)
        return

    # Delete default session
    if args.delete_default_session:
        delete_session()
        return

    # Delete session
    if args.delete_session:
        response = delete_session(args.delete_session)
        if response:
            print(response)
        return

    # To session TUI (when --session is given but without prompt)
    if args.session and not args.prompt:
        # - require session name
        # - Show the session in the tui
        go_to_session(ChatScreen, sess_name=args.session)

    # To session TUI --show-conversations
    # - require session name, conversation count
    # - Show the session in the tui while only showing set amount of 
    #   conversation at start up

    # To session TUI --focus
    # - require session name
    # - Show the session in the tui while only showing one conversation turn
    #   at a time (One user and assistant message box each)

    # --- Attachments ---------------------------------------------------------

    # Attach file(s)
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
            upload_documents_table(rows)
            if rows else "No documents uploaded."
        )
        return

    # --- Expert --------------------------------------------------------------

    # Create expert
    if args.create_expert:
        exp_name, des = (args.create_expert + [None])[:2]
        Expert(conn=CONN, exp_name=exp_name, des=des)
        return

    # Delete expert
    # - require expert name
    # - Delete expert WITHOUT DELETING ANY DOCUMENTS

    # ALL Experts
    # - Show all created experts

    # Expert documents
    # - Show all embedded documents across ALL experts in the database

    # Expert learn
    # - require expert name, document paths
    # - store document into the database, and link the document to the expert

    # Expert description
    # - require document name
    # - model read document that was uploaded to the database and generate a
    #   summary description that accurately describe the entire document,
    #   allowing better retrieval

    # Expert forget
    # - require expert name, document name that is in the database
    # - unlink the document and expert in the database table,
    #   WITHOUT REMOVING THE DOCUMENT

    # Expert qualifications
    # - require expert name
    # - Show all documents that is linked to the expert

    # Expert link
    # - require expert name, document name in the database
    # - Link an already embedded document from the database to the expert

    # --- Logs ----------------------------------------------------------------

    # App logs
    if args.app_logs:
        structured_app_log()
        return

    # Prompt logs
    if args.prompt_logs:
        return

    # --- TUI -----------------------------------------------------------------

    # Start TUI (No arguments and question)
    if not args.prompt and not any([args.attachments]):
        from src.app.tui.main_app import MainApp
        MainApp().run()
        return
