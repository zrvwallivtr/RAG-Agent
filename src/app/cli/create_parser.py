import argparse
from pathlib import Path

from rich_argparse import RichHelpFormatter

from src.config.files_and_directories import UPLOAD_DIR
from src.config.config_toml import MODEL


def create_parser() -> argparse.ArgumentParser:
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
        nargs="+",
        default=None,
        metavar=("SESSION_NAME", "PROMPT"),
        help="Continue a selected session, if not prompt is given, go to session TUI",
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
    parser.add_argument(
        "--session-experts",
        action="store_true",
        help="Show all the enabled experts in the session",
    )

    # Attachments arguments
    parser.add_argument(
        "--attachments",
        "-a",
        nargs="+",
        type=Path,
        default=None,
        metavar=("FILE_PATH"),
        help=("Read specified attachments " f"(It must be uploaded to {UPLOAD_DIR})"),
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
        help="Create an expert, with an optional description (quote it)",
    )
    parser.add_argument(
        "--delete-expert",
        default=None,
        metavar=("EXPERT_NAME"),
        help="Delete an expert (no documents will be deleted)",
    )
    parser.add_argument(
        "--all-experts",
        action="store_true",
        help="Show all available experts",
    )
    parser.add_argument(
        "--enable-session-expert",
        nargs="+",
        metavar=("SESSION_NAME", "EXPERT_NAME"),
        help="Select and enable experts in the specified session",
    )
    parser.add_argument(
        "--disable-session-expert",
        nargs="+",
        metavar=("SESSION_NAME", "EXPERT_NAME"),
        help="Select and disable experts in the specified session",
    )
    parser.add_argument(
        "--expert-uploads",
        nargs="+",
        metavar=("EXPERT_NAME", "DOCUMENT_PATH"),
        help="Embed documents and link them to an expert",
    )
    parser.add_argument(
        "--expert-document-description",
        default=None,
        metavar=("DOCUMENT_NAME"),
        help="Call model to generate a description for the embedded document",
    )
    parser.add_argument(
        "--expert-learn",
        nargs="+",
        metavar=("EXPERT_NAME", "DOCUMENT_NAME"),
        help="Link already embedded documents to an expert",
    )
    parser.add_argument(
        "--expert-forget",
        nargs="+",
        metavar=("EXPERT_NAME", "DOCUMENT_NAME"),
        help="Unlink documents from an expert",
    )
    parser.add_argument(
        "--expert-qualifications",
        default=None,
        metavar=("EXPERT_NAME"),
        help="Show all documents that are accessable by this expert",
    )
    parser.add_argument(
        "--expert-all-documents",
        action="store_true",
        help="Show all expert documents across all experts",
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
