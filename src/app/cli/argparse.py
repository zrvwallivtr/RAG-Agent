import argparse
from pathlib import Path

from app.tui.screens.chat_screen.main_module import LLM_TKNIZR
from rich_argparse import RichHelpFormatter

from src.config.postgres import CONN
from src.config.files_and_directories import UPLOAD_DIR
from src.config.config_toml import MODEL, EMBED_MODEL

from .create_parser import create_parser
from .interface import init_logger, safe_print, print_cli_hooks
from .tables import (
    installed_models_table,
    sessions_table,
    upload_documents_table,
    session_enabled_experts,
)
from .logs import structured_app_log, structured_prompt_log
from .expert import (
    load_existing_expert,
    show_all_experts_table,
    show_expert_documents_table,
    show_all_documents_table,
)

from src.libs.helpers.sessions_helpers import delete_session
from src.libs.custom_dataclasses import AgentHooks

from src.core import Agent
from src.agent.models.ollama import ollama_pull_model, ollama_clt
from src.agent.chat_session_logs import ChatSessionLogs
from src.agent.tokenizers import LocalTokenizer
from src.rag.expert import Expert, ExpertDocuments, SessionExperts
from src.rag.attach_files import Attachments

from src.app.tui.screens.dashboard_screen.session_selector import go_to_session
from src.app.tui.screens.chat_screen import ChatScreen

from src import logger


def build_parser() -> None:
    parser = create_parser()
    args = parser.parse_args()
    init_logger(args)
    logger.set_verbose(verbose=args.verbose)

    # --- Models / Tokenizer --------------------------------------------------

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
    # General arguments for session features.

    # List available sessions
    if args.list_session:
        chat_logs = ChatSessionLogs(conn=CONN, llm_tknizr=LocalTokenizer(MODEL))
        sess_dict = chat_logs.get_all_existing_session_metadata()
        sessions_table(sess_dict)
        return

    # Create new session
    if args.new_session:
        chat_logs = ChatSessionLogs(
            conn=CONN, sess_name=args.new_session, llm_tknizr=LocalTokenizer(MODEL)
        )
        chat_logs.create_session(print_cli_hooks())

        if not args.prompt:
            return

        agent = Agent(sess_name=args.new_session)
        answer = agent.ask(prompt=args.prompt)
        return

    # Delete default session
    if args.delete_default_session:
        delete_session(llm_tknizr=LLM_TKNIZR, hooks=print_cli_hooks())
        return

    # Delete session
    if args.delete_session:
        response = delete_session(
            sess_name=args.delete_session,
            llm_tknizr=LLM_TKNIZR,
            hooks=print_cli_hooks(),
        )
        if response:
            print(response)
        return

    # Show all experts enabled in session
    if args.session_experts:
        exp_dict = SessionExperts.all_expert_data_from_session(conn=CONN)
        session_enabled_experts(exp_dict)
        return

    # To session TUI --show-conversations
    # - require session name, conversation count
    # - Show the session in the tui while only showing set amount of
    #   conversation at start up

    # To session TUI --focus
    # - require session name
    # - Show the session in the tui while only showing one conversation turn
    #   at a time (One user and assistant message box each)

    # --- Attachments ---------------------------------------------------------
    # Attaching files for additional context.

    # Attach file(s)
    if args.attachments:
        if not args.prompt:
            safe_print("Error: Prompt required")
            return

        agent = Agent(sess_name=args.session)
        answer = agent.ask(
            prompt=args.prompt,
            paths=args.attachments,
            hooks=print_cli_hooks(),
        )
        return

    # List all uploaded attachment(s) in session
    if args.list_attachments:
        chat_sess_logs = ChatSessionLogs(
            conn=CONN, sess_name=args.session, llm_tknizr=LocalTokenizer(MODEL)
        )
        attchmnts = Attachments(conn=CONN, chat_sess_logs=chat_sess_logs)
        # doc_kw_bs = DocumentKnowledgeBase(conn=CONN, chat_sess_logs=chat_sess_logs)

        rows = attchmnts.all_attachments_data_from_session()
        safe_print(upload_documents_table(rows) if rows else "No documents uploaded.")
        return

    # --- Expert --------------------------------------------------------------
    # Arguments for the expert feature and document processing for the expert.

    # Create expert
    if args.create_expert:
        if len(args.create_expert) > 2:
            parser.error(
                "--create-expert takes EXPERT_NAME and an optional "
                "DESCRIPTION (quote the description)"
            )

        exp_name, des = (args.create_expert + [None])[:2]
        if Expert.exists(CONN, exp_name):
            safe_print("Expert '%s' already exists" % exp_name)
            return
        Expert(conn=CONN, exp_name=exp_name, des=des, hooks=print_cli_hooks())
        return

    # Delete expert (No documents will be deleted)
    if args.delete_expert:
        exp = load_existing_expert(args.delete_expert)
        if exp:
            exp.delete_expert()
            safe_print("Expert '%s' has been deleted" % args.delete_expert)
            return

        safe_print("Deletion failed: Expert '%s' does not exists" % args.delete_expert)
        return

    # Show all created experts
    if args.list_experts:
        exp_dict = Expert.all_experts_data(conn=CONN)
        show_all_experts_table(exp_dict)
        return

    # Link experts to session
    if args.enable_session_expert:
        if len(args.enable_session_expert) < 2:
            parser.error(
                "--enable-session-expert needs SESSION_NAME and "
                "at least one EXPERT_NAME"
            )

        sess_name, *exp_list = args.enable_session_expert
        for exp_name in [*exp_list]:
            sess_exp = SessionExperts(conn=CONN, sess_name=sess_name, exp_name=exp_name)
            is_linked = sess_exp.link_expert_to_session(sess_name)
            if not is_linked:
                safe_print(
                    "Error: Failed to enable expert '%s' for session '%s'"
                    % (exp_name, sess_name)
                )
                return
            safe_print(
                "Expert '%s' has been enabled for session '%s'" % (exp_name, sess_name)
            )
        return

    # Unlink experts from session
    if args.disable_session_expert:
        if len(args.disable_session_expert) < 2:
            parser.error(
                "--disable-session-expert needs SESSION_NAME and "
                "at least one EXPERT_NAME"
            )

        sess_name, *exp_list = args.disable_session_expert
        for exp_name in [*exp_list]:
            sess_exp = SessionExperts(conn=CONN, sess_name=sess_name, exp_name=exp_name)
            is_unlinked = sess_exp.unlink_expert_from_session(sess_name)
            if not is_unlinked:
                safe_print(
                    "Error: Failed to disable expert '%s' for session '%s'"
                    % (exp_name, sess_name)
                )
                return
            safe_print(
                "Expert '%s' has been disabled for session '%s'" % (exp_name, sess_name)
            )
        return

    # Upload document for an expert
    if args.upload_to_expert:
        if len(args.expert_uploads) < 2:
            parser.error(
                "--expert-uploads needs EXPERT_NAME and at least one DOCUMENT_PATH"
            )

        exp_name, *paths = args.expert_uploads

        results = {}
        for p in paths:
            path = Path(p).expanduser()
            if not path.is_file():
                print("Error: '%s' is not a file. Skipping" % p)
                continue
            exp_doc = ExpertDocuments(
                conn=CONN,
                exp_name=exp_name,
            )
            out = exp_doc.chunk_document_to_database(path, hooks=print_cli_hooks())
            if not out:
                safe_print(
                    "Error: Nothing was stored for '%s'. "
                    "Run --app-logs to see why" % p
                )
                continue

            for title, (doc_id, status) in out.items():
                if status == "added":
                    safe_print(
                        "Embedded and stored '%s' (document id %d)" % (title, doc_id)
                    )
                elif status == "linked":
                    safe_print(
                        "'%s' was already embedded (id %d). Linked to '%s', "
                        "nothing re-embeedded" % (title, doc_id, exp_name)
                    )
                else:
                    safe_print(
                        "'%s' (id %d) is already linked to '%s'. "
                        "There is nothing to do" % (title, doc_id, exp_name)
                    )
        return

    # Generate document description
    if args.generate_expert_doc_des:
        try:
            ExpertDocuments.generate_document_description(
                CONN,
                args.expert_document_description,
                hooks=print_cli_hooks(),
            )
            print(
                "Description for the document '%s' has been generated"
                % args.expert_document_description
            )
        except ValueError as e:
            print("Error: %s" % e)
        return

    # Link the document to the expert
    if args.expert_learn:
        if len(args.expert_learn) < 2:
            parser.error(
                "--expert-learn needs EXPERT_NAME and " "at least one DOCUMENT_NAME"
            )
        exp_name, *doc_names = args.expert_learn
        exp_doc = ExpertDocuments(conn=CONN, exp_name=exp_name)
        results = exp_doc.link_multiple_documents(doc_names, hooks=print_cli_hooks())
        return

    # Unlink the document and the expert
    if args.expert_forget:
        if len(args.expert_forget) < 2:
            parser.error(
                "--expert-forget needs EXPERT_NAME and " "at least one DOCUMENT_NAME"
            )
        exp_name, *doc_names = args.expert_forget
        exp_doc = ExpertDocuments(conn=CONN, exp_name=exp_name)
        results = exp_doc.unlink_multiple_documents(doc_names, hooks=print_cli_hooks())
        return

    # Show all documents that is linked to the expert
    if args.expert_qualifications:
        exp_name = args.expert_qualifications
        exp_doc = ExpertDocuments(conn=CONN, exp_name=exp_name)
        doc_dict = exp_doc.all_documents_data_from_self()
        show_expert_documents_table(doc_dict)
        return

    # All documents for every experts
    if args.all_expert_documents:
        doc_dict = ExpertDocuments.all_documents_data_across_experts(CONN)
        show_all_documents_table(doc_dict)
        return

    # --- Logs ----------------------------------------------------------------
    # App logs, for general application logs and show prompt logs, to see the
    # formatted content the model actually reads.

    # App logs
    if args.app_logs:
        structured_app_log()
        return

    # Prompt logs
    if args.prompt_logs:
        return

    # --- CLI interaction with the agent --------------------------------------
    # By adding a prompt and session name (optional), to directly interact
    # with the agent in the CLI environment.
    #
    # See TUI section for when session name is given without prompt attached.

    # When prompt is provided
    if args.prompt:
        agent = Agent(sess_name=str(args.session[0]))
        answer = agent.ask(prompt=args.prompt, hooks=print_cli_hooks())
        return

    # --- TUI -----------------------------------------------------------------
    # Activate by not adding any prompt.
    #
    # Go to the dashboard if no arguments is added, by adding the session
    # argument it will go directly to the chat session in the TUI.

    if not args.prompt and not args.attachments:
        tar_sess = None

        if args.session:
            # Check if session name is valid
            tar_sess = args.session[0].strip("[]'")
            chat_logs = ChatSessionLogs(conn=CONN, llm_tknizr=LocalTokenizer(MODEL))
            sess_dict = chat_logs.get_all_existing_session_metadata() or {}
            names = [d["session_name"] for d in sess_dict.values()]
            if tar_sess not in names:
                print("Error: Session '%s' not found" % tar_sess)
                return

        # Start TUI with the optional arguments
        from src.app.tui.main_app import MainApp

        MainApp(sess_name=tar_sess).run()
        return
