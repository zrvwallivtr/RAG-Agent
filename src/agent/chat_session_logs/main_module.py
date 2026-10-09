import mimetypes
import json
from pathlib import Path
from typing import Literal, Any, Callable

from src.config.models import MODEL
from src.config.prompts import SYS_PROMPT, COMPRESS_PROMPT

from .schema import ensure_schema
from .compression import Compression
from .lookup_chat_sessions import LookupChatSessions
from .write_chat_sessions import WriteChatSessions
from .read_chat_logs import ReadChatLogs
from .write_chat_logs import WriteChatLogs

from src.agent.models.llm import TokenCallback
from src.agent.tokenizers import LocalTokenizer
from src.agent.models import llm
from src import logger

app_log = logger.app_logger(f"{__name__}.app")

_sys_prompt_tkns: int | None = None


class ChatSessionLogs:
    def __init__(self, conn, sess_name: str | None = None):
        self.conn = conn

        self.model      = MODEL
        self.sys_prompt = SYS_PROMPT
        self.cmp_prompt = COMPRESS_PROMPT
        self.sess_name  = sess_name.strip() if sess_name else "default_session"

        self.local_tknizr = LocalTokenizer(model=self.model)

        ensure_schema(self.conn)

        # Chat session classes
        self.write_chat_sess = WriteChatSessions(
            conn=self.conn, sess_name=self.sess_name, sys_prompt_tkns=self._system_prompt_tokens
        )
        self.lookup_chat_sess = LookupChatSessions(
            conn=self.conn, sess_name=self.sess_name, sys_prompt_tkns=self._system_prompt_tokens
        )

        # Chat log classes
        self.read_chat_logs = ReadChatLogs(
            conn=self.conn, lookup_chat_sess=self.lookup_chat_sess, sys_prompt=self.sys_prompt
        )
        self.write_chat_logs = WriteChatLogs(conn=self.conn, lookup_chat_sess=self.lookup_chat_sess)

        self.compression = Compression(self, write_chat_logs=self.write_chat_logs)

    def _system_prompt_tokens(self) -> int:
        global _sys_prompt_tkns
        if _sys_prompt_tkns is None:
            _sys_prompt_tkns = self.local_tknizr.count_string_tokens(text=SYS_PROMPT) or 0
        return _sys_prompt_tkns

    # --- Lookup chat sessions -------------------------------------

    def get_session_id(self):
        return self.lookup_chat_sess.get_id()

    def get_or_create_session_id(self):
        return self.lookup_chat_sess.get_or_create_id()

    def get_session_name(self, sess_id: str):
        return self.lookup_chat_sess.get_name(sess_id)

    def get_session_id_from_name(self, sess_name: str):
        return self.lookup_chat_sess.get_id_from_name(sess_name)

    def get_session_data(self, sess_name: str | None = None):
        return self.lookup_chat_sess.get_data(sess_name)

    def get_all_existing_session_metadata(self):
        return self.lookup_chat_sess.get_all_metadata()

    # --- Write to chat sessions -----------------------------------

    def create_session(self):
        return self.write_chat_sess.create()

    def update_session_used_tokens(self):
        msgs = self.get_active_conversations()
        sess_used_tkns = self.local_tknizr.count_history_tokens(msgs)
        return self.write_chat_sess.update_used_tokens(sess_used_tkns or 0)

    # --- Read chat logs -------------------------------------------

    def get_active_conversations(self):
        return self.read_chat_logs.active_conversations()

    def get_chat_history(self, filter: Literal["compressed", "not_compressed", "all"]):
        return self.read_chat_logs.conversation_history(filter)

    def get_latest_conversation_turn(self):
        return self.read_chat_logs.latest_conversation_turn()

    def get_old_conversations(self):
        return self.read_chat_logs.before_latest_conversations()

    def get_session_last_modified_time(self, sess_id: str | None = None):
        return self.read_chat_logs.last_modified_time(sess_id)

    def get_latest_modified_chat_session(self):
        return self.read_chat_logs.latest_modified_session()

    # --- Write to chat logs ---------------------------------------

    def add_conversation_turn(self, *args, **kwargs,):
        return self.write_chat_logs.add_turn(*args, **kwargs,)

    def clear_session_chat_logs(self):
        return self.write_chat_logs.clear_session_logs()

    # --- Compression ----------------------------------------------

    def default_compress_active_conversations(
        self,
        unfmt_prompt: str,
        cmp_method: Literal["default", "instruct"],
        callback: TokenCallback | None = None,
    ):
        return self.compression.compress_with_pre_written(
            unfmt_prompt=unfmt_prompt, cmp_method="default", callback=callback
        )

    def instruct_compress_active_conversations(
        self,
        clean_prompt: str,
        unfmt_prompt: str,
        cmp_method: Literal["default", "instruct"],
        attchmnt_dict: dict[Path, dict[str, str]] | None = None,
        callback: TokenCallback | None = None,
    ):
        return self.compression.compress(
            clean_prompt=clean_prompt,
            unfmt_prompt=unfmt_prompt,
            attchmnt_dict=attchmnt_dict,
            cmp_method=cmp_method,
            callback=callback
        )

    def auto_compress_active_conversations(
        self, unfmt_prompt: str
    ):
        return self.compression.auto_compress(unfmt_prompt=unfmt_prompt)
