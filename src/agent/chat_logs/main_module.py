import mimetypes
import json
from pathlib import Path
from typing import Literal, Any

from src.config import models
from src.config import prompts

from .schema import ensure_schema
from .compression import Compression
from .chat_sessions_sql import ChatSessionsSQL
from .chat_logs_sql import ChatLogsSQL

from src.agent.tokenizers import LocalTokenizer
from src import format_context
from src.agent.models import llm
from src import logger

app_log = logger.app_logger(f"{__name__}.app")

MODEL           = models.MODEL
SYS_PROMPT      = prompts.SYS_PROMPT
COMPRESS_PROMPT = prompts.COMPRESS_PROMPT

_sys_prompt_tkns: int | None = None


class ChatSessionLogs:
    def __init__(self, conn, sess_name: str | None = None):
        self.conn = conn
        self.cur  = self.conn.cursor()  # some model still uses this

        self.model      = MODEL
        self.sys_prompt = SYS_PROMPT
        self.cmp_prompt = COMPRESS_PROMPT
        self.sess_name = sess_name.strip() if sess_name else "default_session"

        self.local_tknizr = LocalTokenizer(model=self.model)

        ensure_schema(self.conn)
        self.chat_sess   = ChatSessionsSQL(conn=self.conn, sess_name=self.sess_name, sys_prompt_tkns=self._system_prompt_tokens)
        self.chat_logs   = ChatLogsSQL(conn=self.conn, chat_sess=self.chat_sess, sys_prompt=self.sys_prompt)
        self.compression = Compression(self, chat_logs=self.chat_logs)

    def _system_prompt_tokens(self) -> int:
        global _sys_prompt_tkns
        if _sys_prompt_tkns is None:
            _sys_prompt_tkns = self.local_tknizr.count_string_tokens(text=SYS_PROMPT) or 0
        return _sys_prompt_tkns

    # Lookup chat sessions

    def get_session_id(self):
        return self.chat_sess.get_id()

    def get_or_create_session_id(self):
        return self.chat_sess.get_or_create_id()

    def get_session_name(self, sess_id: str):
        return self.chat_sess.get_name(sess_id)

    def get_session_id_from_name(self, sess_name: str):
        return self.chat_sess.get_id_from_name(sess_name)

    def get_session_data(self, sess_name: str | None = None):
        return self.chat_sess.get_data(sess_name)

    def get_all_existing_session_metadata(self):
        return self.chat_sess.get_all_metadata()

    # Write to chat sessions

    def create_session(self):
        return self.chat_sess.create()

    def update_session_used_tokens(self, n: int):
        return self.chat_sess.update_used_tokens(n)

    # Read chat logs

    def get_active_conversations(self):
        return self.chat_logs.active_conversations()

    def get_chat_history(self, filter: Literal["compressed", "not_compressed", "all"]):
        return self.chat_logs.conversation_history(filter)

    def get_latest_conversation_turn(self):
        return self.chat_logs.latest_conversation_turn()

    def get_old_conversations(self):
        return self.chat_logs.before_latest_conversations()

    def get_session_last_modified_time(self, sess_id: str | None = None):
        return self.chat_logs.last_modified_time(sess_id)

    def get_latest_modified_chat_session(self):
        return self.chat_logs.latest_modified_session()

    # Write to chat logs

    def add_conversation_turn(self, *args, **kwargs,):
        return self.chat_logs.add_turn(*args, **kwargs,)

    def clear_session_chat_logs(self):
        return self.chat_logs.clear_session_logs()

    # Compression section

    def compress_active_conversations(self, prompt: str, context: list[dict] | None = None):
        return self.compression.compress(prompt, context)

    def auto_compress_active_conversations(self):
        return self.compression.auto()
