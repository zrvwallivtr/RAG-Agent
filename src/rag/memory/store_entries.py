from typing import Literal

from src.config.models import MODEL
from src.config.prompts import MEM_PROMPT, MEM_MANUAL_PROMPT

from .memory_formatting import format_coupled_memory_string

from src.libs.helpers.string_helpers import hash_string
from src.libs.helpers.database_helpers import fetch_one, execute

from src.agent.chat_session_logs import ChatSessionLogs
from src.agent.models.embed import embedding_content
from src.logger import app_logger, prompt_logger
from src.agent.models.llm import user_message, model_response_with_new_system_prompt_and_context
from src.format_context import separates_old_new_conversations

app_log = app_logger(f"{__name__}.app")

StoreMemoryResult = tuple[list[str], int, int, int]


class StoreEntries:
    def __init__(self, conn, chat_sess_logs: ChatSessionLogs):
        self._conn = conn

        self.chat_sess_logs   = chat_sess_logs

    def _is_memory_exist(self, new_hash: str) -> bool:
        """
        Check if the same memory was added before (same hash, accross sessions). Return True
        if a duplicate exist.
        """
        row = fetch_one(
            conn=self._conn,
            sql="SELECT id FROM memory WHERE content_hash = %s",
            params=(new_hash,)
        )
        if not row:
            return False
        return True

    def _add_memory_entry(
        self,
        embdings: list[float],
        cont_tkns: int,
        cont: str,
        cont_hash: str,
        ctgry: str,
        extraction: Literal["manual", "auto"]
    ):
        """Add new entry to 'memory' table."""
        row = fetch_one(
            conn=self._conn,
            sql=(
                """
                INSERT INTO memory (embeddings, prompt_tokens, content, content_hash, category, extraction)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id;
                """
            ),
            params=(str(embdings), cont_tkns, cont, cont_hash, ctgry, extraction),
        )

        if not row:
            return
        return str(row[0])

    def _embed_content_to_memory(
        self, cont: str, ctgry: str, extraction: Literal["manual", "auto"]
    ) -> tuple[str, int] | None:
        """Embeds texts and adds to memory logs."""
        cont_hash = hash_string(cont)
        if self._is_memory_exist(cont_hash):
            return

        response = embedding_content(cont)
        if not response:
            return
        cont, embdings, cont_tkns = response

        cont_hash = hash_string(cont)

        mem_id = self._add_memory_entry(
            embdings=embdings,
            cont_tkns=cont_tkns,
            cont=cont,
            cont_hash=cont_hash,
            ctgry=ctgry,
            extraction=extraction
        )
        if not mem_id:
            return
        return mem_id, cont_tkns

    def add_formatted_to_memory(
        self, conn, ext_out: str, extraction: Literal["manual", "auto"]
    ) -> tuple[list[str], int]:
        """Format memory entry(s) and add to memory. Return a list of created database ID(s)."""
        created_ids = []
        tol_tkns = 0
        count = 0

        mem_dict = format_coupled_memory_string(ext_out, extraction)

        for cont, ctgry in mem_dict.items():
            count += 1
            response = self._embed_content_to_memory(cont, ctgry, extraction)
            if not response:
                continue
            mem_id, tkn_used = response
            created_ids.append(mem_id)
            tol_tkns += tkn_used

        return created_ids, tol_tkns

    def store_extracted_memory(
        self,
        extraction: Literal["manual", "auto"],
        prompt: str | None = None
    ) -> StoreMemoryResult | None:
        """
        Depends of the system prompt to decide whether to extract memory
        automatically or manually.

        Prompt MUST be provided for manual extraction.
        """
        # Manual memory extraction
        if extraction == "manual":
            system_prompt = MEM_MANUAL_PROMPT

            if not prompt:
                app_log.warning("Memory extraction failed: No prompt provided")
                return

            new_convs = [user_message(prompt)]

        # Auto memory extraction
        else:
            system_prompt = MEM_PROMPT
            new_convs = self.chat_sess_logs.get_latest_conversation_turn()

        try:
            # Format prompt where old and new conversations are
            # indicated clearly.
            old_convs = self.chat_sess_logs.get_old_conversations()
            fmt_prompt = separates_old_new_conversations(
                old_convs=old_convs, new_convs=new_convs
            )

            # Model extracts key information from formatted prompt
            response = model_response_with_new_system_prompt_and_context(
                model=MODEL, sys_prompt=system_prompt, prompt=fmt_prompt
            )

            if not response:
                return None
            ext_out, p_tkns, o_tkns = response

            # Store memory for every memory in extracted output
            created_ids, total_tkn_used = self.add_formatted_to_memory(
                conn=self._conn, ext_out=ext_out, extraction=extraction
            )
            return created_ids, p_tkns, o_tkns, total_tkn_used

        except Exception as e:
            app_log.warning("Memory extraction synthesis failed: %s", e, exc_info=True)
            return None

    def delete_entry_by_id(self, id: str) -> bool:
        """Removes vector ID reference key directly from database."""
        row = execute(
            conn=self._conn,
            sql="DELETE FROM memory WHERE id = %s;",
            params=(id,)
        )
        return row is not None

    def delete_entries_by_ids(self, ids: list[str]) -> None:
        """Removes a list vector ID reference key directly from database."""
        del_count = 0

        for id in ids:
            row = self.delete_entry_by_id(id=id)
            if row:
                del_count += 1

        count = len(ids) - del_count
        if not count == 0:
            app_log.warning("Failed to delete %d memory entry(s)", count)
            return
        return
