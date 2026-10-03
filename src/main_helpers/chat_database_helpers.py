from src.config import postgres
from src.agent.chat_logs import ChatLogs
from src.agent.tokenizers import Tknizr


def update_session_used_tokens_on_prompt_submitted(
    chat_logs: ChatLogs, tknizr: Tknizr, cur
):
    """
    Only call at the end of every model response where the latest conversation
    entry is sent to the database.
    This function retrieves then fetches all conversations from current session
    and use the tokenizers to calculate and update the token count to the database.
    """
    msgs = chat_logs.get_active_conversations()
    chat_hist_tkns = tknizr.count_history_tokens(msgs) or 0

    # Update database
    cur.execute(
        """
        UPDATE chat_sessions
        SET session_used_tokens = %s
        WHERE LOWER(session_name) = LOWER(%s);
        """,
        (chat_hist_tkns, chat_logs.sess_name)
    )
    chat_logs.conn.commit()
