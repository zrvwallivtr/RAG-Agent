from src import logger


app_log = logger.app_logger(f"{__name__}.app")


def _manage_token_budget(self, prompt: str, reserve: int) -> None:

    """
    Reserves extra tokens for model response. If exceeds
    maximum tokens, the model summarise previous messages
    to free up token space.
    """
    app_log.debug("Estimating token usage for session '%s'", self.sess_name)
    curr_hist_tkns = self.tknizr.count_history_tokens(self.chat_logs.get_actv_convs())
    if not curr_hist_tkns:
        return

    est_next = curr_hist_tkns + (len(prompt) // 4)

    if not self.tknizr.model_max_tkns:
        app_log.warning("Failed to load 'manage token budget feature': Model maximum token limit not set")
        return

    if self.tknizr.model_max_tkns - est_next - reserve < 0:
        app_log.info("Current tokens exceeds threshold")
        self.chat_logs.auto_compresss_active_conv()
        return
