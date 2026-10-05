from src.agent import tokenizers


def get_tknizr_list() -> list | None:
    """Return all installed tokenizers as a list."""
    tknizr_dict = tokenizers.fetch_all_installed_tokenizers()
    if not tknizr_dict:
        return

    tknizr_list = []

    for tknizr in tknizr_dict:
        tknizr_list.append(tknizr["name"])
    return tknizr_list
