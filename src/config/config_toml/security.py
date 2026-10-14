from src.config.files_and_directories import _cfg

# User toggle path safety feature (default: on):
#
# When enabled, agent can only read contents uploaded to the
# fixed uploaded directory ('UPLOAD_DIR' in the 
# files_and_directories module)
ENABLE_PATH_SAFETY = (
    _cfg.get("security", {}).get("enable_path_safety")
    or True
)
