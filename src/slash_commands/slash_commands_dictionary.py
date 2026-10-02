"""
All slash command-related strings are stored here. Always import related
strings from following dictionary to maintain consistency between CLI and
TUI.
"""

slash_cmds_dict = {
    "memorise": {
        "cmd": "/memorise",
        "no_prompt_warn": "Command '/memorise' aborted: No prompt was provided. Please specifiy instructions",
        "mock_response": "Information has been extracted and added to database",
        "removed_info": "Removed saved memory"
    },

    "recall": {
        "cmd": "/recall",
        "no_prompt_warn": "Command '/recall' aborted: No prompt was provided",
        "attachments_warn": "Command '/recall' does not support attachment uploads. Ignoring uploaded contents"
    },

    "compress": {
        "cmd": "/compress",
        "no_prompt_attachments_warn": "Command '/compress' aborted: Prompt must be provided if attachment is uploaded",
        "loading_info": "Compressing...",
    }
}
