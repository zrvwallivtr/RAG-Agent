# Terminal User Interface Reference

Start the TUI by running `agent` with no prompt.

## Layout

- **Chat area:** All user and assistant messages. Each assistant message shows the tokens used for that turn.
- **Status bar:**
    - *Attachments (N pending):* files that will be sent with your next prompt.
    - *Token counter:* `used/max tokens` for the session. With live updating that includes the tokens in the prompt box.
- **Command bar:** Press `:` to open it, `Esc` to close it.

## Key bindings

These work in normal mode, when nothing is focused.

| Keys | Action |
| --- | --- |
| `j` / `k` | Scroll the chat down / up |
| `gg` | Jump to the top of the chat |
| `G` | Jump to the bottom of the chat |
| `i` or `a` | Focus the prompt box |
| `Space` then `a` | Open the attachments manager |
| `:` | Open the command bar (not available while the prompt box is focused) |

## Commands

Type a command in the command bar and press `Enter`.

| Command | Description |
| --- | --- |
| `:q`, `:quit` | Exist the app |
| `:d`, `:dashboard` | Go to the dashboard |
| `:s`, `:session` | Open the default session chat |
| `:s SESSION_NAME`, `:session SESSION_NAME` | Open the chat in the session `SESSION_NAME` |
| `:a PATH...`, `:attach PATH...` | Add files to the pending attachments (Chat screen only) |

## Attachments

Use `:a` or `:attach` to queue files, then send your prompt. Paths are resolved relative to the uploads directory, and quoted paths with spaces are supported. If any path is not found, none are added. Open the attachments manager (`Space` then `a`) to  review or remove pending files. Pending attachments are cleared once the prompt has been sent.
