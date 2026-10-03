# Command Line Interface Reference

```sh
agent [PROMPT] [OPTIONS]
```

- `agent` to open the [TUI](./TUI_REFERENCE.md).
- `agent "PROMPT"` sends a single prompt in teh defaul session.
- Flags that manage things (install, list, delete) run and exit without requir sending a prompt.

## Sessions

| Flag | Description |
| --- | --- |
| `-s`, `--session SESSION_NAME` | Use an existing session for this prompt. The default sesison is used without it. |
| `--new-session SESSION_NAME` | Create a session. If a prompt is given, it is sent right away. |
| `-ls`, `--list-session` | List all available sessions. |
| `--delete-session SESSION_NAME` | Delete a session. |
| `--delete-default-session` | Reset the default session. |

## Attachments

| Flag | Description |
| --- | --- |
| `-a`, `--attachments FILENAME...` | Attach one or more files to the prompt. A prompt is required, and all attached files must be in the upload directory. |
| `-la`, `--list-attachments` | List all uploaded documents stored in the database. |

## Models and tokenizers

| Flag | Description |
| --- | --- |
| `-m`, `--model MODEL_NAME` | Select the chat model (default: the `model` model in the `config.toml`). |
| `-lm`, `--list-models` | List all installed Ollama models. |
| `--install-model MODEL_NAME` | Pull an Ollama model, for example `ministral-3:3b` or `nomic-embed-text`. |
| `--install-tokenizer` | Install a fallback tokenizer that is specified in `~/.agent_app/config.toml` and tokenizers for all currently installed models. No argument required. |

## Output

| Flag | Description |
| --- | --- |
| `-q`, `--quiet` | Supress non-essential output. |
| `-v`, `--verbose` | Show debug-level logs. |
| `--no-color` | Disable coloured output. |
| `--width N` | Force the console width, useful for non-TTY output. |
| `-h`, `--help` | Show help. |

## Examples

```sh
# Set up
agent --install-model ministral-3:3b
agent --install-model nomic-embed-text
agent --install-tokenizer

# Chat
agent "What is Python?"
agent -s python_project "Where di we leave off?"

# Attach files (Must be in the uploads directory)
agent -a report.pdf notes.md "Summarise these files."

# Manage sessions
agent --new-session python_project
agent -ls
agent --delete-session python_project
```
