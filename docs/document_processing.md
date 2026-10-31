# Document Processing

Attached files are converted to plain text or Markdown, added to the current turn's context, and stored in the knowledge base so they can be retrieved later.

## Supported formats

| Type | Extensions |
| --- | --- |
| Plain text | `.txt` |
| Data and configuration | `.csv`, `.xlsx`, `.yaml`, `.toml`, `.xml` |
| Documents | `.pdf`, `.docx`, `.epub` |
| Code and scripts | `.py`, `.js`, `.ts`, `.tsx`, `.json`, `.md`, `.sh`, `.html`, `.css`, `.rs`, `.go` |
| Anything else | Read as plain text if it is a text-based format |

## What happens to an attachment

1. The file content is read and added to the prompt for that turn. It is not saved in the chat history. Only the filename, size and MIME type are stored with message.
2. After the agent replies, the content is embedded and saved to the knowledge base.
3. In later turns, relevant content from documents attached earlier in the session is retrieved automatically by similarity.

## Safeguards

- **Deduplication:** Content is hashed (SHA-256) before embedding. Re-uploading identical content is detected and skipped.
- **Path safety:** Document reads are resolved and validated against a fixed uploads directory to prevent path traversal.

## Parsing

With `enable_docling = true`, attachments are parsed with Docling. With `false`, the basic parsers are used. See the `[file_reader]` section of the [configuration reference](./CONFIGURATION.md).
