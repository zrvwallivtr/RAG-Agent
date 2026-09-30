import shlex
from pathlib import Path


class Attachments:
    def _set_pending_attachments(self, paths: list[Path]) -> str:
        """Triggered by command input ':a or :attach', stores paths as a list."""
        self.pending_attchmnt.extend(paths)
        return f"{len(paths)} file(s) attached. {len(self.pending_attchmnt)} file(s) in total"


    def clear_pending_attachments(self) -> None:
        self.pending_attchmnt = []


    def attach_command(self, cmd: str, prefix: str) -> str:
        paths = cmd[len(prefix):].strip()

        if not paths:
            return "No file path(s) provided"

        try:
            all_paths = shlex.split(paths)
        except ValueError as e:
            return f"Invalid path syntax: {e}"

        attchmnts = []
        missing = []
        for p in all_paths:
            path = Path(p).expanduser()
            if path.exists():
                attchmnts.append(path)
            else:
                missing.append(p)

        if missing:
            return f"File(s) not found {', '.join(missing)}"

        msg = self._set_pending_attachments(attchmnts)
        return msg
