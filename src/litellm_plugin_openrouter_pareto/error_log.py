from __future__ import annotations

import os
import sys
from pathlib import Path


def _resolve_log_path() -> Path:
    env = os.environ.get("OPENROUTER_PARETO_ERROR_LOG")
    if env:
        return Path(env)
    import platformdirs

    return platformdirs.user_log_path("litellm-plugin-openrouter-pareto") / "errors.log"


def or_error_log(line: str) -> None:
    path = _resolve_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            str(path),
            os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW,
            0o600,
        )
        try:
            os.fchmod(fd, 0o600)
        except OSError:
            os.close(fd)
            sys.stderr.write("openrouter-pareto: error log permission tightening failed (body suppressed)\n")
            return
        with os.fdopen(fd, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        sys.stderr.write("openrouter-pareto: error log write failed (body suppressed)\n")
