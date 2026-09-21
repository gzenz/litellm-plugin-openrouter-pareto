from __future__ import annotations

import os
import sys
from pathlib import Path


def _resolve_error_log_path() -> Path:
    env = os.environ.get("OPENROUTER_PARETO_ERROR_LOG")
    if env:
        return Path(env)
    import platformdirs

    return platformdirs.user_log_path("litellm-plugin-openrouter-pareto") / "errors.log"


def _resolve_decision_log_path() -> Path:
    env = os.environ.get("OPENROUTER_PARETO_ROUTING_LOG")
    if env:
        return Path(env)
    import platformdirs

    return platformdirs.user_log_path("litellm-plugin-openrouter-pareto") / "routing.log"


def _append_line(path: Path, line: str, what: str) -> None:
    """Append one line, owner-only, never following a symlink. Any failure is
    reported to stderr with the line suppressed and never raised: a logging
    problem must not fail the request that is being logged. The `what` label
    names the log so a stderr line identifies which one failed."""
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
            sys.stderr.write(
                f"openrouter-pareto: {what} permission tightening failed (body suppressed)\n"
            )
            return
        with os.fdopen(fd, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        sys.stderr.write(f"openrouter-pareto: {what} write failed (body suppressed)\n")


def or_error_log(line: str) -> None:
    _append_line(_resolve_error_log_path(), line, "error log")


def or_decision_log(line: str) -> None:
    _append_line(_resolve_decision_log_path(), line, "routing log")
