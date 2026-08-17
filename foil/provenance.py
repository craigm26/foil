"""Provenance stamp for result files.

Every runner attaches `stamp(__file__)` to its result JSON, so "which code
produced this file" is a lookup instead of archaeology. Historical result
files predate this module and do not carry it; that absence is itself the
record of when the practice started (2026-08-17 tooling review, item 5).
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=Path(__file__).parent.parent,
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def stamp(script: str | None = None) -> dict:
    """Provenance dict for embedding in a result file.

    No timestamp on purpose: results are committed to git, and the commit
    supplies a signed time. A wall-clock here would just disagree with it.
    """
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "script": Path(script).name if script else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
