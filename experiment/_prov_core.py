"""Shared provenance primitives — ONE definition, two callers.

``freeze.py`` (the registered-experiment spec) and ``provenance.py``
(the result-artifact header) both need git identity, a clean-tree
guard, and a canonical hash. Defining these once here — rather than
copy-pasting between modules that can silently drift — is the discipline
rule applied to provenance code itself: a second definition is a second
thing that can be wrong.

Nothing here depends on the working directory; ``PROJECT_ROOT`` is the
single anchor (from ``config``).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from typing import Any

from config import PROJECT_ROOT

# OS/editor cruft and bytecode do not affect what code produces a result
# or a predictor, so they do not make the tree "dirty" for provenance
# purposes. Any modified/added/deleted *source* path does.
_GIT_IGNORE = (".DS_Store", ".pyc", ".pyo")


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def git_clean() -> bool:
    """True iff no uncommitted changes to files that affect what code
    produced the artifact.

    The recorded ``git_sha`` must reproduce the artifact. Bytecode and
    editor cruft do not affect it (else pre-existing tracked junk would
    block every artifact forever); untracked *directories* (e.g.
    ``.idea/``) are not source. Any modified/added/deleted source path
    -- including an untracked source *file* -- makes it dirty.
    """
    out = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
    )
    for line in out.splitlines():
        path = line[3:].strip().strip('"')
        if " -> " in path:  # rename: check the destination
            path = path.split(" -> ", 1)[1]
        if path.endswith(_GIT_IGNORE):
            continue
        if path.endswith("/") or path.startswith(".idea/"):
            continue  # untracked dir -- not a source file
        return False
    return True


def canonical(obj: dict[str, Any], *, exclude: str) -> bytes:
    """Deterministic bytes for hashing: drop the named hash field, sort
    keys. The field that stores a hash must never be part of what it
    hashes — same trick freeze.py uses for ``spec_hash``."""
    body = {k: v for k, v in obj.items() if k != exclude}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
