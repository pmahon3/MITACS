"""``make_result()`` — the standard provenance envelope for every
claim-grade result artifact (raises C1–C4 to C5's bar).

A result file under ``experiment/results/`` is a *record of what a
specific code+data state produced*. For a reviewer to trust a quoted
number without rerunning, the file must answer two distinct questions,
and they must not be conflated (see ``mitacs-provenance-requirements``):

  * **inputs_fingerprint** — "rerun *these* inputs, get this result":
    a canonical hash over {git_sha, frozen_spec hash, library versions,
    declared RNG seeds, plus any caller-supplied input fingerprint}.
    This is the reproducibility claim, the analogue of C5's spec_hash.
  * **body_sha256** — "this file was not edited after it was written":
    a hash over the result body bytes, recorded in the header and
    recomputed on load. Pure tamper-evidence for *this artifact*, the
    analogue of ``freeze.load_verified()``.

Both are independently verifiable; neither implies the other.

Integrity policy is **strict, mirroring freeze.py**: ``make_result``
*refuses* to write from a dirty source tree, because a recorded
``git_sha`` that does not reproduce the artifact is the exact integrity
hole this module exists to close. A result is not weaker than a frozen
spec — commit first.

A one-line **grade banner** (CLAIM-GRADE / METHOD / INSPECTION-ONLY) is
mandatory and appears in the header so an artifact can never be silently
mis-cited above its grade.

Usage::

    from experiment.provenance import make_result, Grade

    body = render_my_result()                 # the human-readable text
    make_result(
        path=RESULTS / "backtest_postcutoff_2025-01-01_2026-05-16.txt",
        grade=Grade.CLAIM,
        title="Post-cutoff multi-step backtest vs Ontario actuals",
        body=body,
        inputs={"data_fingerprint": actuals_fingerprint(CUTOFF)},
        seeds={"numpy": 0},                    # {} if the result is RNG-free
    )

    # later, by a reviewer:
    from experiment.provenance import load_verified_result
    hdr, body = load_verified_result(path)     # raises if tampered
"""
from __future__ import annotations

import enum
import json
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from ._prov_core import canonical, git_clean, git_sha, sha256_hex

# Libraries whose version changes can move a result. edynamics is the
# editable sibling install (see CLAUDE.md); numpy/pandas drive the
# numerics; scipy is used by the diagnostics/gates.
_PINNED_LIBS = ("numpy", "pandas", "scipy", "edynamics")

_HEADER_OPEN = "<<<PROVENANCE"
_HEADER_CLOSE = "PROVENANCE>>>"


class Grade(str, enum.Enum):
    """Artifact grade — determines how a reader may cite it.

    CLAIM       defensible in the writeup / to reviewers (needs R+A+I).
    METHOD      an architectural/methodological assertion (A+I, R where
                it rests on a computation).
    INSPECTION  explicitly exploratory; MUST NOT be quoted as a result.
    """

    CLAIM = "CLAIM-GRADE"
    METHOD = "METHOD/DESIGN"
    INSPECTION = "INSPECTION-ONLY"


def _lib_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in _PINNED_LIBS:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "NOT-INSTALLED"
    return out


def _frozen_spec_hash() -> str | None:
    """The registered experiment's spec hash if a frozen spec exists.

    Recorded so a result is bound to the predictor registration it was
    produced under. ``None`` for results that do not depend on the
    frozen predictor (e.g. the synthetic validation gates)."""
    try:
        from .freeze import load_verified

        return load_verified()["spec_hash"]
    except Exception:
        return None


def build_header(
    *,
    grade: Grade,
    title: str,
    inputs: dict[str, Any] | None,
    seeds: dict[str, Any] | None,
    body: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct the provenance header dict (no I/O).

    ``inputs_fingerprint`` is a canonical hash over the reproducibility-
    determining inputs; ``body_sha256`` is a separate hash over the
    result text. The two hash fields are excluded from what they hash.
    """
    sha = git_sha()
    clean = git_clean()
    repro = {
        "git_sha": sha,
        "frozen_spec_hash": _frozen_spec_hash(),
        "lib_versions": _lib_versions(),
        "seeds": seeds or {},
        "inputs": inputs or {},
    }
    header: dict[str, Any] = {
        "schema": "mitacs.experiment.result_provenance/1",
        "grade": grade.value,
        "title": title,
        "generated_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "git_sha": sha,
        "git_clean": clean,
        "reproducibility": repro,
        # canonical hash over `reproducibility` only: "rerun these
        # inputs -> this result". Excludes nothing of itself (lives at
        # the top level), so a plain canonical() over the sub-dict.
        "inputs_fingerprint": sha256_hex(
            canonical(repro, exclude="__none__")
        ),
        # tamper-evidence over the human-readable body bytes.
        "body_sha256": sha256_hex(body.encode()),
    }
    if extra:
        header["extra"] = extra
    return header


def _render(header: dict[str, Any], body: str) -> str:
    h = json.dumps(header, indent=2, sort_keys=True)
    return f"{_HEADER_OPEN}\n{h}\n{_HEADER_CLOSE}\n\n{body}"


def make_result(
    *,
    path: Path,
    grade: Grade,
    title: str,
    body: str,
    inputs: dict[str, Any] | None = None,
    seeds: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Write ``body`` to ``path`` with a verifiable provenance header.

    Refuses a dirty source tree (the recorded ``git_sha`` would not
    reproduce the artifact) unless ``allow_dirty=True`` is passed
    *explicitly* — reserved for INSPECTION-ONLY scratch captures, never
    for CLAIM/METHOD artifacts (enforced below).

    Returns the header dict (so callers can echo the fingerprints).
    """
    if grade is not Grade.INSPECTION and allow_dirty:
        raise ValueError(
            f"allow_dirty is only permitted for INSPECTION-ONLY "
            f"artifacts; {grade.value} results must be reproducible "
            f"from a clean tree."
        )
    if not allow_dirty and not git_clean():
        raise SystemExit(
            "refusing to write a provenanced result from a dirty tree: "
            "the recorded git_sha would not reproduce it. Commit or "
            "stash source changes first (this is the same bar freeze.py "
            "holds — a result is not weaker than a frozen spec)."
        )
    header = build_header(
        grade=grade,
        title=title,
        inputs=inputs,
        seeds=seeds,
        body=body,
        extra=extra,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render(header, body))
    return header


def load_verified_result(path: Path) -> tuple[dict[str, Any], str]:
    """Read a result file, recomputing and checking ``body_sha256``.

    Raises ``ValueError`` if the body was edited after the header was
    written (tamper-evident, mirroring ``freeze.load_verified``).
    Returns ``(header, body)``.
    """
    text = path.read_text()
    if not text.startswith(_HEADER_OPEN):
        raise ValueError(f"{path} has no provenance header")
    _, rest = text.split(_HEADER_OPEN + "\n", 1)
    hjson, body = rest.split("\n" + _HEADER_CLOSE + "\n", 1)
    body = body.lstrip("\n")
    header = json.loads(hjson)
    recomputed = sha256_hex(body.encode())
    if recomputed != header.get("body_sha256"):
        raise ValueError(
            f"{path}: body hash mismatch (header says "
            f"{header.get('body_sha256')}, body hashes to {recomputed}). "
            f"The result was modified after generation — integrity "
            f"compromised."
        )
    return header, body
