"""Pre-registration registry: read/write/verify ``notes/preregistrations/``.

The registry is append-only. Each topic lives in its own dated
subdirectory; each YAML file inside is hash-stamped with
``body_sha256`` (excluding the field itself from what it hashes) and
references prior files by hash. ``.claude/agents/*.md`` specify which
file is written by which agent; this module is the I/O substrate.

The hash chain makes the registry tamper-evident: ``verify_entry()``
recomputes every body hash and confirms every reference. ``/audit
arbiter`` refuses to arbitrate an entry whose chain is broken.

CLI usage::

    python -m experiment.audit.registry list
    python -m experiment.audit.registry verify <topic>
    python -m experiment.audit.registry new <slug>
    python -m experiment.audit.registry status <topic>
"""
from __future__ import annotations

import argparse
import datetime as _dt
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from config import PROJECT_ROOT

from .._prov_core import canonical, git_clean, git_sha, sha256_hex


REGISTRY = PROJECT_ROOT / "notes" / "preregistrations"
TEMPLATE = REGISTRY / "_template"

# Field that holds a file's own hash — must be excluded from the
# canonical bytes when computing the hash, same convention as
# ``freeze.py`` (``spec_hash``) and ``provenance.make_result``
# (``body_sha256`` over the body bytes).
_HASH_FIELD = "body_sha256"


# Required top-level fields by schema. Catches a write that forgot a
# field; not a complete schema check (the YAML templates and the agent
# .md specs are the canonical specification).
REQUIRED_TOP_LEVEL: dict[str, set[str]] = {
    "phase_a": {
        "schema", "written_at", "git_sha", "git_clean", "body_sha256",
        "topic", "research_question", "hypothesis", "variables",
        "data", "metric", "falsification_criterion",
        "corroboration_criterion", "ambiguous_region", "baselines",
        "stopping_criterion", "ci_method", "affects_registered_spec",
    },
    "phase_b": {
        "schema", "written_at", "git_sha", "git_clean", "body_sha256",
        "references", "algorithm", "hyperparameter_selection_rule",
        "random_seeds", "test_set_untouched",
        "deviations_from_phase_a", "pre_experiment_checklist",
    },
    "proponent": {
        "schema", "written_at", "git_sha", "git_clean", "body_sha256",
        "references", "forecast", "what_would_change_my_mind",
        "confidence",
    },
    "devils_advocate": {
        "schema", "written_at", "git_sha", "git_clean", "body_sha256",
        "references", "strongest_argument", "specific_failure_modes",
        "severity", "counter_prediction", "what_would_change_my_mind",
        "opportunity_cost",
    },
    "multiverse": {
        "schema", "written_at", "git_sha", "git_clean", "body_sha256",
        "references", "specification_grid", "cells", "n_cells_run",
        "median_effect", "fraction_same_sign", "verdict",
        "recommendation",
    },
    "result": {
        "schema", "written_at", "git_sha", "git_clean", "body_sha256",
        "references", "artifact", "code_path_audit", "primary_result",
    },
    "arbiter": {
        "schema", "written_at", "git_sha", "git_clean", "body_sha256",
        "references", "preconditions", "numerical_record", "verdict",
        "verdict_reasoning", "finding_status", "follow_up",
        "calibration_record", "memory_update_required",
    },
}


# ---------------------------------------------------------------------------
# Read / hash
# ---------------------------------------------------------------------------


def read_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML registry file. Refuses if missing or empty."""
    if not path.exists():
        raise FileNotFoundError(f"registry file not found: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return data


def compute_body_hash(data: dict[str, Any]) -> str:
    """SHA-256 over the canonical bytes of ``data``, excluding the
    ``body_sha256`` field itself. Same trick freeze.py and
    provenance.make_result use."""
    return sha256_hex(canonical(data, exclude=_HASH_FIELD))


def stamp(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``data`` with ``git_sha``, ``git_clean``, and
    ``body_sha256`` filled in (current values)."""
    out = dict(data)
    out["git_sha"] = git_sha()
    out["git_clean"] = git_clean()
    # body_sha256 is computed last, over the rest:
    out["body_sha256"] = compute_body_hash(out)
    return out


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


def verify_file(path: Path) -> tuple[bool, list[str]]:
    """Verify a single registry YAML file.

    Returns ``(ok, errors)``. Checks:

    1. File parses as YAML.
    2. ``schema`` field is one of the known schemas.
    3. All required top-level fields per the schema are present.
    4. ``body_sha256`` recomputes to the stored value (tamper-evidence).
    5. Each entry in ``references`` exists and its referenced
       ``body_sha256`` matches what is currently on disk.
    """
    errors: list[str] = []
    try:
        data = read_yaml(path)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        return False, [f"parse error: {e}"]

    schema = data.get("schema")
    if schema not in REQUIRED_TOP_LEVEL:
        return False, [f"unknown schema: {schema!r}"]

    missing = REQUIRED_TOP_LEVEL[schema] - set(data.keys())
    if missing:
        errors.append(
            f"missing required fields for schema {schema!r}: "
            f"{sorted(missing)}"
        )

    stored = data.get(_HASH_FIELD)
    recomputed = compute_body_hash(data)
    if stored != recomputed:
        errors.append(
            f"body_sha256 mismatch: header has {stored!r}, recomputed "
            f"{recomputed!r}. File was modified after stamping."
        )

    for ref in data.get("references", []) or []:
        ref_file = ref.get("file")
        ref_hash = ref.get("body_sha256")
        if not ref_file or not ref_hash:
            errors.append(f"reference missing file/hash: {ref!r}")
            continue
        ref_path = (path.parent / ref_file).resolve()
        try:
            ref_data = read_yaml(ref_path)
            actual_hash = ref_data.get(_HASH_FIELD)
        except (FileNotFoundError, ValueError) as e:
            errors.append(f"reference {ref_file} unreadable: {e}")
            continue
        if actual_hash != ref_hash:
            errors.append(
                f"reference {ref_file} hash mismatch: stored "
                f"{ref_hash!r}, on-disk {actual_hash!r}"
            )

    return not errors, errors


def verify_entry(topic_dir: Path) -> tuple[bool, dict[str, list[str]]]:
    """Verify every YAML file in a topic subdirectory."""
    if not topic_dir.is_dir():
        return False, {str(topic_dir): ["not a directory"]}
    out: dict[str, list[str]] = {}
    any_failed = False
    for f in sorted(topic_dir.glob("*.yaml")):
        ok, errs = verify_file(f)
        if not ok:
            out[f.name] = errs
            any_failed = True
    return not any_failed, out


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def entry_status(topic_dir: Path) -> str:
    """One-word status for a topic.

    SETTLED       — arbiter.yaml exists, finding_status == SETTLED, chain verifies
    PROVISIONAL   — arbiter.yaml exists, finding_status == PROVISIONAL
    AMBIGUOUS     — arbiter.yaml exists, verdict == AMBIGUOUS
    AWAITING-ARB  — result.yaml exists, arbiter.yaml does not
    AWAITING-RES  — phase_b.yaml exists, result.yaml does not
    AWAITING-B    — phase_a.yaml exists, phase_b.yaml does not
    EMPTY         — no phase_a.yaml
    BROKEN        — chain verification failed
    """
    arb = topic_dir / "arbiter.yaml"
    res = topic_dir / "result.yaml"
    pb = topic_dir / "phase_b.yaml"
    pa = topic_dir / "phase_a.yaml"

    ok, _ = verify_entry(topic_dir)
    if not ok:
        return "BROKEN"

    if arb.exists():
        data = read_yaml(arb)
        v = data.get("verdict")
        s = data.get("finding_status")
        if v == "AMBIGUOUS":
            return "AMBIGUOUS"
        if s == "SETTLED":
            return "SETTLED"
        return "PROVISIONAL"
    if res.exists():
        return "AWAITING-ARB"
    if pb.exists():
        return "AWAITING-RES"
    if pa.exists():
        return "AWAITING-B"
    return "EMPTY"


def list_entries() -> list[tuple[str, str]]:
    """All topic directories with their current status."""
    if not REGISTRY.exists():
        return []
    out: list[tuple[str, str]] = []
    for d in sorted(REGISTRY.iterdir()):
        if not d.is_dir() or d.name.startswith("_") or d.name == "README.md":
            continue
        out.append((d.name, entry_status(d)))
    return out


# ---------------------------------------------------------------------------
# Create a new entry
# ---------------------------------------------------------------------------


def new_entry(slug: str, *, when: _dt.date | None = None) -> Path:
    """Create a new topic directory with templates copied in.

    The slug is sluggified mildly (spaces → hyphens, lowercased). The
    directory name is ``<YYYY-MM-DD>_<slug>`` per the README convention.
    Refuses if the target directory already exists (append-only).
    """
    if not TEMPLATE.exists():
        raise FileNotFoundError(
            f"template directory missing: {TEMPLATE}. "
            f"Run from a properly initialized registry."
        )
    when = when or _dt.date.today()
    slug = slug.strip().lower().replace(" ", "-").replace("_", "-")
    if not slug or any(c.isspace() for c in slug):
        raise ValueError(f"invalid slug: {slug!r}")
    name = f"{when.isoformat()}_{slug}"
    target = REGISTRY / name
    if target.exists():
        raise FileExistsError(
            f"registry entry already exists: {target}. "
            f"Append-only — emit a new dated entry that references this one."
        )
    target.mkdir(parents=True)
    # Copy templates the analyst will fill in immediately. Other
    # templates are copied lazily by the relevant agent when needed.
    for fname in ("phase_a.yaml", "proponent.yaml", "devils_advocate.yaml"):
        shutil.copy(TEMPLATE / fname, target / fname)
    return target


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_list(_args: argparse.Namespace) -> int:
    entries = list_entries()
    if not entries:
        print("(registry empty)")
        return 0
    width = max(len(n) for n, _ in entries)
    for name, status in entries:
        print(f"  {name:<{width}}  {status}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    topic_dir = REGISTRY / args.topic
    if not topic_dir.exists():
        print(f"no such entry: {args.topic}", file=sys.stderr)
        return 2
    ok, errors = verify_entry(topic_dir)
    if ok:
        print(f"  {args.topic}: chain verified")
        return 0
    print(f"  {args.topic}: BROKEN")
    for fname, errs in errors.items():
        print(f"    {fname}:")
        for e in errs:
            print(f"      - {e}")
    return 1


def _cmd_status(args: argparse.Namespace) -> int:
    topic_dir = REGISTRY / args.topic
    if not topic_dir.exists():
        print(f"no such entry: {args.topic}", file=sys.stderr)
        return 2
    print(f"  {args.topic}: {entry_status(topic_dir)}")
    return 0


def _cmd_new(args: argparse.Namespace) -> int:
    target = new_entry(args.slug)
    print(f"created {target.relative_to(PROJECT_ROOT)}")
    print(f"templates copied: phase_a.yaml, proponent.yaml, devils_advocate.yaml")
    print(f"fill these BEFORE touching data. invoke /audit preregister "
          f"+ /audit devils-advocate to walk through the fields.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list all entries with status")

    pv = sub.add_parser("verify", help="verify the hash chain of an entry")
    pv.add_argument("topic")

    ps = sub.add_parser("status", help="one-line status of an entry")
    ps.add_argument("topic")

    pn = sub.add_parser("new", help="create a new dated entry")
    pn.add_argument("slug")

    args = p.parse_args(argv)
    return {
        "list": _cmd_list,
        "verify": _cmd_verify,
        "status": _cmd_status,
        "new": _cmd_new,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
