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
    "thread": {
        "schema", "written_at", "git_sha", "git_clean", "body_sha256",
        "references", "topic", "question", "root", "current_node_id",
        "state", "nodes", "branching_rules", "amendments",
    },
    # phase-fidelity outputs (Check T / Check P / Check R; see
    # .claude/agents/phase-fidelity.md). The schema is per-agent, not
    # per-check variant; filename may differ
    # (phase_fidelity_check_r.yaml, phase_fidelity_check_p.yaml, ...).
    # Workflow gap surfaced by Q2A Check R: arbiter precondition
    # hash_chain_intact calls registry.verify_entry, which requires the
    # schema to be known. Adding here closes the gap once.
    "phase_fidelity": {
        "schema", "written_at", "git_sha", "git_clean", "body_sha256",
        "references", "aggregate_verdict", "fidelity_coverage_passed",
    },
}

# Optional fields per schema (not required for verify, but checked if present)
OPTIONAL_TOP_LEVEL: dict[str, set[str]] = {
    "thread": {"state_history", "resolution"},
}


# Valid thread states (lifecycle state machine).
_THREAD_STATES = {"planning", "active", "exhausted", "resolved", "abandoned"}

# Valid per-node statuses within a thread.
_NODE_STATUSES = {"planned", "active", "settled", "closed", "exhausted"}


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


def stamp(
    data: dict[str, Any],
    *,
    self_path: Path | str | None = None,
) -> dict[str, Any]:
    """Return a copy of ``data`` with ``git_sha``, ``git_clean``, and
    ``body_sha256`` filled in (current values).

    ``self_path``: when the YAML being stamped is itself a registry
    file (the usual case — phase_a.yaml, result.yaml, etc.), pass its
    path here so that file's own untracked existence does NOT count
    toward ``git_clean = False``. The artifact's own existence in the
    working tree is not a source change of the state that produced it;
    this is the same self-reference issue ``freeze.py`` avoids by
    capturing clean-status BEFORE writing the spec. Here we capture
    it AFTER but exclude the artifact path explicitly — equivalent.
    """
    out = dict(data)
    out["git_sha"] = git_sha()
    out["git_clean"] = git_clean(
        exclude_paths=[self_path] if self_path is not None else None,
    )
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
        # Threads legitimately evolve via state advancement and append-
        # internal amendments. A reference to a thread.yaml should be
        # treated as "this thread exists; here's the snapshot I pinned
        # at write time" — not as a tamper-detection check on the
        # thread's current hash. We tolerate the hash mismatch IF the
        # reference's recorded hash matches any prior thread snapshot
        # in the amendments chain. (The phase_a's `thread` block carries
        # the snapshot pin separately for tamper-detection on the
        # specific version the experiment was registered against.)
        if ref_data.get("schema") == "thread":
            # Accept current hash OR any prior hash in amendments / state_history
            prior_hashes = {
                am.get("prior_tree_hash")
                for am in (ref_data.get("amendments") or [])
                if isinstance(am, dict)
            }
            prior_hashes |= {
                sh.get("prior_hash")
                for sh in (ref_data.get("state_history") or [])
                if isinstance(sh, dict)
            }
            prior_hashes.discard(None)
            if ref_hash == actual_hash or ref_hash in prior_hashes:
                continue
            errors.append(
                f"reference {ref_file} (thread): recorded hash "
                f"{ref_hash!r} matches neither current "
                f"({actual_hash!r}) nor any prior hash in amendments + "
                f"state_history ({sorted(prior_hashes)!r}). Thread "
                f"tampering suspected."
            )
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
# Thread support
# ---------------------------------------------------------------------------


def is_thread_dir(topic_dir: Path) -> bool:
    """A directory is a thread if it contains thread.yaml."""
    return (topic_dir / "thread.yaml").exists()


def verify_thread_schema(thread_data: dict[str, Any]) -> list[str]:
    """Structural checks on a thread.yaml beyond REQUIRED_TOP_LEVEL.

    Verifies:
      - state is one of the valid lifecycle states
      - each node has the required NodeSpec fields and valid status
      - branching_rules' keys are existing node_ids
      - branching_rules' values point to existing node_ids (or null)
      - amendments form a valid chain (prior_tree_hash matches)
      - current_node_id, if non-null, points at an existing node
    """
    errors: list[str] = []
    state = thread_data.get("state")
    if state not in _THREAD_STATES:
        errors.append(
            f"invalid thread state {state!r}; must be one of {sorted(_THREAD_STATES)}"
        )

    nodes = thread_data.get("nodes") or {}
    if not isinstance(nodes, dict) or not nodes:
        errors.append("nodes must be a non-empty dict of {node_id: NodeSpec}")
        return errors

    node_id_set = set(nodes.keys())
    for nid, node in nodes.items():
        if not isinstance(node, dict):
            errors.append(f"node {nid!r}: not a mapping")
            continue
        # required NodeSpec fields
        for fld in ("id", "name", "parent_id", "parent_branch", "status",
                    "phase_a_skeleton"):
            if fld not in node:
                errors.append(f"node {nid!r}: missing field {fld!r}")
        # status must be valid
        if node.get("status") not in _NODE_STATUSES:
            errors.append(
                f"node {nid!r}: invalid status {node.get('status')!r}; "
                f"must be one of {sorted(_NODE_STATUSES)}"
            )
        # parent_id must reference an existing node OR be null
        pid = node.get("parent_id")
        if pid is not None and pid not in node_id_set:
            errors.append(
                f"node {nid!r}: parent_id {pid!r} does not exist in nodes"
            )

    # branching_rules must be a dict {node_id: {branch_label: child_node_id_or_null}}
    rules = thread_data.get("branching_rules") or {}
    if not isinstance(rules, dict):
        errors.append("branching_rules must be a mapping")
    else:
        for src_nid, branches in rules.items():
            if src_nid not in node_id_set:
                errors.append(
                    f"branching_rules: source node {src_nid!r} does not exist"
                )
                continue
            if not isinstance(branches, dict):
                errors.append(
                    f"branching_rules[{src_nid!r}]: must be a mapping of {{branch_label: child_id_or_null}}"
                )
                continue
            for label, child in branches.items():
                if child is not None and child not in node_id_set:
                    errors.append(
                        f"branching_rules[{src_nid!r}][{label!r}]: child {child!r} does not exist"
                    )

    # current_node_id: null or existing node
    cni = thread_data.get("current_node_id")
    if cni is not None and cni not in node_id_set:
        errors.append(
            f"current_node_id {cni!r} does not exist in nodes"
        )

    # Amendment chain: each amendment's prior_tree_hash must match the
    # previous amendment's resulting body_sha256 (or the original
    # thread's pre-amendment hash for the first amendment).
    # We can't fully verify the chain without knowing the original-
    # creation body, so we just check structure.
    amendments = thread_data.get("amendments") or []
    if not isinstance(amendments, list):
        errors.append("amendments must be a list")
    else:
        for i, am in enumerate(amendments):
            if not isinstance(am, dict):
                errors.append(f"amendment[{i}]: not a mapping")
                continue
            for fld in ("amendment_id", "written_at", "git_sha",
                        "prior_tree_hash", "reason", "changes"):
                if fld not in am:
                    errors.append(f"amendment[{i}]: missing {fld!r}")

    return errors


def thread_state(topic_dir: Path) -> dict[str, Any]:
    """Rich state of a thread directory.

    Returns a dict with the thread-level state plus a derived view of
    each node's current status (cross-checking phase_a_path against the
    actual experiment directories).
    """
    thread_path = topic_dir / "thread.yaml"
    if not thread_path.exists():
        return {"is_thread": False}
    try:
        td = read_yaml(thread_path)
    except (ValueError, yaml.YAMLError) as e:
        return {"is_thread": True, "broken": str(e)}

    schema_errors = verify_thread_schema(td)
    nodes = td.get("nodes") or {}

    # For each node, cross-check phase_a_path against actual file
    node_states = {}
    for nid, node in nodes.items():
        pa_path = node.get("phase_a_path")
        if pa_path:
            full = (REGISTRY / pa_path).resolve() if not Path(pa_path).is_absolute() \
                else Path(pa_path)
            # Some authors store paths relative to PROJECT_ROOT, others relative to REGISTRY.
            # Try both.
            alt_paths = [
                Path(pa_path),
                REGISTRY / pa_path,
                REGISTRY.parent.parent / pa_path,
                PROJECT_ROOT / pa_path,
            ]
            exists = any(p.exists() for p in alt_paths)
        else:
            exists = False
        node_states[nid] = {
            "status": node.get("status"),
            "phase_a_path": pa_path,
            "phase_a_exists": exists,
        }

    return {
        "is_thread": True,
        "topic": td.get("topic"),
        "question": td.get("question"),
        "state": td.get("state"),
        "current_node_id": td.get("current_node_id"),
        "node_count": len(nodes),
        "node_states": node_states,
        "amendment_count": len(td.get("amendments") or []),
        "schema_errors": schema_errors,
    }


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def entry_status(topic_dir: Path) -> str:
    """One-word status for a topic.

    For threads (containing thread.yaml):
      THREAD:<state>  — where <state> is one of planning/active/exhausted/resolved/abandoned

    For experiments:
      SETTLED       — arbiter.yaml exists, finding_status == SETTLED, chain verifies
      PROVISIONAL   — arbiter.yaml exists, finding_status == PROVISIONAL
      AMBIGUOUS     — arbiter.yaml exists, verdict == AMBIGUOUS
      AWAITING-ARB  — result.yaml exists, arbiter.yaml does not
      AWAITING-RES  — phase_b.yaml exists, result.yaml does not
      AWAITING-B    — phase_a.yaml exists, phase_b.yaml does not
      EMPTY         — no phase_a.yaml
      BROKEN        — chain verification failed
    """
    ok, _ = verify_entry(topic_dir)
    if not ok:
        return "BROKEN"

    if is_thread_dir(topic_dir):
        ts = thread_state(topic_dir)
        if ts.get("schema_errors"):
            return "BROKEN"
        return f"THREAD:{ts.get('state', 'unknown')}"

    arb = topic_dir / "arbiter.yaml"
    res = topic_dir / "result.yaml"
    pb = topic_dir / "phase_b.yaml"
    pa = topic_dir / "phase_a.yaml"

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


def new_thread(slug: str, *, when: _dt.date | None = None) -> Path:
    """Create a new thread directory with the thread.yaml template.

    Convention: directory name is ``<YYYY-MM-DD>_<slug>-thread`` —
    the ``-thread`` suffix distinguishes lines of inquiry from
    individual experiments in directory listings.
    """
    if not TEMPLATE.exists():
        raise FileNotFoundError(
            f"template directory missing: {TEMPLATE}."
        )
    when = when or _dt.date.today()
    slug = slug.strip().lower().replace(" ", "-").replace("_", "-")
    if not slug or any(c.isspace() for c in slug):
        raise ValueError(f"invalid slug: {slug!r}")
    if slug.endswith("-thread"):
        # Don't double-suffix
        name = f"{when.isoformat()}_{slug}"
    else:
        name = f"{when.isoformat()}_{slug}-thread"
    target = REGISTRY / name
    if target.exists():
        raise FileExistsError(
            f"thread directory already exists: {target}. "
            f"Threads are append-internal — amend the existing thread.yaml "
            f"rather than creating a new directory."
        )
    target.mkdir(parents=True)
    shutil.copy(TEMPLATE / "thread.yaml", target / "thread.yaml")
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


# Thread CLI subcommands ----------------------------------------------


def _cmd_thread_new(args: argparse.Namespace) -> int:
    target = new_thread(args.slug)
    print(f"created {target.relative_to(PROJECT_ROOT)}")
    print(f"template copied: thread.yaml")
    print(f"fill the question, root, nodes, and branching_rules BEFORE "
          f"creating any node's phase_a.yaml. Invoke /audit thread-coordinator "
          f"to walk through the planning structure.")
    return 0


def _cmd_thread_status(args: argparse.Namespace) -> int:
    topic_dir = REGISTRY / args.topic
    if not topic_dir.exists():
        print(f"no such thread: {args.topic}", file=sys.stderr)
        return 2
    if not is_thread_dir(topic_dir):
        print(f"not a thread directory (no thread.yaml): {args.topic}",
              file=sys.stderr)
        return 2
    ts = thread_state(topic_dir)
    print(f"# thread: {args.topic}")
    print(f"  topic:         {ts.get('topic')}")
    print(f"  question:      {ts.get('question')}")
    print(f"  state:         {ts.get('state')}")
    print(f"  current_node:  {ts.get('current_node_id')}")
    print(f"  node_count:    {ts.get('node_count')}")
    print(f"  amendments:    {ts.get('amendment_count')}")
    if ts.get("schema_errors"):
        print(f"  schema_errors:")
        for e in ts["schema_errors"]:
            print(f"    - {e}")
        return 1
    print(f"  per-node:")
    for nid, ns in (ts.get("node_states") or {}).items():
        marker = " <-- current" if nid == ts.get("current_node_id") else ""
        pa_info = ""
        if ns.get("phase_a_path"):
            pa_info = f"  pa={'OK' if ns['phase_a_exists'] else 'MISSING'}"
        print(f"    {nid:>6s}  status={ns['status']:>10s}{pa_info}{marker}")
    return 0


def _cmd_thread_list(_args: argparse.Namespace) -> int:
    """List only thread entries."""
    if not REGISTRY.exists():
        print("(registry empty)")
        return 0
    threads = []
    for d in sorted(REGISTRY.iterdir()):
        if not d.is_dir() or d.name.startswith("_") or d.name == "README.md":
            continue
        if is_thread_dir(d):
            ts = thread_state(d)
            threads.append((d.name, ts.get("state"), ts.get("node_count", 0)))
    if not threads:
        print("(no threads)")
        return 0
    width = max(len(n) for n, _, _ in threads)
    for name, state, nc in threads:
        print(f"  {name:<{width}}  state={state:<10s}  nodes={nc}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list all entries (experiments + threads) with status")

    pv = sub.add_parser("verify", help="verify the hash chain of an entry")
    pv.add_argument("topic")

    ps = sub.add_parser("status", help="one-line status of an entry")
    ps.add_argument("topic")

    pn = sub.add_parser("new", help="create a new dated experiment entry")
    pn.add_argument("slug")

    # Thread subcommands
    pt = sub.add_parser("thread", help="thread (line of inquiry) operations")
    pt_sub = pt.add_subparsers(dest="thread_cmd", required=True)

    ptn = pt_sub.add_parser("new", help="create a new thread directory")
    ptn.add_argument("slug")

    pts = pt_sub.add_parser("status", help="rich status of a thread")
    pts.add_argument("topic")

    pt_sub.add_parser("list", help="list only thread entries")

    args = p.parse_args(argv)

    if args.cmd == "thread":
        return {
            "new": _cmd_thread_new,
            "status": _cmd_thread_status,
            "list": _cmd_thread_list,
        }[args.thread_cmd](args)

    return {
        "list": _cmd_list,
        "verify": _cmd_verify,
        "status": _cmd_status,
        "new": _cmd_new,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
