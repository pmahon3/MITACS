#!/usr/bin/env python3
"""
lab_note.py — frontmatter helper for notes/lab/ session entries.

Owns nothing the agents (preregister / arbiter / thread-coordinator)
already own. Owns nothing the human writes (focus, prose body,
deferred items). Owns the *index* — the machine-derivable
frontmatter fields that go stale as a session continues past the
entry's initial write.

Modes:

    --new <date-or-session-id>    scaffold a new lab note with
                                  frontmatter pre-populated from
                                  registry + git state since the
                                  last lab note's commit.
    --refresh <path>              rewrite ONLY the frontmatter of
                                  an existing note; preserve the
                                  body verbatim. Refreshes the
                                  machine-derivable fields; never
                                  touches focus, deferred items,
                                  status_at_end.
    --diff <path>                 same as --refresh but print the
                                  proposed change and exit; do not
                                  write.

Discipline:
    - This tool does not commit. The user batches.
    - This tool does not touch the prose body. The body IS the
      session narrative; only the human writes it.
    - This tool does not infer focus, deferred items, or status:
      these are judgment, not derivable from git or the registry.

Run from anywhere in the repo. Resolves paths relative to the
repo root (the directory containing CLAUDE.md).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml

# Machine-derivable fields that --refresh rewrites:
DERIVABLE_FIELDS = {
    "commits",
    "preregistrations_touched",
    "memory_files_touched",
    "writeups_touched",
    "settled_this_session",
    "dispatched_this_session",
    "cross_programme_refs",
}

# Judgment fields that --refresh leaves alone:
JUDGMENT_FIELDS = {
    "date",
    "session_id",
    "duration_hours",
    "focus",
    "deferred_to_next_session",
    "status_at_end",
}

FRONTMATTER_RE = re.compile(
    r"^(<!--.*?-->\s*)?---\n(?P<frontmatter>.*?\n)---\n(?P<body>.*)\Z",
    re.DOTALL,
)


def repo_root() -> Path:
    """Find the repo root by walking up from this file to CLAUDE.md."""
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "CLAUDE.md").exists():
            return p
    raise RuntimeError("Could not find repo root (no CLAUDE.md upward)")


ROOT = repo_root()
LAB_DIR = ROOT / "notes" / "lab"
REGISTRY_DIR = ROOT / "notes" / "preregistrations"
MEMORY_DIR = Path.home() / ".claude" / "projects" / "-Users-pmahon-Research-Dynamics-MITACS" / "memory"


def git(*args: str, cwd: Optional[Path] = None) -> str:
    """Run git and return stdout. Empty string on nonzero exit."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd or ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout
    except FileNotFoundError:
        return ""


def last_lab_note_commit() -> Optional[str]:
    """Return the SHA of the most recent commit that added/modified a
    lab note other than the current target. None if no prior notes
    exist (first-ever session)."""
    out = git("log", "--format=%H", "--", "notes/lab/*.md")
    shas = [s for s in out.strip().splitlines() if s]
    return shas[0] if shas else None


def first_added_commit(target_path: Path) -> Optional[str]:
    """The commit that first added target_path. Returns the SHA of
    that commit's PARENT (i.e. the commit immediately before the
    session-block began). None if the file is untracked."""
    rel = target_path.relative_to(ROOT)
    out = git("log", "--diff-filter=A", "--format=%H", "--", str(rel))
    shas = [s for s in out.strip().splitlines() if s]
    if not shas:
        return None
    # shas[-1] is the oldest, i.e. the original-add commit. Its parent
    # is the session boundary: the commit before this lab note existed.
    add_sha = shas[-1]
    parent = git("rev-parse", f"{add_sha}^").strip()
    return parent or None


def session_boundary_for_refresh(target_path: Path, existing_fm: dict) -> Optional[str]:
    """Determine the session-start commit boundary for a --refresh.

    Three strategies in order of preference:

    1. If the frontmatter already lists commits, the FIRST entry's
       SHA names the session's first commit. The boundary is that
       commit's parent.

    2. Otherwise, the commit that first added the lab note. The
       boundary is that commit's parent.

    3. Otherwise (file untracked entirely), fall back to the most
       recent prior lab note's commit.

    Strategy 1 lets the human curate the session window once; the
    helper preserves it across refreshes.
    """
    commits_field = existing_fm.get("commits") or []
    if commits_field:
        first_line = commits_field[0]
        # Format: "abc1234 subject" — first whitespace-separated token.
        first_sha = first_line.split(None, 1)[0] if first_line else ""
        if first_sha:
            parent = git("rev-parse", f"{first_sha}^").strip()
            if parent:
                return parent
    parent = first_added_commit(target_path)
    if parent:
        return parent
    # File is untracked; fall back.
    return last_lab_note_commit()


def commits_since(since_sha: Optional[str]) -> list[tuple[str, str]]:
    """List (short_sha, subject) tuples for commits since since_sha,
    in chronological order (oldest first). If since_sha is None,
    returns all commits."""
    if since_sha:
        rng = f"{since_sha}..HEAD"
    else:
        rng = "HEAD"
    out = git("log", "--format=%h %s", "--reverse", rng)
    pairs: list[tuple[str, str]] = []
    for line in out.strip().splitlines():
        if not line:
            continue
        parts = line.split(" ", 1)
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
    return pairs


def changed_files_since(since_sha: Optional[str], path_prefix: str) -> list[str]:
    """List repo-relative paths under path_prefix that changed since
    since_sha (committed and uncommitted). Sorted unique."""
    paths: set[str] = set()
    # Committed changes:
    if since_sha:
        rng = f"{since_sha}..HEAD"
    else:
        rng = "HEAD"
    out = git("diff", "--name-only", rng, "--", path_prefix)
    for line in out.strip().splitlines():
        if line:
            paths.add(line)
    # Uncommitted (staged + unstaged):
    for diff_args in (["diff", "--name-only", "--cached"], ["diff", "--name-only"]):
        out = git(*diff_args, "--", path_prefix)
        for line in out.strip().splitlines():
            if line:
                paths.add(line)
    # Untracked (these often matter for new preregistrations):
    out = git("ls-files", "--others", "--exclude-standard", path_prefix)
    for line in out.strip().splitlines():
        if line:
            paths.add(line)
    return sorted(paths)


def preregistrations_touched_since(since_sha: Optional[str]) -> list[str]:
    """Directory names under notes/preregistrations/ that had any
    file change since since_sha."""
    changed = changed_files_since(since_sha, "notes/preregistrations/")
    dirs: set[str] = set()
    for p in changed:
        parts = Path(p).parts
        # parts[0] = "notes", parts[1] = "preregistrations", parts[2] = dir
        if len(parts) >= 3 and parts[1] == "preregistrations" and parts[2] != "_template":
            dirs.add(parts[2])
    return sorted(dirs)


def memory_files_touched_since(since_sha: Optional[str]) -> list[str]:
    """Memory file basenames that changed since since_sha.
    Memory lives outside the repo; we can only check git history of
    the memory dir if it's a git repo. Fall back to mtime check vs
    the commit's timestamp."""
    # Memory is in ~/.claude/projects/.../memory/; not in this repo.
    # Check mtimes against the timestamp of since_sha.
    if not MEMORY_DIR.exists():
        return []
    if since_sha:
        ts_out = git("show", "-s", "--format=%ct", since_sha)
        try:
            since_ts = int(ts_out.strip())
        except ValueError:
            return []
    else:
        since_ts = 0
    touched: list[str] = []
    for f in MEMORY_DIR.glob("*.md"):
        if f.stat().st_mtime > since_ts:
            touched.append(f.name)
    return sorted(touched)


def writeups_touched_since(since_sha: Optional[str]) -> list[str]:
    """writeup/tex/ paths that changed since since_sha."""
    changed = changed_files_since(since_sha, "writeup/")
    return [p for p in changed if p.endswith((".tex", ".md", ".bib"))]


def registry_state_snapshot() -> dict[str, str]:
    """Return {topic: status} from the audit registry. Empty dict if
    registry isn't usable."""
    try:
        out = subprocess.run(
            [str(ROOT / ".venv" / "bin" / "python"), "-m", "experiment.audit.registry", "list"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return {}
    state: dict[str, str] = {}
    for line in out.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Format: "  <topic>  <status>"
        parts = line.split()
        if len(parts) < 2:
            continue
        topic = parts[0]
        status = parts[-1]
        state[topic] = status
    return state


def settled_and_dispatched(
    since_sha: Optional[str],
) -> tuple[list[str], list[str]]:
    """Partition preregistrations_touched into (settled, dispatched).

    A preregistration is 'settled' if its current status is SETTLED
    and its arbiter.yaml was newly added since since_sha.

    A preregistration is 'dispatched' if its phase_a.yaml was newly
    added since since_sha (regardless of current status).

    Anything that doesn't fit either is just preregistrations_touched.
    """
    touched = preregistrations_touched_since(since_sha)
    state = registry_state_snapshot()
    if since_sha:
        rng = f"{since_sha}..HEAD"
    else:
        rng = "HEAD"
    added_files = set()
    out = git("log", "--diff-filter=A", "--name-only", "--format=", rng, "--", "notes/preregistrations/")
    for line in out.strip().splitlines():
        if line:
            added_files.add(line)
    # Also count uncommitted new files:
    out = git("ls-files", "--others", "--exclude-standard", "notes/preregistrations/")
    for line in out.strip().splitlines():
        if line:
            added_files.add(line)

    settled: list[str] = []
    dispatched: list[str] = []
    for topic in touched:
        topic_dir = f"notes/preregistrations/{topic}/"
        topic_added_files = [f for f in added_files if f.startswith(topic_dir)]
        has_new_phase_a = any(f.endswith("/phase_a.yaml") for f in topic_added_files)
        has_new_arbiter = any(f.endswith("/arbiter.yaml") for f in topic_added_files)
        current_status = state.get(topic, "")
        if has_new_arbiter and current_status == "SETTLED":
            settled.append(topic)
        elif has_new_phase_a:
            dispatched.append(topic)
    return settled, dispatched


def parse_frontmatter(text: str) -> tuple[Optional[dict], str, str]:
    """Return (frontmatter_dict, frontmatter_raw, body_text).
    frontmatter_dict is None if the file has no frontmatter."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return None, "", text
    raw = m.group("frontmatter")
    body = m.group("body")
    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError as e:
        raise SystemExit(f"Failed to parse frontmatter YAML: {e}")
    return data, raw, body


class _StringDumper(yaml.SafeDumper):
    """Dumper tuned for lab-note frontmatter style. Removes the
    implicit timestamp resolver so 'YYYY-MM-DD' strings emit
    unquoted, and overrides increase_indent so list items get
    the 2-space indent under their parent key that the existing
    lab notes use."""

    def increase_indent(self, flow=False, indentless=False):
        # The base class makes top-level list items flush-left.
        # Forcing indentless=False adds the 2-space indent.
        return super().increase_indent(flow=flow, indentless=False)


# Drop PyYAML's implicit timestamp resolver so date strings don't get
# quoted on output. Match the implicit_resolvers list on the base.
_StringDumper.yaml_implicit_resolvers = {
    k: [(tag, regexp) for tag, regexp in v if tag != "tag:yaml.org,2002:timestamp"]
    for k, v in yaml.SafeDumper.yaml_implicit_resolvers.items()
}


def _str_representer(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="")


_StringDumper.add_representer(str, _str_representer)


def _coerce_for_dump(value):
    """Recursively coerce date/datetime/non-string scalars to strings
    where the lab-note convention expects them (date, session_id).
    Lists and dicts are walked; other scalars pass through."""
    if isinstance(value, (_dt.date, _dt.datetime)):
        return value.isoformat() if isinstance(value, _dt.datetime) else value.strftime("%Y-%m-%d")
    if isinstance(value, list):
        return [_coerce_for_dump(v) for v in value]
    if isinstance(value, dict):
        return {k: _coerce_for_dump(v) for k, v in value.items()}
    return value


def render_frontmatter(data: dict) -> str:
    """Render a frontmatter dict to YAML in the template's field order.
    Matches the existing lab-note style: lists indented under their
    parent key (PyYAML's default-with-indent=2 mode)."""
    field_order = [
        "date", "session_id", "duration_hours", "focus",
        "commits", "preregistrations_touched", "memory_files_touched",
        "writeups_touched", "cross_programme_refs",
        "settled_this_session", "dispatched_this_session",
        "deferred_to_next_session", "status_at_end",
    ]
    ordered: dict = {}
    for k in field_order:
        if k in data:
            ordered[k] = _coerce_for_dump(data[k])
    # Preserve any extra keys (e.g. session-specific additions):
    for k, v in data.items():
        if k not in ordered:
            ordered[k] = _coerce_for_dump(v)
    return yaml.dump(
        ordered,
        Dumper=_StringDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=200,
        indent=2,
    )


def derive_frontmatter_updates(since_sha: Optional[str]) -> dict:
    """Compute the machine-derivable fields for the session since
    since_sha."""
    commits = commits_since(since_sha)
    commit_lines = [f"{sha} {subject}" for sha, subject in commits]
    settled, dispatched = settled_and_dispatched(since_sha)
    touched = preregistrations_touched_since(since_sha)
    # Anything in touched but neither in settled nor dispatched is
    # surfaced as preregistrations_touched (e.g. amendments to threads):
    pre_touched = sorted(set(touched))
    return {
        "commits": commit_lines,
        "preregistrations_touched": pre_touched,
        "memory_files_touched": memory_files_touched_since(since_sha),
        "writeups_touched": writeups_touched_since(since_sha),
        "settled_this_session": settled,
        "dispatched_this_session": dispatched,
        # cross_programme_refs is not derivable; leave alone.
    }


def cmd_new(arg: str) -> int:
    """Scaffold a new lab note. arg is either YYYY-MM-DD or
    YYYY-MM-DD-<letter>."""
    m = re.match(r"^(\d{4}-\d{2}-\d{2})(-[a-z])?$", arg)
    if not m:
        print(f"error: --new arg must be YYYY-MM-DD or YYYY-MM-DD-<letter>, got {arg!r}", file=sys.stderr)
        return 2
    date = m.group(1)
    suffix = m.group(2) or ""
    session_id = f"{date}{suffix or '-a'}"
    fname = f"{date}{suffix}.md" if suffix else f"{date}.md"
    target = LAB_DIR / fname
    if target.exists():
        print(f"error: {target} already exists; refusing to overwrite", file=sys.stderr)
        return 1
    since = last_lab_note_commit()
    derived = derive_frontmatter_updates(since)
    fm = {
        "date": date,
        "session_id": session_id,
        "focus": [],
        **derived,
        "cross_programme_refs": [],
        "deferred_to_next_session": [],
        "status_at_end": "in-progress",
    }
    body_template = """\

# Session {session_id} — short title

## What happened

(Free-form prose. Chronological narrative of the session.)

## Decisions

(Decisions locked in this session and their rationale.)

## Open questions

(Questions raised but not resolved.)

## Next session entry point

(One or more one-liners the next session should resume from, in
order of dependency. Mirror these into `deferred_to_next_session`
in the frontmatter.)
""".format(session_id=session_id)
    out = f"---\n{render_frontmatter(fm)}---\n{body_template}"
    target.write_text(out)
    print(f"wrote {target}")
    print(f"  session_id: {session_id}")
    print(f"  commits since last lab note ({since[:8] if since else 'BEGIN'}): {len(derived['commits'])}")
    print(f"  preregistrations touched: {len(derived['preregistrations_touched'])}")
    return 0


def cmd_refresh(path_arg: str, *, dry_run: bool = False) -> int:
    """Rewrite the frontmatter of an existing note; preserve body."""
    target = Path(path_arg)
    if not target.is_absolute():
        target = ROOT / target
    if not target.exists():
        # Try resolving under notes/lab/ if user passed a bare filename:
        alt = LAB_DIR / Path(path_arg).name
        if alt.exists():
            target = alt
        else:
            print(f"error: {target} not found", file=sys.stderr)
            return 1
    text = target.read_text()
    fm, _, body = parse_frontmatter(text)
    if fm is None:
        print(f"error: {target} has no frontmatter", file=sys.stderr)
        return 1
    since = session_boundary_for_refresh(target, fm)
    derived = derive_frontmatter_updates(since)
    new_fm = dict(fm)  # start from existing; preserve judgment fields
    for k, v in derived.items():
        new_fm[k] = v
    new_raw = render_frontmatter(new_fm)
    new_text = f"---\n{new_raw}---\n{body}"
    if dry_run:
        diff = _frontmatter_diff(fm, new_fm)
        if not diff:
            print("no changes")
            return 0
        for line in diff:
            print(line)
        return 0
    target.write_text(new_text)
    diff = _frontmatter_diff(fm, new_fm)
    print(f"refreshed {target}")
    if diff:
        for line in diff:
            print(f"  {line}")
    else:
        print("  (no frontmatter changes)")
    return 0


def _frontmatter_diff(old: dict, new: dict) -> list[str]:
    """Human-readable diff of two frontmatter dicts.
    Lists only fields that changed."""
    lines: list[str] = []
    for k in sorted(set(old) | set(new)):
        ov = old.get(k)
        nv = new.get(k)
        if ov == nv:
            continue
        if isinstance(ov, list) and isinstance(nv, list):
            added = [x for x in nv if x not in (ov or [])]
            removed = [x for x in (ov or []) if x not in nv]
            if added or removed:
                lines.append(f"{k}:")
                for x in added:
                    lines.append(f"  + {x}")
                for x in removed:
                    lines.append(f"  - {x}")
        else:
            lines.append(f"{k}: {ov!r} -> {nv!r}")
    return lines


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Frontmatter helper for notes/lab/ session entries.",
        epilog="Owns only the machine-derivable frontmatter index. Does not touch body, focus, deferred items, or status.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--new", metavar="YYYY-MM-DD[-x]", help="Scaffold a new lab note.")
    group.add_argument("--refresh", metavar="PATH", help="Rewrite frontmatter of an existing note.")
    group.add_argument("--diff", metavar="PATH", help="Show what --refresh would change; don't write.")
    args = parser.parse_args(argv)
    if args.new:
        return cmd_new(args.new)
    if args.refresh:
        return cmd_refresh(args.refresh, dry_run=False)
    if args.diff:
        return cmd_refresh(args.diff, dry_run=True)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
