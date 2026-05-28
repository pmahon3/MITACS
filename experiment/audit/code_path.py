"""Diagnostic-vs-production drift detector.

Implements the single most load-bearing discipline rule in this
project's history (``CLAUDE.md`` §"Discipline rule"):

    A diagnostic that reimplements production logic is a hypothesis,
    not a finding, until confirmed through the production code path.

Given a script, inspect (statically, via the ``ast`` module) whether
it calls the whitelisted production functions or reimplements
equivalent logic inline. Returns a verdict + structured evidence that
``.claude/agents/code-path-auditor.md`` consumes to make the final
call.

Heuristic, not authoritative. The LLM agent reads this output and
applies judgment for edge cases (mixed scripts, novel functions not
in the whitelist, etc.). Errors of omission (a production call we
didn't whitelist) are addressed by maintaining the whitelist below;
the lab adds entries as new production surfaces emerge.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import enum
import re
import sys
from pathlib import Path
from typing import Iterable

from config import PROJECT_ROOT


# ---------------------------------------------------------------------------
# Production whitelist — symbols whose CALL counts as "production path"
# ---------------------------------------------------------------------------
#
# Drawn directly from ``.claude/agents/code-path-auditor.md``. Maintain
# this list as new production entry points emerge; orphans (functions
# removed from the codebase) get flagged by the agent during re-audit.

PRODUCTION_SYMBOLS: set[str] = {
    # experiment.backtest
    "experiment.backtest.main",
    "experiment.backtest._complete_delivery_days",
    # experiment.predict
    "experiment.predict.predict",
    "experiment.predict._build_pre_cutoff",
    "experiment.predict._daytype",
    # experiment.predict_multistep
    "experiment.predict_multistep.predict_multistep",
    "experiment.predict_multistep.compose",
    # experiment._actuals
    "experiment._actuals.load_actuals",
    "experiment._actuals.load_pre_cutoff_actuals",
    "experiment._actuals.zscore_params",
    "experiment._actuals.zscore_transform",
    "experiment._actuals.mu_at",
    "experiment._actuals.sigma_at",
    # processing.innovations.estimator
    "processing.innovations.estimator.build_local_gaussian_semigroup",
    "processing.innovations.estimator.local_drift_and_diffusion",
    "processing.innovations.estimator.diffusion_spectrum",
    "processing.innovations.estimator._theta_loo_cv",
    "processing.innovations.estimator._local_fit_at",
    "processing.innovations.estimator.global_ols_fit",
    "processing.innovations.estimator.student_t_mle_fit",
    "processing.innovations.estimator.mixture_2_gaussian_mle_fit",
    "processing.innovations.estimator.mixture_3_gaussian_mle_fit",
    "processing.innovations.estimator.kde_residual_fit",
    "processing.innovations.estimator.patra_sen_fit",
    "processing.innovations.estimator.sliced_inverse_regression_fit",
    # processing.innovations.validation
    "processing.innovations.validation.synthetic",
    "processing.innovations.validation.rebaseline",
    # experiment.freeze / experiment.provenance
    "experiment.freeze.load_verified",
    "experiment.freeze.register",
    "experiment.provenance.make_result",
    "experiment.provenance.load_verified_result",
    "experiment.provenance.build_header",
    # experiment.fourier_climatology
    "experiment.fourier_climatology.fit_fourier_params",
}

# Short names (e.g. ``mu_at``) also count when imported from a
# production module. Computed once on import.
PRODUCTION_SHORT_NAMES: set[str] = {s.rsplit(".", 1)[-1] for s in PRODUCTION_SYMBOLS}


# ---------------------------------------------------------------------------
# Suspicious-reimplementation heuristics
# ---------------------------------------------------------------------------
#
# Patterns that, in this codebase, almost always indicate the script
# re-derives something a production function already computes.

REIMPLEMENTATION_PATTERNS: list[tuple[str, str]] = [
    # Reconstructing the climatology mean
    (r"\bgroupby\s*\(\s*\[[^\]]*month[^\]]*hour[^\]]*\]\s*\)\s*\..*mean",
     "computes month×hour mean — should call zscore_params or mu_at"),
    # Reconstructing climatology std
    (r"\bgroupby\s*\(\s*\[[^\]]*month[^\]]*hour[^\]]*\]\s*\)\s*\..*std",
     "computes month×hour std — should call zscore_params or sigma_at"),
    # Inline z-score from raw (raw - μ) / σ
    (r"\(\s*\w+\s*-\s*mu\w*\s*\)\s*/\s*s(igma|d)\w*",
     "inline z-score — should call zscore_transform"),
    # np.linalg.lstsq on demand-shaped arrays (the local-linear OLS)
    (r"np\.linalg\.lstsq\s*\(",
     "inline OLS — verify it's not reimplementing local_drift_and_diffusion"),
    # Manual residual covariance
    (r"\.T\s*@\s*\w+\s*/\s*(max\s*\()?(\s*len\s*\(\w+\)\s*-\s*1)",
     "inline residual covariance — should call local_drift_and_diffusion"),
    # _fit_one and friends (the historical reimplementation pattern)
    (r"def\s+_fit_one\b",
     "private _fit_one function — historical reimplementation pattern"),
    (r"def\s+_local_diag\b",
     "private _local_diag function — historical reimplementation pattern"),
    # Manual delay embedding construction
    (r"np\.stack\s*\(\s*\[\s*vals?\[[^\]]*-\s*j\]",
     "manual lag-vector construction — verify it's not bypassing "
     "_build_pre_cutoff"),
]


# ---------------------------------------------------------------------------
# Verdict types
# ---------------------------------------------------------------------------


class Verdict(str, enum.Enum):
    PRODUCTION_PATH = "PRODUCTION-PATH"
    REIMPLEMENTED = "REIMPLEMENTED"
    MIXED = "MIXED"
    NO_COMPUTATION = "NO-COMPUTATION"  # script just loads + summarises


@dataclasses.dataclass
class CodePathReport:
    path: Path
    verdict: Verdict
    production_calls: list[tuple[str, int]]      # (fully-qualified name, line)
    production_imports: list[str]                # imported names
    suspicious_inline: list[tuple[str, int, str]]  # (pattern desc, line, snippet)

    def render(self) -> str:
        out = [f"## Code-Path Audit: {self.path}",
               f"",
               f"### Verdict: {self.verdict.value}",
               f""]
        if self.production_imports:
            out.append("### Production imports")
            for name in self.production_imports:
                out.append(f"- `{name}`")
            out.append("")
        if self.production_calls:
            out.append("### Production calls")
            for name, line in self.production_calls:
                out.append(f"- {self.path.name}:{line}: `{name}`")
            out.append("")
        if self.suspicious_inline:
            out.append("### Suspicious inline computation")
            for desc, line, snippet in self.suspicious_inline:
                out.append(f"- {self.path.name}:{line}: {desc}")
                out.append(f"    `{snippet.strip()[:120]}`")
            out.append("")
        out.append("### Notes")
        if self.verdict is Verdict.PRODUCTION_PATH:
            out.append(
                "Production-path verified. The claim may be CITED "
                "subject to the other audits (arbiter, multiverse)."
            )
        elif self.verdict is Verdict.REIMPLEMENTED:
            out.append(
                "PROVISIONAL. The script reimplements production logic. "
                "Rerun the computation through production functions "
                "before citing. Suggested fixes: replace each suspicious "
                "inline block above with a call to the corresponding "
                "production function."
            )
        elif self.verdict is Verdict.MIXED:
            out.append(
                "PROVISIONAL. The script calls some production functions "
                "but also reimplements parts of the pipeline inline. "
                "Identify which inline blocks could be replaced before "
                "citing."
            )
        else:
            out.append(
                "Script appears to LOAD production artifacts and "
                "summarise without re-deriving anything. This is "
                "PRODUCTION-PATH on the substantive axis; verify that "
                "the loaded artifact itself was produced by production."
            )
        return "\n".join(out)


# ---------------------------------------------------------------------------
# AST walking
# ---------------------------------------------------------------------------


def _qualname_from_attr(node: ast.AST) -> str | None:
    """Reconstruct a dotted name from an ast.Attribute / ast.Name chain.

    ``a.b.c`` -> ``"a.b.c"``; anything else -> ``None``.
    """
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _walk_imports(tree: ast.AST) -> dict[str, str]:
    """Map local name -> fully-qualified name for relevant imports.

    Only tracks imports from modules whose prefix appears in
    PRODUCTION_SYMBOLS, so the dict stays small.
    """
    out: dict[str, str] = {}
    prod_modules = {s.rsplit(".", 1)[0] for s in PRODUCTION_SYMBOLS}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if not any(mod == p or mod.startswith(p + ".") or
                       p.startswith(mod + ".") for p in prod_modules):
                continue
            for alias in node.names:
                local = alias.asname or alias.name
                out[local] = f"{mod}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in prod_modules:
                    local = alias.asname or alias.name
                    out[local] = alias.name
    return out


def _calls(tree: ast.AST,
           imports: dict[str, str]) -> list[tuple[str, int]]:
    """All function calls that resolve to a production symbol.

    Returns (fully-qualified name, line number) pairs.
    """
    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # f(...) where f is an imported name
        if isinstance(func, ast.Name):
            qn = imports.get(func.id)
            if qn and qn in PRODUCTION_SYMBOLS:
                out.append((qn, node.lineno))
            # NOTE: do NOT fall back to PRODUCTION_SHORT_NAMES here.
            # Names like ``main``, ``register``, ``compose`` are
            # whitelisted at fully-qualified form but are also common
            # local function names; matching on the leaf alone would
            # mark every script's own ``main()`` as production.
            continue
        # mod.func(...) — resolve the dotted chain
        qn = _qualname_from_attr(func)
        if qn is None:
            continue
        # Try to upgrade the leading segment via imports
        head, _, tail = qn.partition(".")
        if head in imports:
            qn_full = f"{imports[head]}.{tail}" if tail else imports[head]
        else:
            qn_full = qn
        if qn_full in PRODUCTION_SYMBOLS:
            out.append((qn_full, node.lineno))
    return out


# ---------------------------------------------------------------------------
# Inline-suspicion scan
# ---------------------------------------------------------------------------


def _suspicious_inline(source: str) -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    lines = source.splitlines()
    for pattern, desc in REIMPLEMENTATION_PATTERNS:
        rx = re.compile(pattern)
        for i, line in enumerate(lines, start=1):
            if rx.search(line):
                out.append((desc, i, line))
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def audit_script(path: Path) -> CodePathReport:
    """Run the static audit on a single script.

    Returns a structured report. The verdict combines:

    * production call count
    * suspicious-inline count
    * whether the script has any computation at all (NO-COMPUTATION
      for a pure load-and-print script).
    """
    source = path.read_text()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        # treat as REIMPLEMENTED so the analyst is forced to look
        return CodePathReport(
            path=path,
            verdict=Verdict.REIMPLEMENTED,
            production_calls=[],
            production_imports=[],
            suspicious_inline=[("could not parse — syntax error", e.lineno or 0,
                                str(e))],
        )

    imports = _walk_imports(tree)
    calls = _calls(tree, imports)
    inline = _suspicious_inline(source)

    n_prod = len(calls)
    n_inline = len(inline)

    if n_prod == 0 and n_inline == 0:
        verdict = Verdict.NO_COMPUTATION
    elif n_prod > 0 and n_inline == 0:
        verdict = Verdict.PRODUCTION_PATH
    elif n_prod == 0 and n_inline > 0:
        verdict = Verdict.REIMPLEMENTED
    else:
        verdict = Verdict.MIXED

    return CodePathReport(
        path=path,
        verdict=verdict,
        production_calls=calls,
        production_imports=sorted(set(imports.values())),
        suspicious_inline=inline,
    )


def audit_paths(paths: Iterable[Path]) -> list[CodePathReport]:
    return [audit_script(p) for p in paths]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", type=Path,
                   help="script(s) to audit")
    p.add_argument("--strict", action="store_true",
                   help="exit nonzero if any script is REIMPLEMENTED or MIXED")
    args = p.parse_args(argv)

    reports = audit_paths(args.paths)
    for r in reports:
        print(r.render())
        print()

    if args.strict:
        if any(r.verdict in (Verdict.REIMPLEMENTED, Verdict.MIXED)
               for r in reports):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
