"""P1 — Patra–Sen per-stratum localization.

Registered phase_a:
    notes/preregistrations/2026-05-27_p1-patra-sen-per-stratum-localization/

Operates on ``Q1.pit_M1``'s ``u_PIT`` column (n=12000 post-cutoff rows;
columns ``[delivery_day, h, day_type, u_PIT, nu_used]``). For each of
five stratification axes, computes per-stratum ``α̂_L`` via the
production ``patra_sen_fit(F_b='uniform')`` and the range statistic
``range_axis = max(α̂_L) − min(α̂_L)`` across that axis's strata. 1000
paired-day bootstrap resamples give 95% CIs on every α̂_L and every
range. The verdict statistic is ``max_axis range_axis`` against the
phase_a-registered cuts: R-A if > 0.20, R-B in (0.10, 0.20], R-C
if < 0.10.

Why F_b=Uniform (and not the literal phase_a's F_b=N(0,1)): see the
methodology amendment in commit 99a8122. Patra & Sen (2016) Theorem 1
gives the equivalence; this is the correct null for a calibrated PIT.

Run::

    python -m experiment.p1_patra_sen --quick           # no bootstrap; ~1s
    python -m experiment.p1_patra_sen --n-bootstrap 1000 --emit-result

The ``--emit-result`` flag stamps a CLAIM-GRADE provenanced result
artifact via ``experiment.provenance.make_result`` (refuses on a
dirty tree; predictor-derived ⇒ ``frozen_spec_required=True``).
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Production-path imports (whitelisted in experiment/audit/code_path.py).
from experiment._actuals import load_actuals
from processing.innovations.estimator import patra_sen_fit

# ──────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────
# Sample-size floor for patra_sen_fit (raises below 20). Strata with
# fewer than this many post-cutoff rows are excluded from the range
# computation rather than crashing the run; this happens occasionally
# during a bootstrap resample when an unlucky day mix evacuates a small
# stratum (e.g. the smallest hour_of_week bucket on a sparse resample).
_MIN_STRATUM_N = 20

# Verdict cuts (phase_a §shape_outcomes).
_R_A_CUT = 0.20
_R_C_CUT = 0.10

# Bootstrap default per phase_a §ci_method.details.
_N_BOOTSTRAP = 1000

# Paths under PROJECT_ROOT (config layer is the path source of truth).
_Q1_PICKLE_REL = Path("scratch/data/distributional_class_q1/q1.pkl")


# ──────────────────────────────────────────────────────────────────────────
# Stratification axes (phase_a §metric, locked)
# ──────────────────────────────────────────────────────────────────────────
def _season_label(month: int) -> str:
    """Winter / summer / shoulder per phase_a §metric.

    Winter: Nov–Mar  Summer: Jun–Aug  Shoulder: Apr–May + Sep–Oct.
    """
    if month in (11, 12, 1, 2, 3):
        return "winter"
    if month in (6, 7, 8):
        return "summer"
    return "shoulder"


def _time_of_day_label(h: int) -> str:
    """Phase_a's 4 bins: overnight 00-05 / morning 06-10 /
    afternoon 11-16 / evening 17-23. ``h`` is 1..24 from Q1's pit_M1
    (1-indexed horizon-hour; the actual delivery-clock hour is
    h - 1 modulo 24 for h ≤ 24, but Q1's convention is that h IS the
    delivery-clock hour for one-day-ahead forecasts on a single
    delivery_day).
    """
    h0 = h - 1  # 0..23 clock hour
    if h0 <= 5:
        return "overnight"
    if h0 <= 10:
        return "morning_ramp"
    if h0 <= 16:
        return "afternoon"
    return "evening"


def _hour_of_week_bin(weekday: int, h: int) -> int:
    """8 bins of 21 hours each. ``weekday`` is 0=Mon..6=Sun;
    ``h`` is 1..24. Total 168 hours of the week collapsed into
    8 bins of 21.
    """
    hour_of_week = weekday * 24 + (h - 1)
    return hour_of_week // 21  # 0..7


def _demand_tercile_labels(demand_per_row: np.ndarray) -> np.ndarray:
    """Cut a (n,) demand array into 3 quantile labels: low/medium/high."""
    q1, q2 = np.quantile(demand_per_row, [1 / 3, 2 / 3])
    labels = np.full(demand_per_row.shape, "medium", dtype=object)
    labels[demand_per_row <= q1] = "low"
    labels[demand_per_row >= q2] = "high"
    return labels


# ──────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────
def load_q1_pit_M1(pickle_path: Path) -> pd.DataFrame:
    """Load Q1's post-cutoff PIT-on-M1 dataframe.

    Returns the ``pit_M1`` DataFrame with columns
    ``[delivery_day, h, day_type, u_PIT, nu_used]`` (n=12000 in the
    canonical run).
    """
    with open(pickle_path, "rb") as f:
        d = pickle.load(f)
    pit_M1 = d.get("pit_M1")
    if pit_M1 is None or not isinstance(pit_M1, pd.DataFrame):
        raise ValueError(
            f"{pickle_path} does not contain a 'pit_M1' DataFrame; "
            "Q1's run output appears to be malformed. Phase_a's "
            "§data.source requires this pickle to carry pit_M1."
        )
    needed = {"delivery_day", "h", "day_type", "u_PIT", "nu_used"}
    missing = needed - set(pit_M1.columns)
    if missing:
        raise ValueError(
            f"{pickle_path}'s pit_M1 missing columns: {sorted(missing)}"
        )
    # Defensive copy + ensure delivery_day is dtype datetime64.
    out = pit_M1.copy()
    out["delivery_day"] = pd.to_datetime(out["delivery_day"])
    return out


def attach_stratum_labels(
    pit: pd.DataFrame, *, demand_series: pd.Series | None
) -> pd.DataFrame:
    """Add 5 stratum-label columns to pit_M1 in place.

    Columns added:
      ``stratum_demand_quantile`` (low/medium/high; needs hourly demand)
      ``stratum_time_of_day``     (overnight/morning_ramp/afternoon/evening)
      ``stratum_season``          (winter/summer/shoulder)
      ``stratum_day_type``        (already in pit_M1.day_type)
      ``stratum_hour_of_week``    (0..7)
    """
    out = pit.copy()
    out["stratum_time_of_day"] = out["h"].apply(_time_of_day_label)
    out["stratum_season"] = out["delivery_day"].dt.month.apply(_season_label)
    out["stratum_day_type"] = out["day_type"]
    weekday = out["delivery_day"].dt.dayofweek  # 0=Mon..6=Sun
    out["stratum_hour_of_week"] = [
        _hour_of_week_bin(int(wd), int(h)) for wd, h in zip(weekday, out["h"])
    ]
    if demand_series is not None:
        # The pit row is a (delivery_day, h) pair; the load_actuals
        # series is indexed by hourly Timestamp. Map each row to its
        # delivery-clock timestamp and look up the demand. h=1 means
        # the 00:00 hour of that delivery_day; h=24 means the 23:00 hour.
        ts = out["delivery_day"] + pd.to_timedelta(out["h"] - 1, unit="h")
        # Reindex demand_series; missing entries become NaN (rare; would
        # indicate a gap in the actuals feed). NaN demand → median
        # stratum so the row stays in play without biasing the cut.
        demand_per_row = demand_series.reindex(ts).to_numpy()
        # Substitute median for NaN before terciling (very few entries).
        if np.any(np.isnan(demand_per_row)):
            med = float(np.nanmedian(demand_per_row))
            demand_per_row = np.where(
                np.isnan(demand_per_row), med, demand_per_row
            )
        out["stratum_demand_quantile"] = _demand_tercile_labels(demand_per_row)
    else:
        out["stratum_demand_quantile"] = pd.NA
    return out


# ──────────────────────────────────────────────────────────────────────────
# Per-stratum α̂_L
# ──────────────────────────────────────────────────────────────────────────
_AXES = (
    "stratum_demand_quantile",
    "stratum_time_of_day",
    "stratum_season",
    "stratum_day_type",
    "stratum_hour_of_week",
)


def alpha_L_per_axis(u: np.ndarray, labels: np.ndarray) -> dict[Any, float]:
    """Per-stratum α̂_L for one axis. NaN for strata with n < _MIN_STRATUM_N.

    ``u``      : (n,) PIT values in [0, 1]
    ``labels`` : (n,) stratum labels (same length as u)
    Returns ``{stratum_label: alpha_L}``.
    """
    out: dict[Any, float] = {}
    # iterate in label-sorted order so the report is deterministic
    for s in sorted(set(labels), key=lambda x: (str(type(x).__name__), x)):
        mask = labels == s
        u_s = u[mask]
        if u_s.size < _MIN_STRATUM_N:
            out[s] = float("nan")
            continue
        fit = patra_sen_fit(u_s, F_b="uniform")
        out[s] = float(fit["alpha_L"])
    return out


def range_alpha_L(per_stratum: dict[Any, float]) -> float:
    """range = max(α̂_L) − min(α̂_L) ignoring NaN strata. NaN if fewer than
    two non-NaN strata in the axis."""
    finite = [v for v in per_stratum.values() if np.isfinite(v)]
    if len(finite) < 2:
        return float("nan")
    return float(max(finite) - min(finite))


# ──────────────────────────────────────────────────────────────────────────
# Paired-day bootstrap
# ──────────────────────────────────────────────────────────────────────────
def paired_day_bootstrap(
    pit: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, dict[Any, list[float]]], dict[str, list[float]]]:
    """Resample at the delivery_day level; recompute α̂_L per stratum
    and range_axis per axis on each resample.

    Returns ``(per_stratum_history, range_history)`` where
      ``per_stratum_history[axis][stratum]`` is a list of length
      n_bootstrap of α̂_L values (some NaN if the resample evacuated
      that stratum);
      ``range_history[axis]`` is a list of length n_bootstrap of
      range values.
    """
    rng = np.random.default_rng(seed)
    delivery_days = pit["delivery_day"].drop_duplicates().to_numpy()
    n_days = len(delivery_days)

    # Pre-index rows by delivery_day so resampling is fast.
    day_to_rows: dict[Any, np.ndarray] = {}
    for d, idx in pit.groupby("delivery_day", sort=False).indices.items():
        day_to_rows[d] = np.asarray(idx, dtype=np.int64)

    # Pre-pull arrays as numpy for speed (avoid pandas overhead in the loop).
    u_arr = pit["u_PIT"].to_numpy()
    axis_labels = {axis: pit[axis].to_numpy() for axis in _AXES}

    per_stratum_history: dict[str, dict[Any, list[float]]] = {
        axis: {} for axis in _AXES
    }
    range_history: dict[str, list[float]] = {axis: [] for axis in _AXES}

    for b in range(n_bootstrap):
        # Resample delivery_days with replacement; concatenate row indices.
        sampled_days = rng.choice(delivery_days, size=n_days, replace=True)
        # Build the index array by concatenating each day's rows.
        idx_parts = [day_to_rows[d] for d in sampled_days]
        idx = np.concatenate(idx_parts)
        u_b = u_arr[idx]
        for axis in _AXES:
            lbl_b = axis_labels[axis][idx]
            per_stratum_b = alpha_L_per_axis(u_b, lbl_b)
            range_history[axis].append(range_alpha_L(per_stratum_b))
            # Accumulate per-stratum history; some strata appear only
            # in some resamples so we use a defaultdict-like pattern.
            for s, v in per_stratum_b.items():
                per_stratum_history[axis].setdefault(s, []).append(v)
    return per_stratum_history, range_history


def percentile_ci(
    values: list[float], lo: float = 2.5, hi: float = 97.5
) -> tuple[float, float]:
    """95% percentile CI over the bootstrap distribution; NaN-safe."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    return (
        float(np.percentile(arr, lo)),
        float(np.percentile(arr, hi)),
    )


# ──────────────────────────────────────────────────────────────────────────
# Verdict
# ──────────────────────────────────────────────────────────────────────────
def verdict_of(max_range: float) -> str:
    """R-A / R-B / R-C from the verdict statistic max_axis(range_axis)."""
    if not np.isfinite(max_range):
        return "INDETERMINATE"
    if max_range > _R_A_CUT:
        return "R-A"
    if max_range > _R_C_CUT:
        return "R-B"
    return "R-C"


# ──────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────
def render_report(
    *,
    n_rows: int,
    n_days: int,
    alpha_L_marginal: float,
    per_axis_point: dict[str, dict[Any, float]],
    per_axis_range_point: dict[str, float],
    per_axis_range_ci: dict[str, tuple[float, float]] | None,
    binding_axis: str,
    verdict: str,
    n_bootstrap: int,
) -> str:
    lines: list[str] = []
    p = lines.append
    p("P1 — Patra–Sen per-stratum localization")
    p("=" * 60)
    p(f"data:        Q1's pit_M1.u_PIT  (n={n_rows} rows, {n_days} delivery days)")
    p(f"F_b:         Uniform[0,1]  (P1 methodology amendment; Theorem 1)")
    p(f"bootstrap:   n_bootstrap = {n_bootstrap}")
    p("")
    p(f"marginal α̂_L (full pool, INSPECTION-ONLY per phase_a §baselines):")
    p(f"  α̂_L = {alpha_L_marginal:.4f}")
    p("")
    p("per-axis range and per-stratum α̂_L:")
    for axis in _AXES:
        rng_pt = per_axis_range_point.get(axis, float("nan"))
        rng_ci = (
            per_axis_range_ci.get(axis) if per_axis_range_ci is not None else None
        )
        ci_str = (
            f"  [{rng_ci[0]:.4f}, {rng_ci[1]:.4f}]"
            if rng_ci is not None
            else ""
        )
        marker = "  ← binding" if axis == binding_axis else ""
        p(f"  {axis}: range = {rng_pt:.4f}{ci_str}{marker}")
        strata = per_axis_point.get(axis, {})
        for s in sorted(strata, key=lambda x: (-strata[x] if np.isfinite(strata[x]) else 0)):
            v = strata[s]
            p(f"      α̂_L[{s!r:>12}] = {v:.4f}")
    p("")
    max_rng = per_axis_range_point.get(binding_axis, float("nan"))
    p(f"verdict statistic: max_axis range = {max_rng:.4f}  (binding axis: {binding_axis})")
    p(f"verdict: {verdict}  "
      f"(R-A cut > {_R_A_CUT}; R-B in ({_R_C_CUT}, {_R_A_CUT}]; R-C < {_R_C_CUT})")
    return "\n".join(lines) + "\n"


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-bootstrap", type=int, default=_N_BOOTSTRAP,
                    help="paired-day bootstrap reps (default 1000, per phase_a)")
    ap.add_argument("--quick", action="store_true",
                    help="skip bootstrap; emit point estimates only (smoke test)")
    ap.add_argument("--seed", type=int, default=20260527,
                    help="paired-day bootstrap seed")
    ap.add_argument(
        "--emit-result", action="store_true",
        help="write the provenanced CLAIM-grade artifact via "
        "experiment.provenance.make_result (refuses a dirty tree); "
        "predictor-derived ⇒ frozen_spec_required=True",
    )
    args = ap.parse_args(argv)

    from config import PROJECT_ROOT
    q1_pickle = PROJECT_ROOT / _Q1_PICKLE_REL

    print(f"# P1 — Patra–Sen per-stratum localization")

    # Pre-run synthetic gate (phase_a §baselines.secondary "in-script
    # self-test"). The gate is fast (~3s wall-clock); running it inline
    # before every CLAIM-GRADE invocation removes the upstream-commit
    # coupling and gives in-band correctness on production patra_sen_fit
    # at both F_b='gaussian' and F_b='uniform' cells. BLOCKING on FAIL.
    print(f"# pre-run synthetic recovery gate (in-script self-test):")
    from processing.innovations.validation.synthetic import (
        patra_sen_recovery,
        _PATRA_SEN_NULL_ZERO_FRAC,
        _PATRA_SEN_NULL_P95_MAX,
        _PATRA_SEN_HALF_MED_LO,
        _PATRA_SEN_HALF_MED_HI,
        _PATRA_SEN_HALF_U_MED_LO,
        _PATRA_SEN_HALF_U_MED_HI,
    )
    t0 = time.time()
    ps = patra_sen_recovery()
    gate_ok = (
        ps["alpha_L_null_zero_frac"] >= _PATRA_SEN_NULL_ZERO_FRAC
        and ps["alpha_L_null_p95"] <= _PATRA_SEN_NULL_P95_MAX
        and _PATRA_SEN_HALF_MED_LO <= ps["alpha_L_half_p50"] <= _PATRA_SEN_HALF_MED_HI
        and ps["alpha_L_u_null_zero_frac"] >= _PATRA_SEN_NULL_ZERO_FRAC
        and ps["alpha_L_u_null_p95"] <= _PATRA_SEN_NULL_P95_MAX
        and _PATRA_SEN_HALF_U_MED_LO <= ps["alpha_L_u_half_p50"] <= _PATRA_SEN_HALF_U_MED_HI
    )
    print(f"#   gauss-A α̂_L p50={ps['alpha_L_null_p50']:.4f}  "
          f"gauss-B p50={ps['alpha_L_half_p50']:.4f}  "
          f"unif-A p50={ps['alpha_L_u_null_p50']:.4f}  "
          f"unif-B p50={ps['alpha_L_u_half_p50']:.4f}  "
          f"({time.time() - t0:.1f}s)")
    if not gate_ok:
        print(
            "#   GATE FAIL — refusing to run P1. Investigate the "
            "production patra_sen_fit before proceeding.",
            file=sys.stderr,
        )
        return 2
    print(f"#   gate PASS")

    print(f"# loading Q1's pit_M1 from {q1_pickle}")
    pit = load_q1_pit_M1(q1_pickle)
    print(f"#   n_rows = {len(pit)}, n_days = {pit['delivery_day'].nunique()}")

    print(f"# loading hourly demand for demand_quantile stratification")
    demand_full = load_actuals(cutoff=None)
    pit = attach_stratum_labels(pit, demand_series=demand_full)

    print(f"# computing point estimates (no bootstrap)")
    t0 = time.time()
    u_arr = pit["u_PIT"].to_numpy()
    # Marginal α̂_L (full pool) — INSPECTION-ONLY per phase_a §baselines.primary.
    alpha_L_marginal = float(
        patra_sen_fit(u_arr, F_b="uniform")["alpha_L"]
    )
    per_axis_point: dict[str, dict[Any, float]] = {}
    per_axis_range_point: dict[str, float] = {}
    for axis in _AXES:
        per = alpha_L_per_axis(u_arr, pit[axis].to_numpy())
        per_axis_point[axis] = per
        per_axis_range_point[axis] = range_alpha_L(per)
    point_time = time.time() - t0
    print(f"#   point estimates done in {point_time:.2f}s")

    # Binding axis = argmax_axis range_axis.
    finite_ranges = {
        a: r for a, r in per_axis_range_point.items() if np.isfinite(r)
    }
    binding_axis = (
        max(finite_ranges, key=finite_ranges.get)
        if finite_ranges
        else _AXES[0]
    )
    max_range = finite_ranges.get(binding_axis, float("nan"))
    verdict = verdict_of(max_range)

    # Bootstrap (optional).
    per_axis_range_ci: dict[str, tuple[float, float]] | None = None
    per_stratum_ci: dict[str, dict[Any, tuple[float, float]]] | None = None
    if not args.quick:
        print(f"# paired-day bootstrap (n={args.n_bootstrap})")
        t0 = time.time()
        ps_hist, rng_hist = paired_day_bootstrap(
            pit, n_bootstrap=args.n_bootstrap, seed=args.seed
        )
        boot_time = time.time() - t0
        print(f"#   bootstrap done in {boot_time:.1f}s")
        per_axis_range_ci = {
            a: percentile_ci(rng_hist[a]) for a in _AXES
        }
        per_stratum_ci = {
            a: {s: percentile_ci(vs) for s, vs in ps_hist[a].items()}
            for a in _AXES
        }
    else:
        print(f"# bootstrap SKIPPED (--quick)")

    report = render_report(
        n_rows=len(pit),
        n_days=int(pit["delivery_day"].nunique()),
        alpha_L_marginal=alpha_L_marginal,
        per_axis_point=per_axis_point,
        per_axis_range_point=per_axis_range_point,
        per_axis_range_ci=per_axis_range_ci,
        binding_axis=binding_axis,
        verdict=verdict,
        n_bootstrap=0 if args.quick else args.n_bootstrap,
    )
    print(report)

    if args.emit_result:
        from experiment.provenance import Grade, make_result
        out_path = (
            PROJECT_ROOT / "experiment" / "results" / "p1_patra_sen.txt"
        )
        # phase_a-registered dependent variables go into ``inputs`` so they
        # are part of the fingerprint and the body verbatim. Use plain
        # types (dicts, lists, floats); tuples become lists.
        inputs: dict[str, Any] = {
            "alpha_L_marginal": alpha_L_marginal,
            "per_axis_range_point": {
                a: per_axis_range_point[a] for a in _AXES
            },
            "binding_axis": binding_axis,
            "max_axis_range": max_range,
            "verdict": verdict,
            "n_bootstrap": args.n_bootstrap if not args.quick else 0,
            "n_rows": len(pit),
            "n_days": int(pit["delivery_day"].nunique()),
        }
        if per_axis_range_ci is not None:
            inputs["per_axis_range_ci"] = {
                a: list(per_axis_range_ci[a]) for a in _AXES
            }
        # Per-stratum α̂_L and CIs (binding-axis ranking is recoverable).
        inputs["per_axis_point"] = {
            a: {str(s): float(v) for s, v in per_axis_point[a].items()}
            for a in _AXES
        }
        if per_stratum_ci is not None:
            inputs["per_stratum_ci"] = {
                a: {
                    str(s): list(per_stratum_ci[a][s])
                    for s in per_stratum_ci[a]
                }
                for a in _AXES
            }
        hdr = make_result(
            path=out_path,
            grade=Grade.CLAIM,
            title="P1 — Patra–Sen per-stratum localization "
                  "(resolution-paths thread / production-path)",
            body=report,
            inputs=inputs,
            seeds={"paired_day_bootstrap": args.seed},
            frozen_spec_required=True,  # predictor-derived CLAIM
        )
        print(f"\nwrote provenanced CLAIM artifact -> {out_path}")
        print(f"  inputs_fingerprint = {hdr['inputs_fingerprint'][:16]}…")
        print(f"  body_sha256        = {hdr['body_sha256'][:16]}…")

    return 0


if __name__ == "__main__":
    sys.exit(main())
