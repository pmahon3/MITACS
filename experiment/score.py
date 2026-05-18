"""Scoring over the settled experiment ledger.

Honest-by-construction. Every label here encodes a decision from memory
``mitacs-scoring-design`` so the framing cannot quietly drift:

  * The headline is our ONE-STEP skill vs naive persistence -- and it is
    explicitly INTERIM scaffolding, not the Goal 3 head-to-head.
  * Persistence baselines: t-168h (same hour 7d back) is day-type-CLEAN
    and primary; t-24h is kept for interpretability but its day-type
    mismatch rate (the 07:00-anchor crosses day-type at weekends) is
    measured and reported, never hidden.
  * IESO error is reported as a DIFFERENT-HORIZON reference only. A
    1h-ahead vs ~24h-ahead comparison is trivially won by the shorter
    horizon; it is NOT a finding and is labelled as such.
  * Interval coverage is MEASURED and reported as miscalibrated -- the
    Gaussian Sigma band is a known proxy of a verified heavy-tailed
    innovation ([[mitacs-rebaseline-facts]]).
  * was_real_time subset is reported SEPARATELY; real-time and
    leak-free-nowcast forecasts are never blended into one number.
  * Zero settled rows -> a well-defined "insufficient data" result, not
    a crash or a misleading 0-error.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .ledger import read_ledger


def _err(pred: pd.Series, act: pd.Series) -> dict:
    e = (pred - act).dropna()
    if e.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "mape": np.nan}
    a = act.reindex(e.index)
    return {
        "n": int(len(e)),
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt((e**2).mean())),
        "mape": float((e.abs() / a.abs()).replace(np.inf, np.nan).mean() * 100),
    }


@dataclass
class ScoreReport:
    n_ledger: int
    n_settled: int
    sufficient: bool
    note: str
    ours: dict = field(default_factory=dict)
    persistence_168h: dict = field(default_factory=dict)
    persistence_24h: dict = field(default_factory=dict)
    persistence_24h_daytype_mismatch_pct: float = float("nan")
    ieso_DIFFERENT_HORIZON_reference: dict = field(default_factory=dict)
    interval_coverage_1sigma_pct: float = float("nan")
    interval_caveat: str = (
        "Gaussian Sigma band; innovation is verified strongly heavy-tailed "
        "-> coverage is expected BELOW nominal ~68%. Reported, not hidden."
    )
    real_time_subset: dict = field(default_factory=dict)
    framing: str = (
        "INTERIM one-step scaffolding. NOT the Goal 3 head-to-head: our "
        "estimator is one-step, IESO is day-ahead. The valid IESO "
        "comparison requires the multi-step predictor (Task 23)."
    )


def _persistence(actual_by_dt: pd.Series, lag_h: int) -> pd.Series:
    """Naive: actual at target - lag hours, aligned to graded targets."""
    shifted = actual_by_dt.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=lag_h)
    return shifted


def _daytype(ts: pd.Timestamp, anchor_h: int) -> str:
    s = ts - pd.Timedelta(hours=anchor_h)
    return "weekday" if s.dayofweek < 5 else (
        "saturday" if s.dayofweek == 5 else "sunday"
    )


def score(anchor_hours: int = 7) -> ScoreReport:
    led = read_ledger()
    graded = led[led["actual_mw"].notna()].copy()
    n_settled = len(graded)

    if n_settled == 0:
        return ScoreReport(
            n_ledger=len(led), n_settled=0, sufficient=False,
            note=(
                "No settled rows yet. Local actuals are stale vs the "
                "forecast targets; real settlement accrues over calendar "
                "time via `python -m experiment.collect` (live). Mechanics "
                "are exercised by experiment/test_settlement.py."
            ),
        )

    graded["target_dt"] = pd.to_datetime(graded["target_dt"])
    g = graded.set_index("target_dt").sort_index()
    act = g["actual_mw"]

    # naive baselines: actual value lag-h before each graded target
    full_act = act  # only graded targets available as truth here
    p168 = full_act.reindex(act.index - pd.Timedelta(hours=168))
    p168.index = act.index
    p24 = full_act.reindex(act.index - pd.Timedelta(hours=24))
    p24.index = act.index

    # t-24h day-type mismatch (07:00 anchor crosses day-type at weekends)
    mism = sum(
        _daytype(t, anchor_hours) != _daytype(t - pd.Timedelta(hours=24),
                                               anchor_hours)
        for t in act.index
    )
    mism_pct = 100.0 * mism / len(act)

    cov = np.nan
    if {"our_pi_lo_mw", "our_pi_hi_mw"}.issubset(g.columns):
        inside = ((act >= g["our_pi_lo_mw"]) & (act <= g["our_pi_hi_mw"]))
        cov = float(inside.mean() * 100)

    rt = g[g["was_real_time"].astype(str).str.lower() == "true"]
    rt_block = (
        _err(rt["our_forecast_mw"], rt["actual_mw"]) if len(rt) else
        {"n": 0, "note": "no strictly real-time (issue_time<target) rows yet"}
    )

    return ScoreReport(
        n_ledger=len(led), n_settled=n_settled, sufficient=True,
        note=f"{n_settled} graded rows.",
        ours=_err(g["our_forecast_mw"], act),
        persistence_168h=_err(p168, act),
        persistence_24h=_err(p24, act),
        persistence_24h_daytype_mismatch_pct=round(mism_pct, 1),
        ieso_DIFFERENT_HORIZON_reference=_err(g["ieso_forecast_mw"], act),
        interval_coverage_1sigma_pct=cov,
        real_time_subset=rt_block,
    )


if __name__ == "__main__":
    import pprint

    r = score()
    pprint.pp(r.__dict__, sort_dicts=False, width=88)
