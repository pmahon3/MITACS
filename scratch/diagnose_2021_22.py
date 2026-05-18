"""EXPLORATORY error diagnostic on the 2021-22 dev benchmark.

Honing step 1 (memory mitacs-honing-methodology): find WHERE/WHY our
errors concentrate so the honing target is chosen EMPIRICALLY, not
guessed. Not recorded, scratch/ only, dev-set analysis.

Caches the (expensive) forecast frame to /tmp so re-analysis is cheap.

Axes (each isolates a different honing lever):
  * hour-of-day & day-type        -- diurnal / regime structure
  * |demand ramp| (|d Actual/dt|) -- ramp vs plateau (the Step-4/2021-22
                                     signature: ramps hardest)
  * signed error vs demand level  -- bias (systematic over/under) vs noise
  * ours-error vs IESO-error      -- where we fail that IESO does NOT:
       high corr  => shared difficulty (intrinsic, not fixable by us)
       low/neg    => OUR-specific failure (fixable by honing)
       IESO good where we're bad => the exogenous-information gap

PROVENANCE-GRADE: INSPECTION-ONLY -- exploratory dev-set; MUST NOT be
cited as a result (Task 36 / PROVENANCE_REQUIREMENTS.md C7, gap #6).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from experiment.predict import _daytype

CACHE = Path("/tmp/bench2122_fc.pkl")


def _frame() -> pd.DataFrame:
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    from scratch.benchmark_2021_22_ieso import compute

    fc, _dims = compute()
    CACHE.write_bytes(pickle.dumps(fc))
    return fc


def main() -> None:
    fc = _frame().copy()
    fc = fc.dropna(subset=["actual_mw"]).sort_index()
    a = fc["actual_mw"]
    fc["err"] = fc["ours_mw"] - a            # signed (>0 = we over-forecast)
    fc["abs_err"] = fc["err"].abs()
    fc["ape"] = (fc["abs_err"] / a) * 100
    fc["hod"] = fc.index.hour
    fc["dtype"] = [_daytype(t, 7) for t in fc.index]
    fc["ramp"] = a.diff().abs()              # |hour-over-hour demand change|

    print(f"n={len(fc)}  overall MAPE {fc.ape.mean():.2f}%  "
          f"MAE {fc.abs_err.mean():.0f}  signed-bias {fc.err.mean():+.0f} MW")

    print("\n-- by day-type --")
    print(fc.groupby("dtype")["ape"].agg(["mean", "count"]).round(2).to_string())

    print("\n-- signed bias by hour-of-day (is error a consistent "
          "over/under, i.e. fixable systematic, vs zero-mean noise?) --")
    bh = fc.groupby("hod")["err"].mean().round(0)
    print(bh.to_string())
    print(f"  => |mean signed bias| avg = {bh.abs().mean():.0f} MW; "
          f"if >> per-hour noise, a bias correction is low-hanging fruit")

    print("\n-- error vs |demand ramp| quartile (ramp = hardest?) --")
    fc["rq"] = pd.qcut(fc["ramp"].fillna(0), 4, labels=["Q1lo", "Q2", "Q3", "Q4hi"])
    print(fc.groupby("rq", observed=True)["ape"].mean().round(2).to_string())

    g = fc.dropna(subset=["ieso_mw"]).copy()
    if len(g):
        g["ieso_abs"] = (g["ieso_mw"] - g["actual_mw"]).abs()
        r = np.corrcoef(g["abs_err"], g["ieso_abs"])[0, 1]
        print(f"\n-- ours-abs-err vs IESO-abs-err corr = {r:.2f} --")
        print("  high(+) => shared/intrinsic difficulty (not ours to fix);")
        print("  low/neg => OUR-specific error (honing can help)")
        # the diagnostic lever: hours where IESO is accurate but we are not
        g["gap"] = g["abs_err"] - g["ieso_abs"]
        worst = g.groupby(g.index.hour)["gap"].mean().sort_values()
        print("\n  mean (ours_abs - ieso_abs) by hod -- most negative = we're"
              " closest to IESO; most positive = our biggest exogenous gap:")
        print(worst.round(0).to_string())
        # is the gap explained by ramp (=> structural/weather) or not?
        gr = g.groupby(pd.qcut(g["ramp"].fillna(0), 4,
                               labels=["Q1lo", "Q2", "Q3", "Q4hi"]),
                       observed=True)[["abs_err", "ieso_abs"]].mean().round(0)
        print("\n  abs-err by ramp quartile, ours vs IESO "
              "(does IESO's edge widen on ramps => weather-driven?):")
        print(gr.to_string())


if __name__ == "__main__":
    main()
