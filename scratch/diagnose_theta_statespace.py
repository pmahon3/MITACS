"""EXPLORATORY: does the per-anchor theta vary by state space, and do
high-error regions coincide with mis-scaled theta?  (Honing diagnostic,
NOT recorded; dev set 2021-22; scratch only.)

Decides whether state-space-adaptive localisation is a REAL lever:
  Q1  Does theta* actually vary across state space (z-demand level)?
      If ~constant -> no lever, adaptivity only adds overfitting risk.
  Q2  Does theta* already track LOCAL DENSITY (knn50 distance)?
      If theta* strongly tracks density already -> CV is implicitly
      doing the adaptive thing; little to gain.  If NOT -> the explicit
      lever (kNN-scaled / region-CV theta) could help.
  Q3  Do HIGH-ERROR rows have systematically different theta vs
      low-error rows in the SAME region?  If high error <-> mis-scaled
      theta -> direct evidence the lever bites.
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
        f = pickle.loads(CACHE.read_bytes())
        if "theta" in f.columns:
            return f
    from scratch.benchmark_2021_22_ieso import compute

    fc, _dims = compute()
    CACHE.write_bytes(pickle.dumps(fc))
    return fc


def main() -> None:
    fc = _frame().copy().dropna(subset=["actual_mw"]).sort_index()
    a = fc["actual_mw"]
    fc["abs_err"] = (fc["ours_mw"] - a).abs()
    fc["ape"] = fc["abs_err"] / a * 100
    fc["hod"] = fc.index.hour

    th = fc["theta"]
    print(f"n={len(fc)}")
    print(f"\nQ1  theta* spread: min={th.min():.3f} p25={th.quantile(.25):.3f} "
          f"median={th.median():.3f} p75={th.quantile(.75):.3f} "
          f"max={th.max():.3f}  uniq={th.nunique()}")
    print(f"    coeff of variation = {th.std()/th.mean():.2f}  "
          f"(near 0 => theta ~constant => NO state-space lever)")

    print("\n    theta median by z-demand-level quintile (state space):")
    fc["zq"] = pd.qcut(fc["q_z0"], 5, labels=["z1lo", "z2", "z3", "z4", "z5hi"])
    print(fc.groupby("zq", observed=True)["theta"]
          .agg(["median", "std", "count"]).round(3).to_string())

    print("\nQ2  theta vs local density (knn50 dist) -- corr "
          f"= {np.corrcoef(fc['theta'], fc['knn50_dist'])[0,1]:+.2f}")
    print("    (strong +corr => CV already adapts theta to density;")
    print("     ~0 => theta NOT tracking density => explicit lever open)")
    print("    theta median by knn50-distance quintile:")
    fc["dq"] = pd.qcut(fc["knn50_dist"], 5,
                       labels=["dense1", "d2", "d3", "d4", "sparse5"])
    print(fc.groupby("dq", observed=True)["theta"]
          .agg(["median", "count"]).round(3).to_string())

    print("\nQ3  do high-error rows have mis-scaled theta vs low-error, "
          "WITHIN the same density bin?")
    print("    (ape, theta) medians by density-bin x error-half:")
    fc["ehalf"] = np.where(
        fc.groupby("dq", observed=True)["ape"].transform(
            lambda s: s > s.median()
        ),
        "hi_err", "lo_err",
    )
    piv = (
        fc.groupby(["dq", "ehalf"], observed=True)
        .agg(ape=("ape", "median"), theta=("theta", "median"),
             n=("ape", "size"))
        .round(3)
    )
    print(piv.to_string())
    print("\n  READ: if within each density bin the hi_err half has a "
          "systematically different theta than the lo_err half, theta is "
          "mis-scaled where it matters -> state-space-adaptive theta is a "
          "real lever. If theta is ~equal across error-halves, the error "
          "is NOT a theta-scaling problem (look elsewhere).")


if __name__ == "__main__":
    main()
