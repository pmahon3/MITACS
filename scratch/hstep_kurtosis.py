"""Per-horizon excess kurtosis of the standardised h-step residual on
the rebaselined augmented-scope forecast cache. Answers a Stage-1
question in writeup/tex/qualifier_exhaustion_memo.tex: does the
heavy-tail problem dissolve at longer horizons (CLT-like averaging
over the composition)? If yes, Single-step is a Stage-1-responsive
lever (direct fitting of Pi_h for h>1 with a Gaussian local kernel
would be adequate at the horizon being forecast). If no, the
non-Gaussianity is a per-step property the composition does not
launder away and direct multi-step fitting cannot rescue.

Reuses the augmented-scope cache produced by run_error_decomposition;
no estimator reimplementation. The cached frame has columns
{horizon_h, z_pred, z_actual, z_err, s2_k}; per-horizon excess kurt
of (z_err / sqrt(s2_k)) is a one-liner groupby.

PROVENANCE: INSPECTION-ONLY. The kurt at h=1 is the same scalar
innovation kurt the production rebaseline characterised at
~24-33 (per memory mitacs-rebaseline-facts); larger spread here is
expected because this cache is the WLS-with-LOO-CV-bandwidth dev
forecast over the 132-day population, not the per-day-type rebaseline
slab. The QUESTION asked is the SHAPE of kurt(h) vs h, not the
absolute h=1 value.
"""
from __future__ import annotations

import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path("/tmp/bench2122_fc_propagated_augmented.pkl")


def main() -> None:
    if not CACHE.exists():
        print(f"ERROR: cache not found: {CACHE}", file=sys.stderr)
        print("Run scratch/run_error_decomposition.py first to populate.",
              file=sys.stderr)
        sys.exit(1)

    with CACHE.open("rb") as f:
        fc = pickle.load(f)

    if not isinstance(fc, pd.DataFrame):
        print(f"ERROR: cache is {type(fc).__name__}, expected DataFrame",
              file=sys.stderr)
        sys.exit(1)

    print(f"loaded cache: n_rows = {len(fc)}, "
          f"n_days = {fc['delivery_date'].nunique()}, "
          f"horizons = {sorted(fc['horizon_h'].unique())[:5]}..."
          f"{sorted(fc['horizon_h'].unique())[-3:]}")

    # Standardised h-step residual: (z_pred - z_actual) / sqrt(s2_k)
    # Sign convention follows run_error_decomposition.py line 351
    # (which uses y - m): residual = z_actual - z_pred. Take absolute
    # squared (kurt is sign-insensitive), but use consistent sign.
    z_std = (fc["z_actual"] - fc["z_pred"]) / np.sqrt(
        np.maximum(fc["s2_k"], 1e-9)
    )

    # Per-horizon excess kurtosis (Fisher; 0 = Gaussian). pandas
    # .kurtosis() returns Fisher excess. Also compute n and tail ratio
    # (|z|>2 / 0.0455 expected under Gaussian).
    by_h = pd.DataFrame({"z_std": z_std, "horizon_h": fc["horizon_h"]})
    rows = []
    for h, g in by_h.groupby("horizon_h"):
        vals = g["z_std"].to_numpy()
        vals = vals[np.isfinite(vals)]
        n = len(vals)
        if n < 30:
            continue
        kurt = float(pd.Series(vals).kurtosis())  # Fisher excess
        tail_p2 = float(np.mean(np.abs(vals) > 2.0))
        tail_ratio = tail_p2 / 0.0455
        rows.append({
            "h": int(h),
            "n": int(n),
            "excess_kurt": kurt,
            "P(|z|>2)": tail_p2,
            "tail_ratio_vs_N(0,1)": tail_ratio,
        })

    res = pd.DataFrame(rows).set_index("h")
    print()
    print("=" * 64)
    print("Per-horizon excess kurtosis of the standardised h-step "
          "residual")
    print(f"(augmented-scope, full-population, n_days = "
          f"{fc['delivery_date'].nunique()})")
    print("=" * 64)
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(res)
    print()

    # Direct comparison: does kurt(h) drop materially as h grows?
    kurt_1 = res.loc[1, "excess_kurt"]
    kurt_24 = res.loc[24, "excess_kurt"] if 24 in res.index \
        else res["excess_kurt"].iloc[-1]
    kurt_min = res["excess_kurt"].min()
    h_at_min = int(res["excess_kurt"].idxmin())
    print(f"h=1 excess kurt          : {kurt_1:.2f}")
    print(f"h=24 (or last) kurt      : {kurt_24:.2f}")
    print(f"minimum kurt across h    : {kurt_min:.2f} at h={h_at_min}")
    print(f"Gaussian reference       :   0.00")

    # Stage-1 verdict bar (pre-stated): if kurt drops below ~3 at any
    # h, direct multi-step fitting at that horizon is Stage-1-responsive
    # (kernel form becomes approximately adequate). If kurt stays
    # high across all h, Single-step is foreclosed-as-non-responsive
    # at Stage 1 and the qualifier exhausts only at Stage 2.
    if kurt_min < 3.0:
        verdict = (f"OPEN-and-RESPONSIVE: kurt drops to "
                   f"{kurt_min:.2f} at h={h_at_min}; direct multi-step "
                   f"fitting at that horizon is a Stage-1 lever")
    elif kurt_min < 8.0:
        verdict = (f"OPEN-but-WEAKLY-RESPONSIVE: kurt drops to "
                   f"{kurt_min:.2f} at h={h_at_min}; direct multi-step "
                   f"fitting helps somewhat but the kernel-form gap "
                   f"is not closed")
    else:
        verdict = (f"OPEN-but-NON-RESPONSIVE: kurt stays "
                   f">={kurt_min:.2f} across all h; composition does "
                   f"not launder the heavy tail at any horizon")
    print()
    print(f"Stage-1 Single-step verdict: {verdict}")


if __name__ == "__main__":
    main()
