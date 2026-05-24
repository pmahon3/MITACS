"""Paired-day-bootstrap hypothesis testing for the SMC backtest grid.

Reads the per-row parquet frames written by scratch.backtest_smc when
launched with --save-dir, and produces 95% CIs on every cell-to-cell
MAE difference relevant to the Liu-Gao pre-registered predictions
(memory mitacs-liu-gao-prediction):

  P1  C vs B; E vs A   (SMC-Gaussian vs mean-iter, same theta regime)
  P2  D vs B; F vs A   (SMC-empirical vs mean-iter, same theta regime)
  P3  |C - E| vs |D - F|   (theta-sensitivity differential)
  P4  coverage per cell (analytical binomial SE; no bootstrap needed)

The paired-day bootstrap: resample delivery days with replacement
(n=500); recompute each cell's MAE on the resampled day-set; take the
difference. B=1000 replicates -> 95% CI on the difference. Sign test
asks whether the CI excludes zero.

Per-horizon table: same logic, grouped by horizon_h. Produces a
24-row table per (X, Y) comparison with per-horizon dMAE ± 95% CI.

PROVENANCE-GRADE: INSPECTION-ONLY.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CELL_LABELS = {
    "A_meaniter_prod":     "A. mean-iter production",
    "B_meaniter_global":   "B. mean-iter global-OLS",
    "C_smc_global_gauss":  "C. SMC-global Gaussian",
    "D_smc_global_emp":    "D. SMC-global empirical",
    "E_smc_prod_gauss":    "E. SMC-prod Gaussian",
    "F_smc_prod_emp":      "F. SMC-prod empirical",
}

# Pairs the Liu-Gao predictions discriminate on, with the predicted sign
# of MAE(X) - MAE(Y).
LIU_GAO_PAIRS = [
    ("P1 global-θ:   C vs B", "C_smc_global_gauss",  "B_meaniter_global",
     "SMC-Gaussian should UNDERPERFORM mean-iter (predicted: +)"),
    ("P1 prod-θ:     E vs A", "E_smc_prod_gauss",    "A_meaniter_prod",
     "SMC-Gaussian should UNDERPERFORM mean-iter (predicted: +)"),
    ("P2 global-θ:   D vs B", "D_smc_global_emp",    "B_meaniter_global",
     "SMC-empirical should MATCH OR BEAT mean-iter (predicted: ≤0)"),
    ("P2 prod-θ:     F vs A", "F_smc_prod_emp",      "A_meaniter_prod",
     "SMC-empirical should MATCH OR BEAT mean-iter (predicted: ≤0)"),
]


def _mae(pred: np.ndarray, act: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - act)))


def _day_keys(df: pd.DataFrame) -> np.ndarray:
    """Per-delivery-day grouping key (int64 ns timestamps)."""
    return df["delivery_date"].astype("int64").to_numpy()


def _bootstrap_dmae(
    fcX: pd.DataFrame, fcY: pd.DataFrame, B: int, seed: int,
    horizon: int | None = None,
) -> tuple[float, float, float]:
    """Paired-day-bootstrap on MAE(X) - MAE(Y).

    Both frames share the delivery_date set (the cells were run over
    the same days). Resamples delivery days with replacement; for each
    replicate, MAE on the resampled day-set is computed as the
    sample-size-weighted average of per-day MAEs (mathematically
    equivalent to recomputing |pred - act|.mean() on the concatenated
    rows, since each day contributes 24 hourly forecasts uniformly).

    Returns (point estimate, lo, hi) at 95%.

    `horizon` restricts to one horizon_h before bootstrapping.
    """
    if horizon is not None:
        fcX = fcX[fcX["horizon_h"] == horizon]
        fcY = fcY[fcY["horizon_h"] == horizon]

    # Pre-compute per-day SUM-of-AE and COUNT for each cell on every
    # common day. Bootstrap then operates on length-N vectors instead
    # of re-concatenating row arrays inside the B-loop.
    def _per_day(fc: pd.DataFrame) -> dict[pd.Timestamp, tuple[float, int]]:
        ae = (fc["our_forecast_mw"] - fc["actual_mw"]).abs()
        agg = ae.groupby(fc["delivery_date"]).agg(["sum", "count"])
        return {d: (float(r["sum"]), int(r["count"]))
                for d, r in agg.iterrows()}

    dX, dY = _per_day(fcX), _per_day(fcY)
    common_days = np.array(sorted(set(dX) & set(dY)))
    if len(common_days) == 0:
        return float("nan"), float("nan"), float("nan")

    xs = np.array([dX[d][0] for d in common_days])  # AE-sum per day
    xn = np.array([dX[d][1] for d in common_days])  # row-count per day
    ys = np.array([dY[d][0] for d in common_days])
    yn = np.array([dY[d][1] for d in common_days])

    point = xs.sum() / xn.sum() - ys.sum() / yn.sum()

    rng = np.random.default_rng(seed)
    n = len(common_days)
    diffs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        diffs[b] = (xs[idx].sum() / xn[idx].sum()
                    - ys[idx].sum() / yn[idx].sum())

    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def _coverage_binomial(fc: pd.DataFrame) -> tuple[float, float]:
    """Analytical binomial coverage + SE for one cell."""
    inside = ((fc["actual_mw"] >= fc["our_pi_lo_mw"])
              & (fc["actual_mw"] <= fc["our_pi_hi_mw"]))
    p = float(inside.mean())
    n = len(fc)
    se = float(np.sqrt(p * (1 - p) / n))
    return 100 * p, 100 * se


def _ensure_daytype(df: pd.DataFrame) -> pd.DataFrame:
    """Derive a `daytype` column from target_dt if missing.
    The mean-iter cells (A, B) did not save daytype; the SMC cells did.
    Stratification needs it on every cell, derived consistently."""
    if "daytype" in df.columns:
        return df
    from config import load_config
    from experiment.predict import _daytype
    anchor_h = load_config().data.day_anchor_hours
    df = df.copy()
    df["daytype"] = df["target_dt"].apply(lambda t: _daytype(t, anchor_h))
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save-dir", required=True,
                    help="directory containing cell_*.parquet (the "
                         "backtest_smc --save-dir output)")
    ap.add_argument("--B", type=int, default=1000,
                    help="bootstrap replicates (default 1000)")
    ap.add_argument("--seed", type=int, default=20260523)
    ap.add_argument("--per-horizon", action="store_true",
                    help="also produce per-horizon dMAE CIs (24 rows "
                         "per comparison)")
    ap.add_argument("--by-daytype", action="store_true",
                    help="repeat coverage + bootstrap dMAE stratified "
                         "by day-type (weekday, saturday, sunday); "
                         "mean-iter cells get their daytype derived from "
                         "target_dt + cfg.data.day_anchor_hours")
    args = ap.parse_args()

    cells = {}
    for tag in CELL_LABELS:
        path = Path(args.save_dir) / f"cell_{tag}.pkl"
        if not path.exists():
            print(f"  missing: {path}")
            continue
        cells[tag] = _ensure_daytype(pd.read_pickle(path))
        print(f"  loaded {tag}: {len(cells[tag])} rows")

    print("\n" + "=" * 74)
    print("INTERVAL COVERAGE (analytical binomial SE, per cell)")
    print("=" * 74)
    print(f"  {'cell':<28} {'coverage':>10} {'SE':>8}    n")
    for tag, df in cells.items():
        cov, se = _coverage_binomial(df)
        print(f"  {CELL_LABELS[tag]:<28} {cov:>9.2f}%  ±{se:>5.2f}  "
              f"{len(df)}")

    print("\n" + "=" * 74)
    print(f"PAIRED-DAY BOOTSTRAP (B={args.B})")
    print("=" * 74)
    for label, x_tag, y_tag, hyp in LIU_GAO_PAIRS:
        if x_tag not in cells or y_tag not in cells:
            print(f"  {label}: skipped (missing parquet)")
            continue
        dmae, lo, hi = _bootstrap_dmae(
            cells[x_tag], cells[y_tag], args.B, args.seed)
        sep = "  excludes 0" if (lo > 0 or hi < 0) else "  includes 0"
        print(f"\n  {label}")
        print(f"    {hyp}")
        print(f"    MAE(X) - MAE(Y) = {dmae:>+7.2f} MW  "
              f"95% CI [{lo:>+7.2f}, {hi:>+7.2f}]{sep}")

    # P3: theta-sensitivity differential.
    # The pre-registered claim is |MAE(D)-MAE(F)| < |MAE(C)-MAE(E)|
    # -- a difference of absolute differences. Reported here as the
    # qualitative side-by-side without a formal CI on the |.|-of-|.|
    # statistic; treat the two θ-sensitivity numbers as point
    # estimates only, NOT as a tested claim.
    needed = ("C_smc_global_gauss", "E_smc_prod_gauss",
              "D_smc_global_emp",   "F_smc_prod_emp")
    if all(k in cells for k in needed):
        ce, _, _ = _bootstrap_dmae(cells["C_smc_global_gauss"],
                                    cells["E_smc_prod_gauss"],
                                    args.B, args.seed + 1)
        df, _, _ = _bootstrap_dmae(cells["D_smc_global_emp"],
                                    cells["F_smc_prod_emp"],
                                    args.B, args.seed + 2)
        print(f"\n  P3 θ-sensitivity (qualitative comparison, no formal "
              f"test):")
        print(f"    Gaussian-SMC  MAE(C) - MAE(E) = {ce:>+6.2f} MW")
        print(f"    empirical-SMC MAE(D) - MAE(F) = {df:>+6.2f} MW")
        print(f"    Prediction is |empirical-SMC| < |Gaussian-SMC|;")
        print(f"    no CI on |.|-of-|.|. Read as point estimates only.")

    if args.per_horizon:
        print("\n" + "=" * 74)
        print(f"PER-HORIZON dMAE (B={args.B}, 95% CI)")
        print("=" * 74)
        for label, x_tag, y_tag, _ in LIU_GAO_PAIRS:
            if x_tag not in cells or y_tag not in cells:
                continue
            print(f"\n  {label}")
            print(f"    {'h':>3} {'dMAE':>+9} {'95% CI':>22}")
            for h in range(1, 25):
                d, lo, hi = _bootstrap_dmae(
                    cells[x_tag], cells[y_tag], args.B, args.seed,
                    horizon=h)
                sep = "  *" if (lo > 0 or hi < 0) else ""
                print(f"    {h:>3} {d:>+9.1f}  "
                      f"[{lo:>+8.1f},{hi:>+8.1f}]{sep}")

    if args.by_daytype:
        print("\n" + "=" * 74)
        print("STRATIFIED BY DAY-TYPE")
        print("=" * 74)
        print("Note: weekday d=2, saturday d=2, sunday d=4. CIs widen on "
              "the smaller populations\n(weekday ~71% of days, sat/sun "
              "~14% each); read sat/sun signs as ~2x noisier than weekday.")
        for dt in ("weekday", "saturday", "sunday"):
            cells_dt = {tag: df[df["daytype"] == dt]
                        for tag, df in cells.items()}
            sample = next(iter(cells_dt.values()))
            n_typical = sample.shape[0]
            n_days = int(sample["delivery_date"].nunique())
            print(f"\n  --- {dt} (n_days={n_days}, n_rows={n_typical} "
                  f"per cell) ---")
            print(f"  {'cell':<28} {'coverage':>10} {'SE':>8}")
            for tag, df in cells_dt.items():
                if len(df) == 0:
                    continue
                cov, se = _coverage_binomial(df)
                print(f"  {CELL_LABELS[tag]:<28} {cov:>9.2f}%  "
                      f"±{se:>5.2f}")
            print()
            for label, x_tag, y_tag, _hyp in LIU_GAO_PAIRS:
                if x_tag not in cells_dt or y_tag not in cells_dt:
                    continue
                if len(cells_dt[x_tag]) == 0 or len(cells_dt[y_tag]) == 0:
                    continue
                d, lo, hi = _bootstrap_dmae(
                    cells_dt[x_tag], cells_dt[y_tag], args.B, args.seed)
                sep = "  excludes 0" if (lo > 0 or hi < 0) else "  includes 0"
                print(f"    {label:<24} dMAE = {d:>+7.2f}  "
                      f"95% CI [{lo:>+7.2f}, {hi:>+7.2f}]{sep}")

    print("=" * 74)


if __name__ == "__main__":
    main()
