"""EXPLORATORY: rescore predictor 2 across a sweep of climatology windows.

The W=2y refit (scripts/fir/rescore_climatology.py) cut MAE by ~9 %
and largely fixed the mid-day forecast-high bias, but introduced a
late-night under-forecast (~-500 MW at HE24).  This sweeps W in
{1, 2, 3, 5, 10} years to see whether a different window length
balances the mid-day and late-night errors better.

Builds on scripts.fir.rescore_climatology.run_backtest: same code
path, same predictor 2 (mean-iter, theta=0), same 500-day population,
just different climatology windows.  Outputs:

  scratch/data/rescore_window_sweep/cell_B_W{N}y.pkl     (per window)
  scratch/data/rescore_window_sweep/headline.csv         (per-W MAE, MAPE, cov)
  scratch/data/rescore_window_sweep/per_hour.csv         (W x HE signed/abs error)

Printed: a per-W summary + a per-W per-HE table for the key hours.

PROVENANCE-GRADE: INSPECTION-ONLY.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config
from experiment import freeze
from experiment._actuals import load_actuals
from experiment.backtest import _complete_delivery_days

from scripts.fir.rescore_climatology import (
    _fit_zscore_params_window,
    run_backtest,
)


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "scratch" / "data" / "rescore_window_sweep"
OUT.mkdir(parents=True, exist_ok=True)

DEFAULT_WINDOWS = (1.0, 2.0, 3.0, 5.0, 10.0)


def _summarise(df: pd.DataFrame, W: float) -> dict:
    err = (df["our_forecast_mw"] - df["actual_mw"]).abs()
    pct = err / df["actual_mw"]
    cov = ((df["our_pi_lo_mw"] <= df["actual_mw"])
           & (df["actual_mw"] <= df["our_pi_hi_mw"])).mean()
    signed = df["our_forecast_mw"] - df["actual_mw"]
    return {
        "W_years":      W,
        "MAE_mw":       float(err.mean()),
        "MAPE_pct":     float(100 * pct.mean()),
        "coverage_pct": float(100 * cov),
        "mean_signed":  float(signed.mean()),
        "rms_signed":   float(np.sqrt((signed ** 2).mean())),
        "n_rows":       int(len(df)),
    }


def _by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Per-HE (HE1..HE24) signed and absolute error."""
    work = df.assign(
        hour=df["horizon_h"],
        signed=df["our_forecast_mw"] - df["actual_mw"],
        absol=(df["our_forecast_mw"] - df["actual_mw"]).abs(),
    )
    return (work.groupby("hour")
                .agg(signed=("signed", "mean"),
                     absol=("absol", "mean"))
                .reset_index())


def main(windows: list[float]) -> None:
    cfg = load_config()
    spec = freeze.load_verified()
    cutoff = pd.Timestamp(spec["data_cutoff"])
    anchor_h = int(cfg.data.day_anchor_hours)
    dims = spec["predictor"]["embedding_dims"]
    raw_full = load_actuals(cutoff=None).dropna()
    raw_pre = raw_full[raw_full.index <= cutoff]
    days = _complete_delivery_days(raw_full, anchor_h)
    days = days[days > cutoff]
    if anchor_h > 0:
        days = days[:-1]
    print(f"sweep over W = {windows}")
    print(f"days: {len(days)}  ({days.min().date()}..{days.max().date()})")
    print()

    headlines = []
    per_hour_rows = []
    for W in windows:
        out_pkl = OUT / f"cell_B_W{int(W) if W == int(W) else W}y.pkl"
        if out_pkl.exists():
            print(f"  W={W}y: cached, loading {out_pkl.name}")
            fc = pd.read_pickle(out_pkl)
        else:
            t0 = time.time()
            print(f"  W={W}y: refitting climatology + rescoring ...", end="", flush=True)
            zp = _fit_zscore_params_window(raw_pre, W)
            fc = run_backtest(zp, raw_full, cutoff, anchor_h, dims, days,
                              log_every=0)
            actual = load_actuals(cutoff=None).dropna()
            fc["actual_mw"] = (fc["target_dt"].map(actual)
                                  .astype("float").round(0).astype("Int64"))
            fc = fc.dropna(subset=["actual_mw"])
            print(f" {time.time()-t0:.1f}s ({len(fc):,d} rows)")
            fc.to_pickle(out_pkl)
        headlines.append(_summarise(fc, W))
        bh = _by_hour(fc)
        bh["W_years"] = W
        per_hour_rows.append(bh)

    hdf = pd.DataFrame(headlines)
    pdf = pd.concat(per_hour_rows, ignore_index=True)
    hdf.to_csv(OUT / "headline.csv", index=False)
    pdf.to_csv(OUT / "per_hour.csv", index=False)

    print()
    print("=== headline ===")
    print(hdf.to_string(index=False,
                         formatters={
                             "W_years":      "{:.1f}".format,
                             "MAE_mw":       "{:7.1f}".format,
                             "MAPE_pct":     "{:5.2f}".format,
                             "coverage_pct": "{:5.1f}".format,
                             "mean_signed":  "{:+7.1f}".format,
                             "rms_signed":   "{:7.1f}".format,
                         }))

    # Add original (full-history) as the baseline if its pickle is around
    orig_pkl = ROOT / "scratch" / "data" / "smc_smap_samples" / "cell_B_meaniter_global.pkl"
    if orig_pkl.exists():
        orig = pd.read_pickle(orig_pkl)
        ofh = _by_hour(orig)
        orig_row = _summarise(orig, np.nan)
        orig_row["W_years"] = "orig"
        print()
        print(f"baseline (full-history climatology):")
        print(f"  MAE={orig_row['MAE_mw']:.1f}  MAPE={orig_row['MAPE_pct']:.2f}%  "
              f"cov={orig_row['coverage_pct']:.1f}%  "
              f"mean_signed={orig_row['mean_signed']:+.1f}  "
              f"rms_signed={orig_row['rms_signed']:.1f}")

    print()
    print("=== signed error by HE (forecast - actual, MW); cols = W in years ===")
    pivot_s = pdf.pivot(index="hour", columns="W_years", values="signed")
    if orig_pkl.exists():
        pivot_s["orig"] = ofh.set_index("hour")["signed"]
    print(pivot_s.round(0).to_string(float_format=lambda x: f"{int(x):>+5d}"))

    print()
    print("=== absolute error by HE (MW); cols = W in years ===")
    pivot_a = pdf.pivot(index="hour", columns="W_years", values="absol")
    if orig_pkl.exists():
        pivot_a["orig"] = ofh.set_index("hour")["absol"]
    print(pivot_a.round(0).to_string(float_format=lambda x: f"{int(x):>5d}"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--windows", type=float, nargs="+", default=list(DEFAULT_WINDOWS))
    args = p.parse_args()
    main(args.windows)
