"""Exploratory probe: run the local-Gaussian conditional-kernel
estimator core on the IEEE DataPort Post-COVID competition load series.

NOT for the paper. A curiosity check on whether the estimator core
(processing.innovations.estimator._local_fit_at -- the production
local-linear-Gaussian fit) produces sane day-ahead forecasts on an
independent, much shorter, post-COVID-unstable demand series.

Why this is "Option C" and not a production-predictor backtest:
the production predictor's (month, hour) climatology needs multi-year
history to populate 288 cells; the competition data spans only
~3.5 months (2020-11-06 .. 2021-02-16). The (month, hour) lookup is
structurally inapplicable. This probe instead uses an hour-of-week
climatology (168 cells, ~14 weeks of data => ~14 samples/cell) --
the regime-appropriate minimal de-seasonalisation for sub-yearly
data -- so the ESTIMATOR CORE can be exercised without the
Ontario-specific preprocessing.

Setup:
  - data: scratch/data/postcovid/Actuals_Interim.csv ("Load (kW)")
  - de-seasonalise by hour-of-week mean/std on a training prefix
  - delay-embed; per-anchor _local_fit_at; iterate 24h day-ahead
  - internal train/test split: fit library = first N_TRAIN_DAYS,
    backtest day-ahead on the held-out tail
  - report MAE/MAPE in kW, plus library size and the theta the
    LOO-CV picks (small library => few neighbours => watch theta)

PROVENANCE-GRADE: INSPECTION-ONLY exploratory probe.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

from processing.innovations.estimator import _local_fit_at

DATA = "scratch/data/postcovid/Actuals_Interim.csv"
N_TRAIN_DAYS = 57          # library prefix; backtest on the ~14-day tail
EMBED_DIMS = [2, 3, 4]     # sweep -- no elbow CSV for this series


def _hour_of_week(idx: pd.DatetimeIndex) -> np.ndarray:
    """0..167: hour-of-day + 24*weekday. Captures the diurnal cycle
    AND the weekday/weekend split -- the dominant structure in a
    sub-yearly load series."""
    return idx.hour.values + 24 * idx.weekday.values


def _deseasonalise(load: pd.Series, train_mask: np.ndarray
                   ) -> tuple[pd.Series, pd.DataFrame]:
    """Hour-of-week z-score. Climatology fit ONLY on the training
    prefix (leakage guard); applied to the whole series."""
    how = _hour_of_week(load.index)
    df = pd.DataFrame({"load": load.values, "how": how}, index=load.index)
    train = df[train_mask]
    clim = train.groupby("how")["load"].agg(["mean", "std"])
    mu = clim["mean"].reindex(how).to_numpy()
    sd = clim["std"].reindex(how).to_numpy()
    z = pd.Series((df["load"].values - mu) / sd, index=load.index,
                  name="z")
    return z, clim


def _embed(z: pd.Series, d: int) -> tuple[np.ndarray, np.ndarray,
                                          pd.DatetimeIndex]:
    """Delay embedding: x_t = (z_t, z_{t-1}, ..., z_{t-d+1}).
    Returns (X, Y, anchor_times) where Y is x_{t+1}."""
    vals = z.to_numpy()
    n = len(vals)
    X, Y, times = [], [], []
    for t in range(d - 1, n - 1):
        x = vals[t - d + 1: t + 1][::-1]   # (z_t, z_{t-1}, ...)
        y = vals[t - d + 2: t + 2][::-1]
        if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
            X.append(x)
            Y.append(y)
            times.append(z.index[t])
    return np.array(X), np.array(Y), pd.DatetimeIndex(times)


def _backtest_one_dim(z: pd.Series, clim: pd.DataFrame, d: int,
                      train_end: pd.Timestamp,
                      load: pd.Series) -> dict:
    """Day-ahead iterated forecast on the held-out tail under
    embedding dimension d. Library = embedded pairs with anchor
    time <= train_end."""
    X_all, Y_all, anchor_times = _embed(z, d)
    lib_mask = anchor_times <= train_end
    X_lib, Y_lib = X_all[lib_mask], Y_all[lib_mask]

    # held-out delivery days: complete 24h days strictly after train_end
    z_full = z
    days = pd.DatetimeIndex(sorted({
        ts.normalize() for ts in z.index if ts > train_end
    }))
    rows = []
    thetas = []
    for D in days:
        targets = [D + pd.Timedelta(hours=h) for h in range(24)]
        if not all(t in z.index for t in targets):
            continue
        issue = targets[0] - pd.Timedelta(hours=1)
        need = [issue - pd.Timedelta(hours=i) for i in range(d)]
        if not all(t in z.index for t in need):
            continue
        zhist = {t: float(z.loc[t]) for t in z.index
                 if issue - pd.Timedelta(hours=d) <= t <= issue}
        for h_idx, t in enumerate(targets):
            prev = t - pd.Timedelta(hours=1)
            lag_t = [prev - pd.Timedelta(hours=i) for i in range(d)]
            try:
                xq = np.array([zhist[lt] for lt in lag_t], dtype=float)
            except KeyError:
                break
            C, Sigma, mu, theta, _ = _local_fit_at(X_lib, Y_lib, xq, d)
            z_next = float(xq @ (C[:, 0] if C.ndim == 2 else C))
            zhist[t] = z_next
            thetas.append(theta)
            # de-z-score back to kW via the hour-of-week climatology
            how = (t.hour + 24 * t.weekday())
            cmu = float(clim["mean"].loc[how])
            csd = float(clim["std"].loc[how])
            rows.append({
                "target": t, "horizon": h_idx + 1,
                "pred_kw": z_next * csd + cmu,
                "actual_kw": float(load.loc[t]),
            })
    if not rows:
        return {"d": d, "n": 0}
    fc = pd.DataFrame(rows)
    err = fc["pred_kw"] - fc["actual_kw"]
    mae = float(err.abs().mean())
    mape = float((err.abs() / fc["actual_kw"]).mean() * 100)
    # persistence-168h baseline on the same targets
    p168 = fc["target"].map(
        lambda t: float(load.get(t - pd.Timedelta(hours=168), np.nan))
    )
    p168_err = (p168 - fc["actual_kw"]).dropna()
    p168_mae = float(p168_err.abs().mean()) if len(p168_err) else np.nan
    return {
        "d": d, "n": len(fc), "n_days": fc["target"].dt.normalize().nunique(),
        "lib_size": len(X_lib),
        "mae_kw": mae, "mape_pct": mape,
        "p168_mae_kw": p168_mae,
        "theta_med": float(np.median(thetas)),
        "theta_min": float(np.min(thetas)),
        "theta_max": float(np.max(thetas)),
    }


def main() -> None:
    print("=" * 72)
    print("Post-COVID competition load: estimator-core probe (Option C)")
    print("=" * 72)

    raw = pd.read_csv(DATA)
    raw["Time"] = pd.to_datetime(raw["Time"])
    load = raw.set_index("Time")["Load (kW)"].sort_index().asfreq("h")
    print(f"load series: {load.index[0]} .. {load.index[-1]}, "
          f"n={len(load)}, kW range {load.min():.0f}-{load.max():.0f}")

    train_end = load.index[0] + pd.Timedelta(days=N_TRAIN_DAYS)
    train_mask = np.asarray(load.index <= train_end)
    print(f"train prefix: <= {train_end} "
          f"({train_mask.sum()} h); backtest on the tail "
          f"({(~train_mask).sum()} h)")

    z, clim = _deseasonalise(load, train_mask)
    print(f"hour-of-week climatology: {len(clim)} cells "
          f"(of 168 possible)")
    print(f"  z (post-deseasonalise): mean={z.mean():+.4f}, "
          f"std={z.std():.4f}, kurt={z.kurtosis():.2f}")
    print()

    print(f"{'d':>3} {'lib':>6} {'days':>5} {'n':>5} "
          f"{'MAE_kW':>9} {'MAPE%':>7} {'p168_MAE':>9} "
          f"{'theta(med/min/max)':>22}")
    print("-" * 72)
    for d in EMBED_DIMS:
        r = _backtest_one_dim(z, clim, d, train_end, load)
        if r.get("n", 0) == 0:
            print(f"{d:>3}  (no complete held-out days)")
            continue
        print(f"{r['d']:>3} {r['lib_size']:>6} {r['n_days']:>5} "
              f"{r['n']:>5} {r['mae_kw']:>9.0f} {r['mape_pct']:>6.2f}% "
              f"{r['p168_mae_kw']:>9.0f}  "
              f"{r['theta_med']:.3f}/{r['theta_min']:.3f}/"
              f"{r['theta_max']:.3f}")

    print()
    print("Reading: compare MAE_kW to p168_MAE (seasonal persistence at")
    print("the same hour one week prior). If the estimator doesn't beat")
    print("p168, the local-library approach has too few useful neighbours")
    print("on this short, post-COVID-unstable series -- which would")
    print("vindicate the intuition against this dataset.")


if __name__ == "__main__":
    main()
