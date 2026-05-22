"""Exploratory probe: run the local-Gaussian conditional-kernel
estimator core under the IEEE DataPort Post-COVID competition's exact
day-ahead protocol.

NOT for the paper. This is the curiosity check the user asked for
("It's just intuition. I want to run it") on whether the estimator
core produces competitive day-ahead forecasts on the competition's
own test window and issue protocol.

Competition protocol (IEEE DataPort Post-COVID load forecasting,
arXiv/IEEE 9739082):
  - issue time: 08:00 on day d-1
  - forecast horizons 16..40 h ahead -> the 24 hours 00:00..23:00 of
    day d
  - metric: MAE in kW (competition reports MW; mean load ~1.1 GW)
  - winning aggregate MAE = 10.9 MW; best single method ~11.2 MW

COVID constraint (user: "You can't use the full 4 year history
because of covid"): the climatology must be fit on POST-COVID data
only. A (month, hour) Ontario-style climatology blends pre/post-COVID
regimes and is structurally inapplicable to a ~3.5-month series
anyway. This probe uses an hour-of-week climatology (168 cells) fit on
the post-COVID prefix strictly before the test window.

Data: scratch/data/postcovid/Actuals_full_postcovid.csv -- the
competition load series stitched from Actuals_Interim.csv plus the
dated daily batch files (2020-11-06 .. 2021-02-16, contiguous, no
gaps). "Load (kW)" column.

Setup:
  - test window: complete delivery days Jan 18 .. Feb 16 2021
  - for each delivery day d: climatology fit on all data strictly
    before the 08:00 day-(d-1) issue time (expanding window, no
    leakage); estimator library = embedded pairs with anchor time
    < issue
  - de-seasonalise by hour-of-week z-score; delay-embed; per-anchor
    _local_fit_at; iterate the one-step drift forward from the issue
    state through horizon 40 h; keep the 24 day-d hours
  - report MAE/MAPE in kW vs the competition's 10.9 MW winner, plus a
    same-protocol hour-of-week climatology baseline and a t-168h
    seasonal-persistence baseline

PROVENANCE-GRADE: INSPECTION-ONLY exploratory probe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from processing.innovations.estimator import _local_fit_at

DATA = "scratch/data/postcovid/Actuals_full_postcovid.csv"
TEST_START = pd.Timestamp("2021-01-18")
TEST_END = pd.Timestamp("2021-02-16")
ISSUE_HOUR = 8                       # 08:00 day d-1
EMBED_DIMS = [2, 3, 4]               # sweep -- no elbow CSV for this series


def _hour_of_week(idx: pd.DatetimeIndex) -> np.ndarray:
    """0..167: hour-of-day + 24*weekday -- diurnal cycle plus the
    weekday/weekend split, the dominant structure in a sub-yearly
    load series."""
    return idx.hour.values + 24 * idx.weekday.values


def _climatology(load: pd.Series, before: pd.Timestamp) -> pd.DataFrame:
    """Hour-of-week mean/std fit on data STRICTLY before `before`
    (expanding window; leakage guard)."""
    train = load[load.index < before]
    how = _hour_of_week(train.index)
    df = pd.DataFrame({"load": train.values, "how": how}, index=train.index)
    return df.groupby("how")["load"].agg(["mean", "std"])


def _z(load_vals: np.ndarray, idx: pd.DatetimeIndex,
       clim: pd.DataFrame) -> np.ndarray:
    """Hour-of-week z-score under a fixed climatology table."""
    how = _hour_of_week(idx)
    mu = clim["mean"].reindex(how).to_numpy()
    sd = clim["std"].reindex(how).to_numpy()
    return (load_vals - mu) / sd


def _embed_library(z: pd.Series, d: int,
                   before: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """Delay-embedded (X, Y=x_{t+1}) pairs with anchor time strictly
    before `before`."""
    vals = z.to_numpy()
    times = z.index
    X, Y = [], []
    for t in range(d - 1, len(vals) - 1):
        if times[t] >= before:
            continue
        x = vals[t - d + 1: t + 1][::-1]
        y = vals[t - d + 2: t + 2][::-1]
        if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
            X.append(x)
            Y.append(y)
    return np.array(X), np.array(Y)


def _backtest_one_dim(load: pd.Series, d: int) -> dict:
    """Competition-protocol day-ahead backtest at embedding dim d."""
    rows = []
    thetas = []
    days = pd.date_range(TEST_START, TEST_END, freq="D")
    for D in days:
        issue = D - pd.Timedelta(days=1) + pd.Timedelta(hours=ISSUE_HOUR)
        targets = [D + pd.Timedelta(hours=h) for h in range(24)]
        # need the 24 day-d hours AND d hours of history at/before issue
        hist_times = [issue - pd.Timedelta(hours=i) for i in range(d)]
        if not all(t in load.index for t in targets + hist_times):
            continue
        if load.loc[targets + hist_times].isna().any():
            continue

        # climatology + library frozen at the issue time (no leakage)
        clim = _climatology(load, issue)
        z_full = pd.Series(
            _z(load.to_numpy(), load.index, clim), index=load.index
        )
        X_lib, Y_lib = _embed_library(z_full, d, issue)
        if len(X_lib) < 50:
            continue

        # iterate the one-step drift from the issue state forward.
        # zhist holds z at every hour from (issue-d) .. issue, then we
        # extend it hour by hour to horizon 40h (issue+1 .. D+23h).
        zhist = {t: float(z_full.loc[t])
                 for t in load.index
                 if issue - pd.Timedelta(hours=d) <= t <= issue}
        horizon_end = D + pd.Timedelta(hours=23)
        t = issue
        while t < horizon_end:
            t_next = t + pd.Timedelta(hours=1)
            lag_t = [t - pd.Timedelta(hours=i) for i in range(d)]
            try:
                xq = np.array([zhist[lt] for lt in lag_t], dtype=float)
            except KeyError:
                break
            C, Sigma, mu, theta, _ = _local_fit_at(X_lib, Y_lib, xq, d)
            z_next = float(xq @ (C[:, 0] if C.ndim == 2 else C))
            zhist[t_next] = z_next
            thetas.append(theta)
            t = t_next

        for h_idx, tgt in enumerate(targets):
            if tgt not in zhist:
                break
            how = int(tgt.hour + 24 * tgt.weekday())
            cmu = float(clim["mean"].loc[how])
            csd = float(clim["std"].loc[how])
            rows.append({
                "target": tgt,
                "horizon_h": h_idx + 16,   # 00:00 of day d is +16h
                "pred_kw": zhist[tgt] * csd + cmu,
                "clim_kw": cmu,            # climatology-only forecast
                "actual_kw": float(load.loc[tgt]),
            })

    if not rows:
        return {"d": d, "n": 0}
    fc = pd.DataFrame(rows)
    err = fc["pred_kw"] - fc["actual_kw"]
    clim_err = fc["clim_kw"] - fc["actual_kw"]
    p168 = fc["target"].map(
        lambda t: float(load.get(t - pd.Timedelta(hours=168), np.nan))
    )
    p168_err = (p168 - fc["actual_kw"]).dropna()
    return {
        "d": d,
        "n": len(fc),
        "n_days": fc["target"].dt.normalize().nunique(),
        "mae_kw": float(err.abs().mean()),
        "mape_pct": float((err.abs() / fc["actual_kw"]).mean() * 100),
        "clim_mae_kw": float(clim_err.abs().mean()),
        "clim_mape_pct": float(
            (clim_err.abs() / fc["actual_kw"]).mean() * 100),
        "p168_mae_kw": float(p168_err.abs().mean()) if len(p168_err) else np.nan,
        "theta_med": float(np.median(thetas)),
        "theta_min": float(np.min(thetas)),
        "theta_max": float(np.max(thetas)),
    }


def main() -> None:
    print("=" * 74)
    print("Post-COVID competition load: competition-protocol estimator probe")
    print("=" * 74)

    raw = pd.read_csv(DATA, index_col=0, parse_dates=True)
    load = raw["Load (kW)"].sort_index().asfreq("h")
    print(f"load series: {load.index[0]} .. {load.index[-1]}, "
          f"n={len(load)}, missing={load.isna().sum()}")
    print(f"  mean load = {load.mean()/1e3:.0f} MW  "
          f"(competition winner MAE = 10.9 MW => ~{10.9/(load.mean()/1e3)*100:.2f}% MAPE)")
    print(f"protocol: issue 08:00 day d-1, horizons 16-40h, "
          f"test {TEST_START.date()}..{TEST_END.date()}")
    print()

    print(f"{'d':>3} {'days':>5} {'n':>5} "
          f"{'MAE_kW':>9} {'MAPE%':>7} {'clim_MAE':>9} {'clim_MAPE%':>10} "
          f"{'p168_MAE':>9} {'theta(med/mn/mx)':>20}")
    print("-" * 74)
    for d in EMBED_DIMS:
        r = _backtest_one_dim(load, d)
        if r.get("n", 0) == 0:
            print(f"{d:>3}  (no complete delivery days)")
            continue
        print(f"{r['d']:>3} {r['n_days']:>5} {r['n']:>5} "
              f"{r['mae_kw']:>9.0f} {r['mape_pct']:>6.2f}% "
              f"{r['clim_mae_kw']:>9.0f} {r['clim_mape_pct']:>9.2f}% "
              f"{r['p168_mae_kw']:>9.0f}  "
              f"{r['theta_med']:.2f}/{r['theta_min']:.2f}/{r['theta_max']:.2f}")

    print()
    print("Reading:")
    print(" * MAE_kW vs 10900 kW (competition winner). The winner used")
    print("   exogenous weather/calendar inputs + an ensemble; this probe")
    print("   uses ONLY lagged load. A gap is expected and informative.")
    print(" * MAE_kW vs clim_MAE: does the estimator's local-linear drift")
    print("   beat its own hour-of-week climatology? If not, the local")
    print("   library has too few useful neighbours on this short,")
    print("   post-COVID-unstable series -- vindicating the intuition.")
    print(" * MAE_kW vs p168_MAE: vs naive seasonal persistence.")


if __name__ == "__main__":
    main()
