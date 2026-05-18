"""EXPLORATORY 2021-22 IESO benchmark -- FOR INSPECTION ONLY.

NOT the registered experiment. NOT a recorded result. Kept in scratch/
deliberately. A separate mini-benchmark with the kappa_Q model
RE-FROZEN strictly before 2021-10-19, forecasting 2021-10-19..2022-02-27
day-ahead, vs the recovered historical IESO OntarioZonalDemand archive
(git history; Ontario-demand basis; day-ahead slice selected by
CreationDate = the ~09:00 issue on the calendar day BEFORE each target,
which is CreationDate-VERIFIED day-ahead).

Honest scope:
  * model re-specified with data strictly < 2021-10-19 (fit library +
    z-score climatology) so the window is genuinely out-of-sample for
    THIS frozen model -- distinct from the registered 2024-cutoff one;
  * dims re-derived from pre-2021-10-19 (elbow rule) -- not the 2024 spec;
  * per-hour-of-day MAPE (the established honest metric);
  * IESO selected at true day-ahead lead via CreationDate.
This is a look, not a deliverable. Reuses the validated estimator core
(no reimplementation).

PROVENANCE-GRADE: INSPECTION-ONLY -- exploratory dev-set; MUST NOT be
cited as a result (Task 36 / PROVENANCE_REQUIREMENTS.md C6, gap #6).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from edynamics.modelling_tools import Embedding, Lag

from experiment._actuals import (
    load_actuals,
    zscore_params,
    zscore_transform,
)
from experiment.predict import _daytype
from processing.innovations.estimator import _local_fit_at

CUTOFF = pd.Timestamp("2021-10-18 23:00:00")
WIN_START = pd.Timestamp("2021-10-19")
WIN_END = pd.Timestamp("2022-02-27")
ANCHOR_H = 7  # day-anchor offset (same convention as the pipeline)


def _elbow_dim(rho_by_d: pd.Series, tol: float = 0.001) -> int:
    peak = rho_by_d.max()
    return int(rho_by_d.index[rho_by_d >= peak - tol * abs(peak)][0])


def _refrozen_dims(z: pd.Series) -> dict:
    """Re-derive embedding dim per day-type from pre-cutoff data via the
    same elbow rule the registered spec uses (1-step rho vs d)."""
    df = z.to_frame("zscore").asfreq("h")
    out = {}
    for dtname in ("weekday", "saturday", "sunday"):
        idx = df.index[[_daytype(t, ANCHOR_H) == dtname for t in df.index]]
        idx = idx[idx <= CUTOFF]
        rho = {}
        for d in range(1, 9):
            lags = [Lag(variable_name="zscore", tau=-i) for i in range(d)]
            fl = idx[d:-1]
            if len(fl) < 500:
                break
            emb = Embedding(data=df, observers=lags, library_times=fl)
            emb.compile()
            b = emb.block
            X, Y = b.iloc[:-1].values, b.iloc[1:].values
            # cheap 1-step in-sample rho proxy (dim-selection only)
            C = np.linalg.lstsq(X, Y, rcond=None)[0]
            pred = (X @ C)[:, 0]
            rho[d] = np.corrcoef(pred, Y[:, 0])[0, 1]
        out[dtname] = _elbow_dim(pd.Series(rho))
    return out


def _ieso_day_ahead(target_dates: pd.DatetimeIndex) -> pd.Series:
    """Historical IESO Ontario forecast, day-ahead slice: per target hour
    take the row whose CreationDate is the latest on the calendar day
    BEFORE the target date (the ~09:00 D-1 issue)."""
    import subprocess

    raw = subprocess.run(
        ["git", "show",
         "5ea811d:src/main/resources/data/demand/forecasts/ieso_forecasts.csv"],
        capture_output=True, text=True, check=True,
    ).stdout
    from io import StringIO

    d = pd.read_csv(StringIO(raw))
    o = d[d.Zone == "Ontario"].copy()
    o["dt"] = pd.to_datetime(o.Date) + pd.to_timedelta(
        o.Hour.astype(int) - 1, unit="h"
    )
    o["created"] = pd.to_datetime(o.CreationDate, errors="coerce")
    o = o.dropna(subset=["created", "Demand"])
    o["target_date"] = o["dt"].dt.normalize()
    o["created_date"] = o["created"].dt.normalize()
    # day-ahead: issue on the day immediately before the target date
    da = o[o["created_date"] == (o["target_date"] - pd.Timedelta(days=1))]
    # if multiple issues that day, take the latest CreationDate
    da = da.sort_values("created").drop_duplicates("dt", keep="last")
    return da.set_index("dt")["Demand"].sort_index()


def compute() -> tuple[pd.DataFrame, dict]:
    """Build the 2021-22 day-ahead forecast frame (ours/actual/ieso) +
    the re-frozen dims. Returned for reuse by the error diagnostic so the
    expensive forecast loop runs once."""
    act = load_actuals(cutoff=None)
    zp = zscore_params(CUTOFF)               # climatology strictly <= cutoff
    z_full = zscore_transform(act, zp)
    dims = _refrozen_dims(z_full)

    df = z_full.to_frame("zscore").asfreq("h")
    lib_cache: dict[int, tuple] = {}

    days = pd.date_range(WIN_START, WIN_END, freq="D")
    rows = []
    for D in days:
        targets = [D + pd.Timedelta(hours=h) for h in range(24)]
        issue_anchor = targets[0] - pd.Timedelta(hours=1)  # D-1 23:00
        dmax = max(dims.values())
        need = [issue_anchor - pd.Timedelta(hours=i) for i in range(dmax)]
        if not all(n in z_full.index for n in need):
            continue
        zhist = {
            ts: float(z_full.loc[ts])
            for ts in z_full.index
            if issue_anchor - pd.Timedelta(hours=dmax) <= ts <= issue_anchor
        }
        for t in targets:
            dt = _daytype(t, ANCHOR_H)
            d = int(dims[dt])
            if d not in lib_cache:
                lags = [Lag(variable_name="zscore", tau=-i) for i in range(d)]
                fl = df.index[df.index <= CUTOFF][d:-1]
                emb = Embedding(data=df, observers=lags, library_times=fl)
                emb.compile()
                b = emb.block
                lib_cache[d] = (b.iloc[:-1].values, b.iloc[1:].values)
            X, Y = lib_cache[d]
            prev = t - pd.Timedelta(hours=1)
            lag_t = [prev - pd.Timedelta(hours=i) for i in range(d)]
            try:
                xq = np.array([zhist[lt] for lt in lag_t], dtype=float)
            except KeyError:
                break
            C, _S, _mu, _th, _ = _local_fit_at(X, Y, xq, d)
            z_next = float(xq @ (C[:, 0] if C.ndim == 2 else C))
            zhist[t] = z_next
            key = (t.month, t.hour)
            mu = float(zp["mu_mh"].loc[key])
            sd = float(zp["sigma_mh"].loc[key])
            # diagnostic instrumentation: per-row theta + a local-density
            # proxy (distance from the query to its 50th nearest library
            # point in the embedding -- larger => sparser region).
            knn50 = float(
                np.partition(np.linalg.norm(X - xq, axis=1), 50)[50]
            )
            rows.append({
                "dt": t, "ours_mw": z_next * sd + mu,
                "theta": float(_th), "q_z0": float(xq[0]),
                "knn50_dist": knn50,
                # mechanism diagnostics: z-space forecast + the
                # round-trip factors + horizon, so the diurnal bias can
                # be decomposed (z-space error vs sigma_mh amplification
                # vs horizon-1 climatology offset).
                "z_pred": z_next,
                "mu_mh": mu, "sigma_mh": sd,
                "horizon_h": int((t - targets[0]) / pd.Timedelta(hours=1)) + 1,
            })

    fc = pd.DataFrame(rows).set_index("dt")
    fc["actual_mw"] = fc.index.map(act)
    ieso = _ieso_day_ahead(pd.DatetimeIndex(fc.index))
    fc["ieso_mw"] = fc.index.map(ieso)
    fc = fc.dropna(subset=["actual_mw"])
    return fc, dims


def mape(p, a):
    m = a != 0
    return float(((p[m] - a[m]).abs() / a[m]).mean() * 100)


def run() -> None:
    fc, dims = compute()
    print(f"re-frozen dims (pre-{CUTOFF.date()}): {dims}")
    n_ieso = fc["ieso_mw"].notna().sum()
    print(f"\n2021-22 benchmark (FOR INSPECTION ONLY -- not recorded)")
    print(f"  window {WIN_START.date()}..{WIN_END.date()}  "
          f"hourly={len(fc)}  with IESO day-ahead={n_ieso}")
    print(f"  OURS  vs actual : MAPE {mape(fc['ours_mw'], fc['actual_mw']):.2f}%"
          f"  MAE {(fc['ours_mw']-fc['actual_mw']).abs().mean():.0f} MW")
    g = fc.dropna(subset=["ieso_mw"])
    if len(g):
        print(f"  IESO  vs actual : MAPE {mape(g['ieso_mw'], g['actual_mw']):.2f}%"
              f"  MAE {(g['ieso_mw']-g['actual_mw']).abs().mean():.0f} MW"
              f"  (n={len(g)}, true day-ahead via CreationDate)")
        print(f"  OURS on same n  : MAPE {mape(g['ours_mw'], g['actual_mw']):.2f}%")
        print("  by hour-of-day (MAPE):  hod  ours  ieso  n")
        for h, gg in g.groupby(g.index.hour):
            print(f"    {h:>2}  {mape(gg['ours_mw'],gg['actual_mw']):>5.2f}  "
                  f"{mape(gg['ieso_mw'],gg['actual_mw']):>5.2f}  {len(gg)}")
    print("\n  CAVEATS: pre-cutoff separate model (NOT the registered "
          "experiment); IESO horizon CreationDate-verified day-ahead; "
          "exploratory inspection only.")


if __name__ == "__main__":
    run()
