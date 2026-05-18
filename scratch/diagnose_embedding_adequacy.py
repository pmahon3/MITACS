"""EXPLORATORY: is the M2 phase-centring bias caused by (i) missing
PHASE state and/or (ii) per-HOUR dimension mis-specification?
(Honing step-3 diagnostic; dev 2021-22 only; scratch/, not recorded.)

Two decisive tests, BEFORE building anything:

TEST A (phase): the z-space signed error swings monotonically
overnight(-)->evening(+). If it is structured against a PHASE/derivative
coordinate (dz/dt of recent demand), the embedding lacks phase state ->
a phase-augmented embedding is the lever. (Uses cached benchmark frame.)

TEST B (per-hour dimension): dims are resolved ONCE per day-type
(wkdy2/sat4/sun3) via the elbow rule on a whole-day-type rho-vs-d curve.
Re-run that SAME rho-vs-d elbow logic STRATIFIED BY HOUR-OF-DAY (dev,
<=cutoff). If optimal d varies materially by hour, AND high-z-bias hours
coincide with hours where the per-day-type d is far from the hour-local
optimum, then per-hour (or per-hour x day-type) dimension is the lever
(under-embedded at the complex evening ramp, over-embedded overnight).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from edynamics.modelling_tools import Embedding, Lag

from experiment._actuals import load_actuals, zscore_params, zscore_transform
from experiment.predict import _daytype

CACHE = Path("/tmp/bench2122_fc.pkl")
CUTOFF = pd.Timestamp("2021-10-18 23:00:00")
ANCHOR_H = 7


def _elbow(rho: pd.Series, tol: float = 0.001) -> int:
    peak = rho.max()
    return int(rho.index[rho >= peak - tol * abs(peak)][0])


# ---- TEST A: phase structure in the z-space residual --------------------
def test_phase() -> None:
    fc = pickle.loads(CACHE.read_bytes()).copy()
    fc = fc.dropna(subset=["actual_mw"]).sort_index()
    fc["z_actual"] = (fc["actual_mw"] - fc["mu_mh"]) / fc["sigma_mh"]
    fc["z_err"] = fc["z_pred"] - fc["z_actual"]
    # phase proxy: recent demand derivative (z-space hour-over-hour change
    # of the ACTUAL, i.e. information available at issue for the lag state)
    za = fc["z_actual"]
    fc["dz_dt"] = za.diff()                 # rising (+) vs falling (-)
    g = fc.dropna(subset=["dz_dt"])
    r = np.corrcoef(g["dz_dt"], g["z_err"])[0, 1]
    print("TEST A -- phase structure in z-space residual")
    print(f"  corr(dz/dt , z_err) = {r:+.3f}  "
          f"(|large| => residual depends on rising/falling phase => "
          f"embedding lacks phase state)")
    q = pd.qcut(g["dz_dt"], 5, labels=["fall--", "fall-", "flat",
                                       "rise+", "rise++"])
    print("  z_err mean by dz/dt quintile:")
    print(g.groupby(q, observed=True)["z_err"].mean().round(3).to_string())


# ---- TEST B: optimal embedding dim STRATIFIED BY HOUR-OF-DAY -------------
def test_per_hour_dim() -> None:
    act = load_actuals(cutoff=None)
    zp = zscore_params(CUTOFF)
    z = zscore_transform(act, zp)
    df = z.to_frame("zscore").asfreq("h")
    dvals = df.index[df.index <= CUTOFF]

    # current per-day-type dims (the frozen-style choice) for reference
    cur = {}
    for dn in ("weekday", "saturday", "sunday"):
        idx = dvals[[_daytype(t, ANCHOR_H) == dn for t in dvals]]
        rho = {}
        for d in range(1, 9):
            fl = idx[d:-1]
            if len(fl) < 400:
                break
            emb = Embedding(
                data=df,
                observers=[Lag(variable_name="zscore", tau=-i) for i in range(d)],
                library_times=fl,
            )
            emb.compile()
            b = emb.block
            X, Y = b.iloc[:-1].values, b.iloc[1:].values
            C = np.linalg.lstsq(X, Y, rcond=None)[0]
            rho[d] = np.corrcoef((X @ C)[:, 0], Y[:, 0])[0, 1]
        cur[dn] = _elbow(pd.Series(rho))
    print("\nTEST B -- per-hour-of-day optimal embedding dimension")
    print(f"  current per-day-type dims (reference): {cur}")
    print("  hod  opt_d  rho@opt   (stratified by hour-of-day, dev<=cutoff)")
    opt_by_h = {}
    for h in range(24):
        idx = dvals[dvals.hour == h]
        rho = {}
        for d in range(1, 9):
            fl = idx[d:-1]
            if len(fl) < 120:
                break
            emb = Embedding(
                data=df,
                observers=[Lag(variable_name="zscore", tau=-i) for i in range(d)],
                library_times=fl,
            )
            emb.compile()
            b = emb.block
            X, Y = b.iloc[:-1].values, b.iloc[1:].values
            if len(X) < 30:
                continue
            C = np.linalg.lstsq(X, Y, rcond=None)[0]
            rho[d] = np.corrcoef((X @ C)[:, 0], Y[:, 0])[0, 1]
        if rho:
            od = _elbow(pd.Series(rho))
            opt_by_h[h] = od
            print(f"  {h:>3}  {od:>5}  {rho[od]:>7.4f}")
    if opt_by_h:
        s = pd.Series(opt_by_h)
        print(f"  optimal-d across hours: min={s.min()} max={s.max()} "
              f"spread={s.max()-s.min()} (large => single per-day-type d "
              f"is wrong at some hours)")


def main() -> None:
    if not CACHE.exists() or "z_pred" not in pickle.loads(
        CACHE.read_bytes()
    ).columns:
        from scratch.benchmark_2021_22_ieso import compute

        fc, _ = compute()
        CACHE.write_bytes(pickle.dumps(fc))
    test_phase()
    test_per_hour_dim()
    print("\nVERDICT GUIDE: A-strong => phase-augmented embedding; "
          "B-spread large & aligned with high-z-bias hours => per-hour "
          "(x day-type) dimension; both weak => accept as documented "
          "operator limitation (don't engineer / don't overfit dev).")


if __name__ == "__main__":
    main()
