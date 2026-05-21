"""Option H — minimal version: fit Pi_h directly at h in {1, 3, 8}
and compare per-h residual kurt to the COMPOSED kurt from k_capture's
augmented-propagation cache. Closes the qualifier-exhaustion-memo
Stage-1 Single-step verdict's open closing condition:

  "verify direct h-step fitting preserves the compositional property
   that residuals become approximately Gaussian at long h"

The cheap version reuses the same forward-iteration structure as
k_capture but skips composition entirely — at each delivery day's
target hour, fit Pi_h DIRECTLY (regress z_{t+h} on x_t over the
library) instead of iterating one-step h times.

Output: per-h residual kurt + day-bootstrap CI, compared to the
composed-residual kurt from k_capture's cache.

Scope: h in {1, 3, 8}. Three horizons spaced to cover the regimes
(load-bearing h=1; mid kurt-drop h=3; CI-clean approximately-Gaussian
h=8).

PROVENANCE-GRADE: INSPECTION-ONLY.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from edynamics.modelling_tools import Embedding, Lag

from experiment._actuals import load_actuals, zscore_params, zscore_transform
from experiment.predict import _daytype
from processing.innovations.estimator import _local_fit_at

from scratch.benchmark_2021_22_ieso import (
    ANCHOR_H, CUTOFF, WIN_END, WIN_START, _refrozen_dims,
)

HS = [1, 3, 8]
OUT = Path("/tmp/h_minimal_resid.pkl")
RNG = np.random.default_rng(20260520)
B = 1000


def _say(msg, t0):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def _kurt(x):
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return float("nan")
    return float(pd.Series(x).kurtosis())


def _ci_kurt(x, days, B=B):
    uniq = np.unique(days)
    by_day = {d: np.where(days == d)[0] for d in uniq}
    boots = np.empty(B)
    for b in range(B):
        samp = RNG.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_day[d] for d in samp])
        boots[b] = _kurt(x[idx])
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return _kurt(x), float(lo), float(hi)


def main():
    t0 = time.time()
    _say("loading actuals", t0)
    act = load_actuals(cutoff=None)
    zp = zscore_params(CUTOFF)
    z = zscore_transform(act, zp)
    dims = _refrozen_dims(z)
    _say(f"dims = {dims}", t0)
    df = z.to_frame("zscore").asfreq("h")

    # Build per-day-type library for direct h-step regression.
    # For each h: x_t -> z_{t+h} (one-step library uses tau=-i for i in 0..d-1;
    # the target shifts h steps forward).
    lib_cache = {}      # keyed by (dt, d, h)
    for dt in ("weekday", "saturday", "sunday"):
        d = dims[dt]
        idx_full = df.index[df.index <= CUTOFF]
        idx_dt = idx_full[[_daytype(t, ANCHOR_H) == dt for t in idx_full]]
        for h in HS:
            # need at least d lags before + h steps ahead
            usable = idx_dt[d : -(h + 1)]
            if len(usable) < 200:
                continue
            lags = [Lag(variable_name="zscore", tau=-i) for i in range(d)]
            emb = Embedding(data=df, observers=lags,
                            library_times=usable)
            emb.compile()
            b = emb.block.values     # X at lag times
            # Y_h: z_{t+h} (one scalar) at each library time t
            Y_h = np.array([
                float(z.loc[t + pd.Timedelta(hours=h)])
                if (t + pd.Timedelta(hours=h)) in z.index
                else np.nan
                for t in usable
            ])
            mask = np.isfinite(Y_h)
            lib_cache[(dt, d, h)] = (b[mask], Y_h[mask])

    days = pd.date_range(WIN_START, WIN_END, freq="D")
    _say(f"direct-h fitting on {len(days)} delivery days x {len(HS)} h", t0)
    rows = []
    for D_i, D in enumerate(days):
        for h_idx, h in enumerate(HS):
            target = D + pd.Timedelta(hours=h - 1)
            dt = _daytype(target, ANCHOR_H)
            d = dims[dt]
            anchor = target - pd.Timedelta(hours=h)
            need = [anchor - pd.Timedelta(hours=i) for i in range(d)]
            if not all(n in z.index for n in need):
                continue
            xq = np.array([float(z.loc[n]) for n in need], dtype=float)
            key = (dt, d, h)
            if key not in lib_cache:
                continue
            X, Y = lib_cache[key]
            # Local fit for the scalar z_{target} = z_{anchor + h}
            try:
                C, S, _mu, _th, _ = _local_fit_at(
                    X, Y[:, None], xq, d
                )
                z_pred = float(xq @ (C[:, 0] if C.ndim == 2 else C))
                sigma00 = float(S[0, 0])
            except Exception:
                continue
            z_act = float(z.loc[target]) if target in z.index else np.nan
            rows.append({
                "day": D, "h": h, "daytype": dt,
                "z_pred": z_pred, "z_actual": z_act, "sigma00": sigma00,
            })
        if (D_i + 1) % 30 == 0:
            _say(f"  {D_i + 1} days processed", t0)

    df_r = pd.DataFrame(rows)
    df_r = df_r.dropna(subset=["z_actual", "z_pred", "sigma00"]).copy()
    df_r["z_std"] = ((df_r["z_actual"] - df_r["z_pred"])
                     / np.sqrt(np.maximum(df_r["sigma00"], 1e-9)))
    with OUT.open("wb") as f:
        pickle.dump({"rows": df_r, "dims": dims}, f)
    _say(f"saved {OUT} ({len(df_r)} rows)", t0)

    # Compare: direct-h kurt vs composed kurt (the latter from
    # scratch/hstep_kurtosis.py output, hard-coded here for the 3 h's).
    composed_pt = {1: 8.60, 3: 1.13, 8: -0.34}     # pooled, baseline
    composed_ci = {1: (0.75, 13.57), 3: (-0.04, 2.29),
                   8: (-0.83, 0.32)}
    print()
    print("=" * 80)
    print("Direct-h vs composed-h excess kurt (pooled across day-types):")
    print("=" * 80)
    print(f"  {'h':>3} {'n':>5} {'direct kurt':>12} {'CI':>20} "
          f"{'composed (cache)':>18} {'CI':>16}")
    for h in HS:
        g = df_r[df_r["h"] == h]
        if len(g) < 30:
            print(f"  {h:>3}  n<30, skipped")
            continue
        pt, lo, hi = _ci_kurt(g["z_std"].to_numpy(), g["day"].to_numpy())
        ck = composed_pt.get(h)
        clo, chi = composed_ci.get(h, (None, None))
        print(f"  {h:>3} {len(g):>5} {pt:>12.3f} "
              f"[{lo:>+6.2f},{hi:>+6.2f}] "
              f"{ck:>18.2f} [{clo:>+5.2f},{chi:>+5.2f}]")

    # Verdict
    print()
    if 1 in df_r["h"].values and 8 in df_r["h"].values:
        k1 = _kurt(df_r[df_r["h"] == 1]["z_std"].to_numpy())
        k8 = _kurt(df_r[df_r["h"] == 8]["z_std"].to_numpy())
        print(f"Direct-h k(1) = {k1:.2f}, k(8) = {k8:.2f}")
        if k8 < 3 and k1 > 5:
            print("Direct-h trend MIRRORS composed-h: kurt decays with h.")
            print("Single-step closing condition: MET — direct h-step "
                  "fitting preserves the compositional property.")
        else:
            print("Direct-h trend DOES NOT mirror composed-h.")
            print("Single-step closing condition: NOT cleanly met.")


if __name__ == "__main__":
    main()
