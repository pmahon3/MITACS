"""Apply the Patra-Sen estimator to Ontario rebaseline residuals.

For each (day-type × preprocessing), pull the standardised library
residuals from production-path local_drift_and_diffusion, then run
Patra-Sen with F_b = N(0,1). Report α̂_0 (point) and α̂_L^95%
(lower CI) per condition.

Reading:
  α̂_L^95% > 0  ⟹  residuals contain a contamination component
                    with at least α̂_L^95% mass (cleanly detected
                    against the N(0,1) null at 95% confidence).
  α̂_L^95% = 0  ⟹  data is compatible with all-N(0,1) at the
                    homogeneity test's 95% bar; no contamination
                    detected.

Identifiability caveat (Patra-Sen Lemma 4): if the contamination
shape shares full support with N(0,1) (e.g. a wider Gaussian), the
estimator gives a LOWER BOUND on α, not α itself. For our purpose
(testing whether ANY PFA-like contamination is present), the lower
bound is what we want.

Sampling protocol:
  - Per day-type, pick 30 representative anchors at random from the
    pre-cutoff intra-day library.
  - For each anchor: pull r0 = scalar residuals from
    local_drift_and_diffusion, standardise by sqrt(Sigma[0,0]).
  - Subsample to a cap (~5000 per anchor) to keep PAVA fast.
  - Run Patra-Sen on the POOLED residuals across the 30 anchors.

Cross-check: compare α̂ under (month-hour, Fourier). If similar,
contamination is intrinsic to κ_Q. If different, contamination is
preprocessing-induced.

PROVENANCE-GRADE: INSPECTION-ONLY.
"""
from __future__ import annotations

import time
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.stats import norm

from edynamics.modelling_tools import Embedding, Lag
from config import load_config
from experiment._actuals import load_actuals
from processing.innovations.estimator import local_drift_and_diffusion
from scratch.fourier_climatology import fit_fourier_params, fourier_transform
from scratch.benchmark_2021_22_ieso import CUTOFF
from scratch.patra_sen import patra_sen

N_ANCHORS = 30
N_PER_ANCHOR_CAP = 2000
RNG_SEED = 7


def _intraday_mask(fl, dayid, d):
    keep = []
    for s in fl:
        win = pd.date_range(
            s - pd.Timedelta(hours=d - 1), s + pd.Timedelta(hours=1), freq="h"
        )
        try:
            keep.append(dayid.loc[win].nunique() == 1)
        except KeyError:
            keep.append(False)
    return np.array(keep)


def _build_zseries(preprocessing: str) -> pd.Series:
    """Build the standardised z-series for the given preprocessing."""
    act = load_actuals(cutoff=None).asfreq("h")
    if preprocessing == "month-hour":
        cfg = load_config()
        df = pd.read_csv(
            cfg.paths.clustered_csv, index_col=0, parse_dates=True
        ).asfreq("h")
        # The clustered_csv has the production z column (month-hour preprocessed)
        return df["zscore"].copy(), df
    elif preprocessing == "fourier":
        fparams = fit_fourier_params(act, CUTOFF)
        z = fourier_transform(act, fparams)
        df = z.to_frame("zscore")
        # add daytype col mirroring the clustered_csv structure
        from experiment.predict import _daytype as _dt
        cfg = load_config()
        ah = cfg.data.day_anchor_hours
        df["daytype"] = [_dt(t, ah) for t in df.index]
        return z, df
    else:
        raise ValueError(preprocessing)


def _pool_standardised_residuals(z: pd.Series, df: pd.DataFrame,
                                 daytype: str, d: int, ah: int,
                                 n_anchors: int, n_per_anchor: int,
                                 rng) -> np.ndarray:
    """Pool standardised library residuals across n_anchors anchors."""
    lags = [Lag(variable_name="zscore", tau=-i) for i in range(d)]
    fl_full = df.index[df["daytype"] == daytype][d:-1]
    fl_full = fl_full[fl_full <= CUTOFF]
    dayid = pd.Series((df.index.hour == ah).cumsum(), index=df.index)
    fl_intra = fl_full[_intraday_mask(fl_full, dayid, d)]

    emb_intra = Embedding(data=df, observers=lags, library_times=fl_intra)
    emb_intra.compile()

    anchors = fl_intra[
        np.sort(rng.choice(len(fl_intra),
                           min(n_anchors, len(fl_intra)), replace=False))
    ]

    pooled = []
    for a in anchors:
        try:
            _C, S, _mu, _theta, resid = local_drift_and_diffusion(
                embedding=emb_intra, anchor=a, day_anchor_hour=ah,
            )
        except Exception:
            continue
        r0 = resid[:, 0]
        s00 = float(S[0, 0])
        if s00 <= 0 or not np.isfinite(s00):
            continue
        z_std = r0 / np.sqrt(s00)
        z_std = z_std[np.isfinite(z_std)]
        if len(z_std) > n_per_anchor:
            idx = rng.choice(len(z_std), n_per_anchor, replace=False)
            z_std = z_std[idx]
        pooled.append(z_std)

    return np.concatenate(pooled) if pooled else np.array([])


def _run_one_condition(preprocessing: str, daytype: str, dim: int,
                       n_anchors: int, n_per_anchor: int,
                       z: pd.Series, df: pd.DataFrame,
                       ah: int, t0: float) -> dict:
    """Run Patra-Sen for one (preprocessing, daytype) cell."""
    print(f"[{time.time()-t0:6.0f}s] pooling residuals: "
          f"{preprocessing} | {daytype} | d={dim}", flush=True)
    rng = np.random.default_rng(RNG_SEED)
    pooled = _pool_standardised_residuals(
        z, df, daytype, dim, ah, n_anchors, n_per_anchor, rng
    )
    if len(pooled) < 100:
        return {"preprocessing": preprocessing, "daytype": daytype,
                "n_pooled": len(pooled), "error": "too few residuals"}

    # Quick sanity: mean and std of standardised residuals
    mean_std = float(pooled.mean())
    std_std = float(pooled.std())
    kurt_pooled = float(pd.Series(pooled).kurtosis())
    print(f"[{time.time()-t0:6.0f}s]   n_pooled={len(pooled)}, "
          f"mean={mean_std:+.4f}, std={std_std:.4f}, "
          f"kurt={kurt_pooled:.2f}", flush=True)

    print(f"[{time.time()-t0:6.0f}s]   running Patra-Sen "
          f"(gamma_grid=600 x O(n log n) PAVA per gamma)...", flush=True)
    res = patra_sen(pooled, F_b=lambda x: norm.cdf(x), gamma_grid_size=600)
    print(f"[{time.time()-t0:6.0f}s]   → α̂ = {res.alpha_hat:.4f}, "
          f"α̂_L^95% = {res.alpha_L_95:.4f}", flush=True)

    return {
        "preprocessing": preprocessing, "daytype": daytype, "d": dim,
        "n_pooled": len(pooled),
        "mean_std": mean_std, "std_std": std_std, "kurt_pooled": kurt_pooled,
        "alpha_hat": res.alpha_hat,
        "alpha_L_95": res.alpha_L_95,
        "threshold_point": res.threshold_point,
        "threshold_L95": res.threshold_L95,
    }


def main() -> None:
    t0 = time.time()
    cfg = load_config()
    ah = cfg.data.day_anchor_hours

    # Per-day-type dim: under month-hour, the production spec is
    # weekday=2, saturday=4, sunday=3 (from cfg.embedding_dim);
    # under fourier, the elbow rule chose weekday=3, saturday=3, sunday=3.
    mh_dims = {dt: cfg.embedding_dim(dt) for dt in cfg.data.daytypes}
    fr_dims = {"weekday": 3, "saturday": 3, "sunday": 3}

    print(f"month-hour dims: {mh_dims}")
    print(f"fourier dims:    {fr_dims}")
    print(f"sampling protocol: n_anchors={N_ANCHORS}, "
          f"cap n_per_anchor={N_PER_ANCHOR_CAP}", flush=True)

    rows = []

    # === month-hour preprocessing ===
    print(f"\n=== preprocessing: month-hour ===")
    z_mh, df_mh = _build_zseries("month-hour")
    for dt in cfg.data.daytypes:
        rows.append(_run_one_condition(
            "month-hour", dt, mh_dims[dt],
            N_ANCHORS, N_PER_ANCHOR_CAP, z_mh, df_mh, ah, t0,
        ))

    # === fourier preprocessing ===
    print(f"\n=== preprocessing: fourier ===")
    z_fr, df_fr = _build_zseries("fourier")
    for dt in cfg.data.daytypes:
        rows.append(_run_one_condition(
            "fourier", dt, fr_dims[dt],
            N_ANCHORS, N_PER_ANCHOR_CAP, z_fr, df_fr, ah, t0,
        ))

    # === Report ===
    print()
    print("=" * 92)
    print("Patra-Sen on Ontario library residuals (production-path; "
          "pooled across anchors)")
    print("=" * 92)
    df_r = pd.DataFrame(rows)
    print(df_r[[
        "preprocessing", "daytype", "d", "n_pooled",
        "mean_std", "std_std", "kurt_pooled",
        "alpha_hat", "alpha_L_95"
    ]].to_string(index=False))
    print()

    # Cross-preprocessing comparison
    print("=" * 92)
    print("Cross-preprocessing comparison:")
    print("=" * 92)
    for dt in cfg.data.daytypes:
        rmh = next((r for r in rows
                    if r["preprocessing"] == "month-hour"
                    and r["daytype"] == dt), None)
        rfr = next((r for r in rows
                    if r["preprocessing"] == "fourier"
                    and r["daytype"] == dt), None)
        if rmh and rfr:
            print(f"  {dt:>10s}:  α̂(mh)={rmh['alpha_hat']:.4f} "
                  f"[L95={rmh['alpha_L_95']:.4f}]  |  "
                  f"α̂(fr)={rfr['alpha_hat']:.4f} "
                  f"[L95={rfr['alpha_L_95']:.4f}]")

    print()
    print("Reading:")
    print("  α̂_L^95% > 0 in BOTH preprocessings → contamination is intrinsic")
    print("    to κ_Q (preprocessing-independent), at least at the level α̂_L^95%.")
    print("  α̂_L^95% > 0 in only ONE preprocessing → contamination is")
    print("    preprocessing-induced; the other preprocessing handles it.")
    print("  α̂_L^95% = 0 in BOTH → no contamination detected at the 95% bar;")
    print("    residuals consistent with N(0,1) at homogeneity test.")
    print()

    out = Path("/tmp/patra_sen_ontario_results.pkl")
    with out.open("wb") as f:
        pickle.dump({"rows": rows, "df": df_r}, f)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
