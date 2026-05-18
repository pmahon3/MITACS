"""Frozen forecast generator -- the registered predictor.

Given the verified frozen spec and a set of target hours (hours IESO
issued a day-ahead forecast for), emit OUR one-step forecast for each,
using the production estimator core. Integrity:

  * refuses to run unless ``freeze.load_verified()`` passes (the model
    cannot be silently changed after forecasts are issued);
  * the neighbour/library pool for every fit is strictly pre-cutoff
    (the leakage guard) -- post-cutoff data only ever enters as the
    *query state* (the recent actual history you are allowed to know
    when predicting the next hour), never the fit;
  * z-score parameters are derived from pre-cutoff actuals only and a
    fingerprint of them is recorded with every forecast (post-hoc drift
    is detectable even though the values are not stored in the spec);
  * the fit uses the SAME ``_local_fit_at`` the production estimator
    uses -- not a parallel reimplementation.

One-step semantics: our forecast for target hour ``t`` is the one-step
push ``x(t-1) @ C`` where ``C`` is the local drift fitted from the
pre-cutoff library near ``x(t-1)``. ``x(t-1)`` is the embedded vector of
recent *actual* z-scores ``[z(t-1), z(t-2), ..., z(t-d)]``. Predictive
interval from the local diffusion ``Sigma`` -- a GAUSSIAN proxy of a
verified strongly heavy-tailed innovation, so interval coverage is
miscalibrated; emitted but explicitly flagged (see memory
``mitacs-rebaseline-facts``).

Usage::

    python -m experiment.predict --targets 2026-02-15      # one day
    python -m experiment.predict --from-forecast-csv       # all archived
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from edynamics.modelling_tools import Embedding, Lag

from config import load_config
from processing.innovations.estimator import _local_fit_at

from . import freeze
from ._actuals import (
    load_actuals,
    zscore_params,
    zscore_params_fingerprint,
    zscore_transform,
)

_DAYTYPE_CATS = ["weekday", "saturday", "sunday"]


def _daytype(ts: pd.Timestamp, anchor_hours: int) -> str:
    """Day-type of a timestamp under the configured day-anchor offset
    (mirrors processing/clustering/process.py)."""
    shifted = ts - pd.Timedelta(hours=anchor_hours)
    dow = shifted.dayofweek
    return "weekday" if dow < 5 else ("saturday" if dow == 5 else "sunday")


def _build_pre_cutoff(zser: pd.Series, d: int, cutoff: pd.Timestamp):
    """Embedding over pre-cutoff z-scores with d lags; returns the
    (X, Y) library arrays + the embedding for query construction."""
    df = zser.to_frame("zscore").asfreq("h")
    lags = [Lag(variable_name="zscore", tau=-i) for i in range(d)]
    lib = df.index[(df.index <= cutoff)][d:-1]
    emb = Embedding(data=df, observers=lags, library_times=lib)
    emb.compile()
    blk = emb.block
    return blk.iloc[:-1].values, blk.iloc[1:].values, emb


def generate(targets: pd.DatetimeIndex) -> pd.DataFrame:
    """Emit the registered one-step forecast for each target hour."""
    spec = freeze.load_verified()  # refuses on hash mismatch
    cutoff = pd.Timestamp(spec["data_cutoff"])
    anchor_h = spec["predictor"]["day_anchor_hour"]
    dims = spec["predictor"]["embedding_dims"]
    zp = zscore_params(cutoff)
    zfp = zscore_params_fingerprint(cutoff)

    # Full actuals: pre-cutoff rows form the fit library; post-cutoff
    # rows ONLY ever serve as query states (recent history), never a fit.
    # Raw demand is z-scored solely with the frozen pre-cutoff climatology.
    raw_full = load_actuals(cutoff=None)
    z_full = zscore_transform(raw_full, zp)

    # cache pre-cutoff library per embedding dimension (2 or 4)
    lib_cache: dict[int, tuple] = {}

    rows = []
    for t in pd.DatetimeIndex(targets):
        dt = _daytype(t, anchor_h)
        d = int(dims[dt])
        if d not in lib_cache:
            lib_cache[d] = _build_pre_cutoff(z_full[z_full.index <= cutoff], d, cutoff)
        X, Y, emb = lib_cache[d]

        # query state = embedded vector at t-1h: [z(t-1),...,z(t-d)]
        anchor_t = t - pd.Timedelta(hours=1)
        lag_times = [anchor_t - pd.Timedelta(hours=i) for i in range(d)]
        if not all(lt in z_full.index for lt in lag_times):
            continue  # insufficient recent history for this target
        x_query = z_full.reindex(lag_times).to_numpy()
        if not np.all(np.isfinite(x_query)):
            continue

        C, Sigma, mu, theta, _ = _local_fit_at(X, Y, x_query, d)
        z_next = float(x_query @ C[:, 0]) if C.ndim == 2 else float(x_query @ C)
        # de-z-score back to MW using the frozen climatology at hour t
        key = (t.month, t.hour)
        mu_mh = float(zp["mu_mh"].loc[key])
        sd_mh = float(zp["sigma_mh"].loc[key])
        fc_mw = z_next * sd_mh + mu_mh
        sd_z = float(np.sqrt(max(Sigma[0, 0], 0.0)))
        # 1-sigma Gaussian-proxy band (miscalibrated under heavy tails)
        lo_mw = (z_next - sd_z) * sd_mh + mu_mh
        hi_mw = (z_next + sd_z) * sd_mh + mu_mh

        rows.append(
            {
                "target_dt": t,
                "our_forecast_mw": fc_mw,
                "our_pi_lo_mw": lo_mw,
                "our_pi_hi_mw": hi_mw,
                "daytype": dt,
                "embedding_dim": d,
                "theta_star": theta,
                "spec_hash": spec["spec_hash"],
                "zscore_params_fp": zfp,
                "interval_caveat": "1sigma_gaussian_proxy_heavy_tailed_miscalibrated",
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", help="comma-separated dates YYYY-MM-DD (24h each)")
    ap.add_argument(
        "--from-forecast-csv",
        action="store_true",
        help="generate for every target hour present in the archived IESO forecast CSV",
    )
    args = ap.parse_args()

    cfg = load_config()
    if args.from_forecast_csv:
        fc = pd.read_csv(cfg.paths.forecast_csv, parse_dates=["datetime"])
        tgt = pd.DatetimeIndex(fc["datetime"])
    elif args.targets:
        tgt = pd.DatetimeIndex(
            [
                d + pd.Timedelta(hours=h)
                for d in pd.to_datetime(args.targets.split(","))
                for h in range(24)
            ]
        )
    else:
        ap.error("give --targets or --from-forecast-csv")

    out = generate(tgt)
    print(f"generated {len(out)} registered forecasts")
    if len(out):
        print(out[["target_dt", "our_forecast_mw", "our_pi_lo_mw", "our_pi_hi_mw",
                    "daytype"]].head(6).to_string(index=False))
