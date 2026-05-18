"""Multi-step day-ahead predictor (Task 23, Goal 3's real deliverable).

A genuine day-ahead forecast for delivery day D: issued at the end of
D-1, forecasting all 24 hours of D using ONLY data available then. The
one-step Pi_Delta is iterated 24 times; each step's prediction feeds the
next step's lag vector (the composition validated against VAR(1)
closed-form in validation/synthetic.py -- error compounds with horizon,
quantified there and reported per-horizon by the backtest).

Integrity (same discipline as the one-step predictor):
  * frozen spec verified (hash gate);
  * fit library strictly <= frozen cutoff (leakage guard);
  * query history uses real actuals only up to issue time (D-1 23:00) --
    legitimate: you know the past when forecasting tomorrow;
  * within day D the lag vector is filled with PREDICTIONS, never peeked
    actuals (that is what makes it a real forecast, and why error grows);
  * day-type / embedding dim re-resolved PER TARGET HOUR (the 07:00
    anchor can switch regime mid-horizon -- faithful to clustering and
    the one-step predictor).

Interval note: the per-step Gaussian Sigma band is a proxy of a verified
heavy-tailed innovation AND ignores cross-step error accumulation, so
multi-step intervals are doubly approximate -- emitted but explicitly
flagged. Point forecasts are the deliverable; intervals are diagnostic.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from processing.innovations.estimator import _local_fit_at

from . import freeze
from ._actuals import load_actuals, zscore_params, zscore_params_fingerprint, zscore_transform
from .predict import _build_pre_cutoff, _daytype

ISSUE_HOUR_OFFSET = pd.Timedelta(hours=1)  # last known actual = D-1 23:00


def day_ahead(delivery_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Iterated 24h day-ahead forecast for each delivery date.

    For date D: last known actual is (D 00:00 - 1h) = D-1 23:00. Forecast
    D 00:00..23:00 by iterating the frozen one-step map, feeding each
    prediction forward.
    """
    spec = freeze.load_verified()
    cutoff = pd.Timestamp(spec["data_cutoff"])
    anchor_h = spec["predictor"]["day_anchor_hour"]
    dims = spec["predictor"]["embedding_dims"]
    zp = zscore_params(cutoff)
    zfp = zscore_params_fingerprint(cutoff)

    raw_full = load_actuals(cutoff=None)
    z_full = zscore_transform(raw_full, zp)  # z-scored with frozen climatology

    lib_cache: dict[int, tuple] = {}
    rows = []

    for D in pd.DatetimeIndex(delivery_dates):
        D = pd.Timestamp(D.date())
        targets = [D + pd.Timedelta(hours=h) for h in range(24)]
        issue_anchor = targets[0] - ISSUE_HOUR_OFFSET  # D-1 23:00

        # need d real actuals immediately before issue (max d across daytypes)
        dmax = max(int(v) for v in dims.values())
        hist_need = [issue_anchor - pd.Timedelta(hours=i) for i in range(dmax)]
        if not all(h in z_full.index for h in hist_need):
            continue
        if not np.all(np.isfinite(z_full.reindex(hist_need).to_numpy())):
            continue

        # rolling z-score history: real actuals <= issue_anchor, then we
        # append our own predictions as we step forward.
        zhist: dict[pd.Timestamp, float] = {
            ts: float(z_full.loc[ts])
            for ts in z_full.index
            if (issue_anchor - pd.Timedelta(hours=dmax)) <= ts <= issue_anchor
        }

        for t in targets:
            dt = _daytype(t, anchor_h)
            d = int(dims[dt])
            if d not in lib_cache:
                lib_cache[d] = _build_pre_cutoff(
                    z_full[z_full.index <= cutoff], d, cutoff
                )
            X, Y, _emb = lib_cache[d]

            prev = t - pd.Timedelta(hours=1)
            lag_times = [prev - pd.Timedelta(hours=i) for i in range(d)]
            # each lag is a real actual (if <= issue_anchor) or a prior
            # prediction (if within day D) -- never a peeked actual in D.
            try:
                x_query = np.array([zhist[lt] for lt in lag_times], dtype=float)
            except KeyError:
                break  # gap; abandon this delivery day
            if not np.all(np.isfinite(x_query)):
                break

            C, Sigma, _mu, theta, _ = _local_fit_at(X, Y, x_query, d)
            z_next = float(x_query @ (C[:, 0] if C.ndim == 2 else C))
            zhist[t] = z_next  # feed forward (NOT the actual)

            key = (t.month, t.hour)
            mu_mh = float(zp["mu_mh"].loc[key])
            sd_mh = float(zp["sigma_mh"].loc[key])
            sd_z = float(np.sqrt(max(Sigma[0, 0], 0.0)))
            rows.append(
                {
                    "delivery_date": D,
                    "target_dt": t,
                    "horizon_h": int((t - targets[0]) / pd.Timedelta(hours=1)) + 1,
                    "our_forecast_mw": z_next * sd_mh + mu_mh,
                    "our_pi_lo_mw": (z_next - sd_z) * sd_mh + mu_mh,
                    "our_pi_hi_mw": (z_next + sd_z) * sd_mh + mu_mh,
                    "daytype": dt,
                    "embedding_dim": d,
                    "theta_star": theta,
                    "spec_hash": spec["spec_hash"],
                    "zscore_params_fp": zfp,
                    "issue_anchor": issue_anchor,
                    "interval_caveat": (
                        "per-step Gaussian proxy; heavy-tailed innovation + "
                        "no cross-step error accumulation -> doubly approximate"
                    ),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dates", required=True,
                    help="comma-separated delivery dates YYYY-MM-DD")
    args = ap.parse_args()
    out = day_ahead(pd.to_datetime(args.dates.split(",")))
    print(f"generated {len(out)} day-ahead forecast rows "
          f"({out['delivery_date'].nunique() if len(out) else 0} delivery days)")
    if len(out):
        print(
            out[["target_dt", "horizon_h", "our_forecast_mw", "daytype"]]
            .head(6)
            .to_string(index=False)
        )
