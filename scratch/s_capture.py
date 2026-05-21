"""Option S — capture h=1 residuals under two state augmentations:
  arm 'phi':    x_t = (z_t, z_{t-1}, ..., z_{t-d+1}, phi_t)
                where phi_t = tanh((z_t - z_{t-1}) / s) and s = the
                dev-set median |z_t - z_{t-1}|. Augmented dim = d+1.
  arm 'lag':    x_t = (z_t, z_{t-1}, ..., z_{t-(d+1)+1})
                pure-lag state at dim d+1 — the enrich-d side
                comparator (advisor confound control).

Both arms refit Pi_1 from scratch under the new state using
production _local_fit_at, then compute the h=1 standardised
residual on the 2021-22 dev window. Output:
  /tmp/s_resid_phi.pkl  (rows: day, daytype, z_pred, z_actual, sigma00)
  /tmp/s_resid_lag.pkl  (same schema, different d)

Scope is h=1 only. Multi-step propagation under the phi arm would
require EKF linearisation of phi_{t+1} (nonlinear in z_{t+1});
deferred (S verdict measurements are all at h=1).

Per-day-type d (baseline elbow): weekday 2, saturday 4, sunday 3.
S augmented dim:                  weekday 3, saturday 5, sunday 4.

PROVENANCE-GRADE: INSPECTION-ONLY.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from edynamics.modelling_tools import Embedding, Lag

from experiment._actuals import load_actuals, zscore_params, zscore_transform
from experiment.predict import _daytype
from processing.innovations.estimator import _local_fit_at

from scratch.benchmark_2021_22_ieso import (
    ANCHOR_H,
    CUTOFF,
    WIN_END,
    WIN_START,
    _refrozen_dims,
)


def _say(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def _build_phi_block(z: pd.Series, d: int, s_scale: float,
                     library_times: pd.DatetimeIndex):
    """Build (X, Y) for the phi-augmented embedding at dim d+1:
        x_t = (z_t, z_{t-1}, ..., z_{t-d+1}, phi_t)
        phi_t = tanh((z_t - z_{t-1}) / s_scale)
    Y is x_{t+1} for one-step fitting. Returns (X, Y) of shape
    (n, d+1) each.
    """
    df = z.to_frame("zscore").asfreq("h")
    lags = [Lag(variable_name="zscore", tau=-i) for i in range(d)]
    fl = library_times
    emb = Embedding(data=df, observers=lags, library_times=fl)
    emb.compile()
    b = emb.block.values    # (n, d)
    # phi at time t requires z_t and z_{t-1}. emb.block columns are
    # [z_t, z_{t-1}, ..., z_{t-d+1}]; phi from col 0 and col 1.
    phi_t = np.tanh((b[:, 0] - b[:, 1]) / s_scale)
    X = np.column_stack([b, phi_t])
    Y_lag = np.roll(X, -1, axis=0)
    return X[:-1], Y_lag[:-1]


def _build_lag_block(z: pd.Series, d_plus: int,
                     library_times: pd.DatetimeIndex):
    """Pure-lag block at dim d_plus."""
    df = z.to_frame("zscore").asfreq("h")
    lags = [Lag(variable_name="zscore", tau=-i) for i in range(d_plus)]
    fl = library_times
    emb = Embedding(data=df, observers=lags, library_times=fl)
    emb.compile()
    b = emb.block.values
    Y_lag = np.roll(b, -1, axis=0)
    return b[:-1], Y_lag[:-1]


def _compute_phi_t(z_hist: dict, t_anchor: pd.Timestamp, s_scale: float,
                   ) -> float:
    """phi at an anchor time, given a history dict z_hist."""
    prev = t_anchor
    prev2 = prev - pd.Timedelta(hours=1)
    if prev not in z_hist or prev2 not in z_hist:
        return float("nan")
    return float(np.tanh((z_hist[prev] - z_hist[prev2]) / s_scale))


def _run_arm(arm: str, dims_base: dict, z_full: pd.Series,
             act_index: pd.DatetimeIndex, s_scale_by_dt: dict,
             t0: float) -> list:
    """Run one S arm. arm in {'phi', 'lag'}.
    Returns list of {day, h=1 residual, daytype, sigma00, z_pred, z_actual}.
    """
    df = z_full.to_frame("zscore").asfreq("h")
    lib_cache: dict[str, tuple] = {}     # keyed by (arm, daytype)

    days = pd.date_range(WIN_START, WIN_END, freq="D")
    rows = []
    n_done = 0
    for D in days:
        target = D                # h=1 is hour 00:00 of D
        anchor = target - pd.Timedelta(hours=1)
        d_base = dims_base[_daytype(target, ANCHOR_H)]
        d_input = d_base + 1 if arm == "lag" else d_base
        d_state = d_base + 1     # both arms end at d+1

        # Need history back d_input hours for the input state
        need = [anchor - pd.Timedelta(hours=i) for i in range(d_input + 1)]
        if not all(n in z_full.index for n in need):
            continue
        zhist = {n: float(z_full.loc[n]) for n in need}

        dt_name = _daytype(target, ANCHOR_H)
        s_scale = s_scale_by_dt[dt_name]
        cache_key = (arm, dt_name, d_base if arm == "phi" else d_input)
        if cache_key not in lib_cache:
            idx = df.index[df.index <= CUTOFF]
            idx = idx[[_daytype(t, ANCHOR_H) == dt_name for t in idx]]
            if arm == "phi":
                idx = idx[d_base + 1:-1]   # need 2 lags for phi
                if len(idx) < 200:
                    continue
                X, Y = _build_phi_block(z_full, d_base, s_scale, idx)
            else:
                idx = idx[d_input:-1]
                if len(idx) < 200:
                    continue
                X, Y = _build_lag_block(z_full, d_input, idx)
            lib_cache[cache_key] = (X, Y)
        X, Y = lib_cache[cache_key]

        # Build the query state x_anchor
        if arm == "phi":
            lag_vec = [zhist[anchor - pd.Timedelta(hours=i)]
                       for i in range(d_base)]
            phi_val = float(np.tanh(
                (zhist[anchor] - zhist[anchor - pd.Timedelta(hours=1)])
                / s_scale))
            xq = np.array(lag_vec + [phi_val], dtype=float)
        else:
            lag_vec = [zhist[anchor - pd.Timedelta(hours=i)]
                       for i in range(d_input)]
            xq = np.array(lag_vec, dtype=float)

        # WLS local fit at the augmented state, refitting Pi_1
        # _local_fit_at signature: (X, Y, xq, d) — d here is state dim
        try:
            C, S, _mu, _th, _ = _local_fit_at(X, Y, xq, d_state)
        except Exception as e:
            _say(f"  fit failed at {target}: {e}", t0)
            continue

        z_next = float(xq @ (C[:, 0] if C.ndim == 2 else C))
        z_act = float(z_full.loc[target]) if target in z_full.index \
            else np.nan
        sigma00 = float(S[0, 0])
        rows.append({
            "day": D,
            "daytype": dt_name,
            "d_input": d_input,
            "d_state": d_state,
            "z_pred": z_next,
            "z_actual": z_act,
            "sigma00": sigma00,
        })
        n_done += 1
        if n_done % 20 == 0:
            _say(f"  [{arm}] {n_done} days processed", t0)

    return rows


def main() -> None:
    t0 = time.time()
    _say("loading actuals + zscore params", t0)
    act = load_actuals(cutoff=None)
    zp = zscore_params(CUTOFF)
    z_full = zscore_transform(act, zp)
    dims_base = _refrozen_dims(z_full)
    _say(f"baseline elbow dims = {dims_base}", t0)
    _say(f"S augmented dim per daytype = "
         f"{ {dt: dims_base[dt] + 1 for dt in dims_base} }", t0)

    # Scale for the tanh — dev-set median |dz| per day-type
    dz = z_full.diff()
    s_scale_by_dt = {}
    for dt in ("weekday", "saturday", "sunday"):
        mask = [_daytype(t, ANCHOR_H) == dt for t in dz.index]
        sub = dz[mask].dropna()
        s_scale_by_dt[dt] = float(sub.abs().median())
    _say(f"s_scale (median |dz|) per daytype = "
         f"{ {k: f'{v:.4f}' for k,v in s_scale_by_dt.items()} }", t0)

    for arm in ("phi", "lag"):
        _say(f"=== running arm: {arm} ===", t0)
        rows = _run_arm(arm, dims_base, z_full, act.index,
                        s_scale_by_dt, t0)
        df = pd.DataFrame(rows)
        out = Path(f"/tmp/s_resid_{arm}.pkl")
        with out.open("wb") as f:
            pickle.dump({"rows": df, "dims_base": dims_base,
                         "s_scale_by_dt": s_scale_by_dt,
                         "arm": arm}, f)
        _say(f"  saved {out} ({len(df)} rows)", t0)


if __name__ == "__main__":
    main()
