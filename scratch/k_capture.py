"""Option K — Step 1 of 2: capture per-day per-step (C_j, Sigma_j, d, x_state)
sequences from the augmented-scope forward iteration. Step 2 of 2
(scratch/k_loo_score.py) consumes this to do the LOO + MC scoring.

Why a separate capture script: the existing augmented-scope cache
(/tmp/bench2122_fc_propagated_augmented.pkl) stores per-row
{z_pred, s2_k, horizon_h, delivery_date} — the propagated outputs —
but NOT the per-step (C_step, Sigma_step) sequences needed for
Monte-Carlo composition under non-Gaussian step innovations.
Re-running the forward loop ONCE with sequence capture is cleaner
than threading capture into run_error_decomposition.py and risking
disturbing its decided verdict.

Reuses (does not reimplement):
  - regenerate()'s exact forward-iteration structure (augmented-scope
    branch), via in-process copy below;
  - production _local_fit_at for the WLS C_j, Sigma_j fits;
  - augmented_state_transition for the gate-validated d_max=4
    uniform (J_aug, Sigma_aug) — but we capture the RAW (C, Sigma, d)
    PER STEP so the MC inner loop can choose to use them directly or
    via the augmented Jacobian.

Output: /tmp/k_step_sequences.pkl, a dict keyed by delivery_date
(pd.Timestamp at 00:00) with value = list of 24 per-step records:
  {
    "h":          int 1..24,
    "dt":         pd.Timestamp (the forecast target hour),
    "daytype":    str (weekday/saturday/sunday),
    "d":          int (embedding dim at this step's day-type),
    "C":          np.ndarray (d, d) — production WLS local-linear drift
    "Sigma":      np.ndarray (d, d) — production WLS local residual cov
    "Sigma_00":   float — innovation variance (the only stochastic coord)
    "x_state":    np.ndarray (d,) — lag-state input to this step
    "z_pred":     float — deterministic point forecast (cache parity)
    "z_actual":   float — realised z at this hour
    "J_aug":      np.ndarray (D_MAX, D_MAX) — augmented Jacobian (=4)
    "Sigma_aug":  np.ndarray (D_MAX, D_MAX) — augmented innovation cov
  }

D_MAX is the same constant the augmented cache used (=4 by elbow).

PROVENANCE-GRADE: INSPECTION-ONLY. Pure capture; no estimator change.
"""
from __future__ import annotations

import pickle
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
from scratch.multistep_variance_propagation import augmented_state_transition

OUT_PATH = Path("/tmp/k_step_sequences.pkl")


def _say(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def main() -> None:
    t0 = time.time()
    _say("loading actuals + zscore params", t0)
    act = load_actuals(cutoff=None)
    zp = zscore_params(CUTOFF)
    z_full = zscore_transform(act, zp)
    dims = _refrozen_dims(z_full)
    D_MAX = max(dims.values())
    df = z_full.to_frame("zscore").asfreq("h")
    _say(f"dims = {dims}, D_MAX = {D_MAX}", t0)

    # Library cache identical to regenerate()
    lib_cache: dict[int, tuple] = {}

    days = pd.date_range(WIN_START, WIN_END, freq="D")
    _say(f"forward-iterating {len(days)} delivery days (capturing per-step "
         f"sequences)", t0)

    captures: dict[pd.Timestamp, list[dict]] = {}
    n_done = 0
    n_skipped_short = 0

    for D in days:
        targets = [D + pd.Timedelta(hours=h) for h in range(24)]
        issue_anchor = targets[0] - pd.Timedelta(hours=1)
        dmax = max(dims.values())
        need = [issue_anchor - pd.Timedelta(hours=i) for i in range(dmax)]
        if not all(n in z_full.index for n in need):
            n_skipped_short += 1
            continue

        zhist = {
            ts: float(z_full.loc[ts])
            for ts in z_full.index
            if issue_anchor - pd.Timedelta(hours=dmax) <= ts <= issue_anchor
        }
        day_rows: list[dict] = []
        for h_idx, t in enumerate(targets):
            dt_name = _daytype(t, ANCHOR_H)
            d = int(dims[dt_name])
            if d not in lib_cache:
                lags = [Lag(variable_name="zscore", tau=-i)
                        for i in range(d)]
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

            C, S, _mu, _th, _ = _local_fit_at(X, Y, xq, d)
            z_next = float(xq @ (C[:, 0] if C.ndim == 2 else C))
            zhist[t] = z_next

            # Augmented (J, Sigma) -- d_max=4 uniform, gate-validated.
            J_aug, Sig_aug = augmented_state_transition(
                C, float(S[0, 0]), d, D_MAX
            )

            # Actual z (may be NaN at the future-of-history boundary)
            z_act = float(z_full.loc[t]) if t in z_full.index else np.nan

            day_rows.append({
                "h": h_idx + 1,
                "dt": t,
                "daytype": dt_name,
                "d": d,
                "C": C.copy(),
                "Sigma": S.copy(),
                "Sigma_00": float(S[0, 0]),
                "x_state": xq.copy(),
                "z_pred": z_next,
                "z_actual": z_act,
                "J_aug": J_aug.copy(),
                "Sigma_aug": Sig_aug.copy(),
            })

        if len(day_rows) == 24:
            captures[D] = day_rows
            n_done += 1
            if n_done % 20 == 0:
                _say(f"  {n_done} days captured", t0)

    _say(f"complete: {n_done} days captured, {n_skipped_short} skipped "
         f"(insufficient history)", t0)
    _say(f"writing to {OUT_PATH}", t0)
    with OUT_PATH.open("wb") as f:
        pickle.dump({
            "captures": captures,
            "D_MAX": int(D_MAX),
            "dims": dims,
            "anchor_h": int(ANCHOR_H),
            "win_start": str(WIN_START),
            "win_end": str(WIN_END),
            "cutoff": str(CUTOFF),
        }, f)
    _say(f"done. total bytes: {OUT_PATH.stat().st_size:,}", t0)


if __name__ == "__main__":
    main()
