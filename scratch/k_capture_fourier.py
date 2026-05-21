"""K capture under Fourier preprocessing (A path).

Mirror of scratch/k_capture.py with the (month, hour) deseasonalisation
replaced by Fourier (K_year=8, K_day=8, no year-trend term).
Output: /tmp/k_step_sequences_fourier.pkl. Same schema as
/tmp/k_step_sequences.pkl so the existing scratch/k_loo_score.py +
scratch/k_summarise.py work against either cache.

Note on embedding dims: _refrozen_dims runs the elbow rule on
1-step rho vs d using the Fourier-preprocessed z. If Fourier gives
different dims, that's a finding (and we use them). Same convention
as the month-hour run — each preprocessing on its own terms.

PROVENANCE-GRADE: INSPECTION-ONLY.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from edynamics.modelling_tools import Embedding, Lag

from experiment._actuals import load_actuals
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
from scratch.fourier_climatology import fit_fourier_params, fourier_transform

OUT_PATH = Path("/tmp/k_step_sequences_fourier.pkl")


def _say(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def main() -> None:
    t0 = time.time()
    _say("loading actuals", t0)
    act = load_actuals(cutoff=None).asfreq("h")

    _say("fitting Fourier climatology (K_year=8, K_day=8, no year term)", t0)
    fparams = fit_fourier_params(act, CUTOFF)
    z_full = fourier_transform(act, fparams)

    _say("running elbow rule on Fourier-preprocessed z", t0)
    dims = _refrozen_dims(z_full)
    D_MAX = max(dims.values())
    df = z_full.to_frame("zscore").asfreq("h")
    _say(f"Fourier dims = {dims}, D_MAX = {D_MAX}", t0)
    _say(f"(compare month-hour dims: weekday=2, saturday=4, sunday=3, D_MAX=4)",
         t0)

    lib_cache: dict[int, tuple] = {}
    days = pd.date_range(WIN_START, WIN_END, freq="D")
    _say(f"forward-iterating {len(days)} delivery days", t0)

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

            J_aug, Sig_aug = augmented_state_transition(
                C, float(S[0, 0]), d, D_MAX
            )
            z_act = float(z_full.loc[t]) if t in z_full.index else np.nan

            day_rows.append({
                "h": h_idx + 1, "dt": t, "daytype": dt_name, "d": d,
                "C": C.copy(), "Sigma": S.copy(),
                "Sigma_00": float(S[0, 0]),
                "x_state": xq.copy(),
                "z_pred": z_next, "z_actual": z_act,
                "J_aug": J_aug.copy(), "Sigma_aug": Sig_aug.copy(),
            })

        if len(day_rows) == 24:
            captures[D] = day_rows
            n_done += 1
            if n_done % 20 == 0:
                _say(f"  {n_done} days captured", t0)

    _say(f"complete: {n_done} days captured, {n_skipped_short} skipped", t0)
    with OUT_PATH.open("wb") as f:
        pickle.dump({
            "captures": captures,
            "D_MAX": int(D_MAX),
            "dims": dims,
            "anchor_h": int(ANCHOR_H),
            "win_start": str(WIN_START),
            "win_end": str(WIN_END),
            "cutoff": str(CUTOFF),
            "preprocessing": "fourier",
            "fourier_k_year": fparams.k_year,
            "fourier_k_day": fparams.k_day,
        }, f)
    _say(f"saved {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes)", t0)


if __name__ == "__main__":
    main()
