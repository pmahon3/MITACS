"""Anchor-RNG-seed stability check on Patra-Sen Ontario result.

The apply_patra_sen_ontario.py result used a fixed RNG_SEED=7 for
selecting which 30 anchors get pooled. This check tests whether the
result is sensitive to that seed.

Findings (Ontario weekday Fourier, 5 anchor seeds):
  seed=7:   pooled kurt=22.01  α̂_L^95% = 0.5117
  seed=17:  pooled kurt=59.63  α̂_L^95% = 0.5117
  seed=27:  pooled kurt=42.29  α̂_L^95% = 0.5200
  seed=37:  pooled kurt=21.01  α̂_L^95% = 0.5083
  seed=47:  pooled kurt=22.53  α̂_L^95% = 0.5100

  L95 range: [0.508, 0.520]  spread = 0.012

Sample kurt varies wildly (21-60) across anchor selections, but
α̂_L^95% is rock-stable at 0.51 — the same CDF-based-vs-moment-based
robustness property the sample-kurt-matched check found.

PROVENANCE-GRADE: INSPECTION-ONLY (anchor stability check).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm

from edynamics.modelling_tools import Embedding, Lag
from config import load_config
from experiment._actuals import load_actuals
from experiment.predict import _daytype as _dt
from processing.innovations.estimator import local_drift_and_diffusion
from scratch.fourier_climatology import fit_fourier_params, fourier_transform
from scratch.benchmark_2021_22_ieso import CUTOFF
from scratch.patra_sen import patra_sen

N_ANCHORS = 30
N_PER_ANCHOR_CAP = 2000


def _intraday_mask(fl, dayid, d):
    keep = []
    for s in fl:
        win = pd.date_range(s - pd.Timedelta(hours=d-1),
                            s + pd.Timedelta(hours=1), freq="h")
        try:
            keep.append(dayid.loc[win].nunique() == 1)
        except KeyError:
            keep.append(False)
    return np.array(keep)


def main():
    cfg = load_config()
    ah = cfg.data.day_anchor_hours

    act = load_actuals(cutoff=None).asfreq("h")
    fp = fit_fourier_params(act, CUTOFF)
    z = fourier_transform(act, fp)
    df = z.to_frame("zscore")
    df["daytype"] = [_dt(t, ah) for t in df.index]

    d = 3
    dt = "weekday"
    print(f"Anchor-RNG-seed stability check: Ontario {dt} Fourier, d={d}")
    print(f"{'rng_seed':>10} {'n_pooled':>10} {'kurt':>8s} "
          f"{'alpha_hat':>10} {'L95':>8}")

    lags = [Lag(variable_name="zscore", tau=-i) for i in range(d)]
    fl_full = df.index[df["daytype"] == dt][d:-1]
    fl_full = fl_full[fl_full <= CUTOFF]
    dayid = pd.Series((df.index.hour == ah).cumsum(), index=df.index)
    fl_intra = fl_full[_intraday_mask(fl_full, dayid, d)]
    emb = Embedding(data=df, observers=lags, library_times=fl_intra)
    emb.compile()

    ls = []
    for rng_seed in [7, 17, 27, 37, 47]:
        rng = np.random.default_rng(rng_seed)
        anchors = fl_intra[
            np.sort(rng.choice(len(fl_intra), N_ANCHORS, replace=False))
        ]
        pooled = []
        for a in anchors:
            try:
                _C, S, _mu, _theta, resid = local_drift_and_diffusion(
                    embedding=emb, anchor=a,
                )
            except Exception:
                continue
            r0 = resid[:, 0]
            s00 = float(S[0, 0])
            if s00 <= 0:
                continue
            z_std = r0 / np.sqrt(s00)
            z_std = z_std[np.isfinite(z_std)]
            if len(z_std) > N_PER_ANCHOR_CAP:
                z_std = z_std[rng.choice(
                    len(z_std), N_PER_ANCHOR_CAP, replace=False
                )]
            pooled.append(z_std)
        pooled = np.concatenate(pooled)
        kurt = float(pd.Series(pooled).kurtosis())
        res = patra_sen(pooled, F_b=lambda x: norm.cdf(x),
                        gamma_grid_size=600)
        print(f"{rng_seed:>10} {len(pooled):>10} {kurt:>8.2f} "
              f"{res.alpha_hat:>10.4f} {res.alpha_L_95:>8.4f}",
              flush=True)
        ls.append(res.alpha_L_95)

    print()
    print(f"L95 range: [{min(ls):.4f}, {max(ls):.4f}]  "
          f"spread = {max(ls)-min(ls):.4f}")
    print()
    if max(ls) - min(ls) < 0.05:
        print("STABLE: L95 spread < 0.05 across anchor seeds. "
              "Patra-Sen result is robust to anchor selection.")


if __name__ == "__main__":
    main()
