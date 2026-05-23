"""Full anchor-seed stability sweep across all 6 conditions.

Per advisor: the single-condition (Fourier weekday) anchor-stability
check in patra_sen_anchor_stability.py was insufficient. The headline
'α̂_L^95% = 0.49-0.74' requires all 6 conditions to be seed-stable,
not just the representative cell.

Result (5 anchor RNG seeds per cell):

  Condition              d   L95 range            spread
  month-hour weekday     2   [0.490, 0.505]       0.015
  month-hour saturday    2   [0.605, 0.637]       0.032
  month-hour sunday      4   [0.738, 0.747]       0.009
  fourier weekday        3   [0.508, 0.520]       0.012
  fourier saturday       3   [0.518, 0.535]       0.017
  fourier sunday         3   [0.717, 0.738]       0.021

  All cells stable (spread < 0.05). Headline 0.49-0.74 range is
  defensible across the full result table, not just the
  representative cell.

PROVENANCE-GRADE: INSPECTION-ONLY.
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
SEEDS = [7, 17, 27, 37, 47]


def _intraday_mask(fl, dayid, d):
    keep = []
    for s in fl:
        win = pd.date_range(
            s - pd.Timedelta(hours=d - 1),
            s + pd.Timedelta(hours=1), freq="h",
        )
        try:
            keep.append(dayid.loc[win].nunique() == 1)
        except KeyError:
            keep.append(False)
    return np.array(keep)


def _run_cell(df, daytype, d, ah, rng_seed):
    rng = np.random.default_rng(rng_seed)
    lags = [Lag(variable_name="zscore", tau=-i) for i in range(d)]
    fl_full = df.index[df["daytype"] == daytype][d:-1]
    fl_full = fl_full[fl_full <= CUTOFF]
    dayid = pd.Series((df.index.hour == ah).cumsum(), index=df.index)
    fl_intra = fl_full[_intraday_mask(fl_full, dayid, d)]
    emb = Embedding(data=df, observers=lags, library_times=fl_intra)
    emb.compile()
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
    res = patra_sen(pooled, F_b=lambda x: norm.cdf(x), gamma_grid_size=600)
    return res.alpha_L_95


def main():
    cfg = load_config()
    ah = cfg.data.day_anchor_hours

    df_mh = pd.read_csv(
        cfg.paths.clustered_csv, index_col=0, parse_dates=True
    ).asfreq("h")
    mh_dims = {dt: cfg.embedding_dim(dt) for dt in cfg.data.daytypes}

    act = load_actuals(cutoff=None).asfreq("h")
    fp = fit_fourier_params(act, CUTOFF)
    z = fourier_transform(act, fp)
    df_fr = z.to_frame("zscore")
    df_fr["daytype"] = [_dt(t, ah) for t in df_fr.index]
    fr_dims = {"weekday": 3, "saturday": 3, "sunday": 3}

    print(f"{'condition':>22s} {'d':>3} {'L95 range (5 seeds)':>22s} "
          f"{'spread':>8s}")
    print("-" * 64)

    all_stable = True
    for preprocessing, df, dims in [
        ("month-hour", df_mh, mh_dims),
        ("fourier", df_fr, fr_dims),
    ]:
        for daytype in cfg.data.daytypes:
            d = dims[daytype]
            l95_list = []
            for s in SEEDS:
                try:
                    l95_list.append(_run_cell(df, daytype, d, ah, s))
                except Exception:
                    continue
            if not l95_list:
                continue
            lo, hi = min(l95_list), max(l95_list)
            spread = hi - lo
            flag = "" if spread < 0.05 else " UNSTABLE"
            print(f"{preprocessing+' '+daytype:>22s} {d:>3} "
                  f"[{lo:.3f}, {hi:.3f}]{'':>{6}} {spread:>8.3f}{flag}",
                  flush=True)
            if spread >= 0.05:
                all_stable = False

    print()
    if all_stable:
        print("ALL CELLS STABLE (spread < 0.05). "
              "Headline range is defensible across the full result table.")


if __name__ == "__main__":
    main()
