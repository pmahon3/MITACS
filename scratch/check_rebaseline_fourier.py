"""Check whether the production rebaseline's kurt 24-33 finding holds
under Fourier preprocessing (vs the existing month-hour-based finding).

Mirrors processing/innovations/validation/rebaseline.py's exact
machinery (per-anchor library residuals via local_drift_and_diffusion)
but on the Fourier-preprocessed z-series rather than the month-hour
one.

Settles a question the compare_mh_vs_fourier.py result surfaced: the
h=1 forecast-cache residual under Fourier is near-Gaussian (kurt
0.14-0.23), while under month-hour-excl-1st it was 0.98. Two
readings:
  Reading A: per-anchor library residuals are still heavy-tailed
             (kurt 24-33), forecast-cache h=1 residual was always
             Gaussian-ish on routine days; the rebaseline kurt and
             the forecast-cache kurt measure DIFFERENT statistics
             and both readings can coexist.
  Reading B: Fourier preprocessing also changed the per-anchor
             library kurt, so the rebaseline's '24-33' finding was
             preprocessing-dependent.

This script discriminates by computing per-anchor library kurt under
Fourier and comparing to the production rebaseline number (32.6 / 24.1 /
27.2 for weekday / saturday / sunday under month-hour).

PROVENANCE-GRADE: INSPECTION-ONLY. Uses production
local_drift_and_diffusion (no reimplementation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from edynamics.modelling_tools import Embedding, Lag
from config import load_config
from experiment._actuals import load_actuals
from processing.innovations.estimator import local_drift_and_diffusion
from scratch.fourier_climatology import fit_fourier_params, fourier_transform
from scratch.benchmark_2021_22_ieso import CUTOFF


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


def _excess_kurt(r):
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 4:
        return float("nan")
    r = r - r.mean()
    s = r.std()
    if s <= 0:
        return float("nan")
    return float(np.mean(r ** 4) / (s ** 4) - 3.0)


def _daytype_of(t, anchor_h):
    """Mirror experiment.predict._daytype convention."""
    from experiment.predict import _daytype
    return _daytype(t, anchor_h)


def main(n_anchors: int = 60, seed: int = 7) -> None:
    cfg = load_config()
    ah = cfg.data.day_anchor_hours

    # Build the Fourier-preprocessed z series + a tagged 'daytype' column,
    # mirroring the structure of cfg.paths.clustered_csv but in-memory.
    act = load_actuals(cutoff=None).asfreq("h")
    fparams = fit_fourier_params(act, CUTOFF)
    z_fourier = fourier_transform(act, fparams)
    df = z_fourier.to_frame("zscore")
    df["daytype"] = [_daytype_of(t, ah) for t in df.index]
    df = df.asfreq("h")

    dayid = pd.Series((df.index.hour == ah).cumsum(), index=df.index)

    # Use the embedding dims that the Fourier elbow-rule chose
    # (uniform d=3 per the k_capture_fourier setup log).
    fourier_dims = {"weekday": 3, "saturday": 3, "sunday": 3}

    print(f"Rebaseline kurt under Fourier preprocessing")
    print(f"  K_year=8, K_day=8; Fourier dims = {fourier_dims}")
    print(f"  Reference (month-hour rebaseline, check_rebaseline_seam.py):")
    print(f"    weekday d=2 kurt=32.56;  saturday d=2 kurt=24.09;  sunday d=4 kurt=27.21")
    print()
    print(f"{'daytype':>10s} {'d':>2s} {'n_a':>4s} {'kurt_all~':>10s} "
          f"{'kurt_seam_excl~':>16s} {'Δ vs month-hour':>16s}")
    print("-" * 80)

    mh_reference = {"weekday": 32.56, "saturday": 24.09, "sunday": 27.21}

    for dt in cfg.data.daytypes:
        d = fourier_dims[dt]
        lags = [Lag(variable_name="zscore", tau=-i) for i in range(d)]
        fl_full = df.index[df["daytype"] == dt][d:-1]
        fl_intra = fl_full[_intraday_mask(fl_full, dayid, d)]

        emb_intra = Embedding(data=df, observers=lags,
                              library_times=fl_intra)
        emb_intra.compile()

        rng = np.random.default_rng(seed)
        anchors = fl_intra[
            np.sort(rng.choice(len(fl_intra),
                               min(n_anchors, len(fl_intra)),
                               replace=False))
        ]

        kurts_all, kurts_excl = [], []
        for a in anchors:
            block = emb_intra.block
            blk = block.loc[block.index != a]
            X_df = blk.iloc[:-1]
            Y_df = blk.iloc[1:]
            keep = Y_df.index.hour != ah
            Y_kept = Y_df[keep]
            target_times = Y_kept.index
            _C, _S, _mu, _theta, resid = local_drift_and_diffusion(
                embedding=emb_intra, anchor=a,
            )
            r0 = resid[:, 0]
            if len(r0) != len(target_times):
                continue
            is_seam = ((target_times.day == 1)
                       & (target_times.hour == 0))
            kurts_all.append(_excess_kurt(r0))
            kurts_excl.append(_excess_kurt(r0[~is_seam]))

        if not kurts_all:
            continue
        med_all = float(np.median(kurts_all))
        med_excl = float(np.median(kurts_excl))
        delta = med_all - mh_reference.get(dt, float("nan"))
        print(f"{dt:>10s} {d:>2d} {len(kurts_all):>4d} "
              f"{med_all:>10.3f} {med_excl:>16.3f} "
              f"{delta:>+16.3f}")

    print()
    print("Reading (the load-bearing discrimination):")
    print("  - If Fourier rebaseline kurt is similar to month-hour (~24-33):")
    print("    Reading A holds. The conditional law is intrinsically heavy-")
    print("    tailed. Forecast-cache h=1 residual being near-Gaussian is a")
    print("    separate property of the propagated-forecast object.")
    print("  - If Fourier rebaseline kurt is materially lower (~0-5):")
    print("    Reading B holds. The 'kurt 24-33' finding was preprocessing-")
    print("    dependent; Fourier produces a near-Gaussian conditional law,")
    print("    and the qualifier-exhaustion programme's premise needs reframing.")


if __name__ == "__main__":
    main()
