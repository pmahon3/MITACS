"""Check whether the rebaseline's reported kurt 24-33 is driven by the
same month-boundary climatology seam that contaminates the forecast
cache.

Approach: mirror processing/innovations/validation/rebaseline.py's
exact structure, but for each anchor compute BOTH:
  (a) excess kurt of all library residuals at the anchor (as the
      rebaseline does)
  (b) excess kurt with library transitions whose TARGET lands on
      first-of-month dropped (seam exclusion analogous to the
      day_anchor_hour mask the production already applies for the
      day-type rollover)

Median across anchors of (a) is the rebaseline's published kurt.
Median across anchors of (b) is the seam-corrected kurt. If (b) is
materially smaller than (a), the rebaseline finding is contaminated
by the seam too, and the qualifier exhaustion programme's premise
needs reframing. If (b) ≈ (a), the rebaseline is honest on its own
object and the forecast-cache contamination is separable.

Uses the production estimator path (innovation_diagnostics calls
local_drift_and_diffusion); the seam exclusion is post-hoc on the
returned `resid` and the same day_anchor_hour mask the production
applies. PROVENANCE-GRADE: INSPECTION-ONLY.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from edynamics.modelling_tools import Embedding, Lag
from config import load_config
from processing.innovations.estimator import local_drift_and_diffusion


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


def main(n_anchors: int = 60, seed: int = 7) -> None:
    cfg = load_config()
    ah = cfg.data.day_anchor_hours

    df = pd.read_csv(
        cfg.paths.clustered_csv, index_col=0, parse_dates=True
    ).asfreq("h")
    dayid = pd.Series((df.index.hour == ah).cumsum(), index=df.index)

    print(f"{'daytype':>10s} {'d':>2s} {'n_a':>4s} {'kurt_all~':>10s} "
          f"{'kurt_seam_excl~':>16s} {'delta~':>9s} "
          f"{'frac_seam_in_lib~':>18s}")
    print("-" * 80)

    for dt in cfg.data.daytypes:
        d = cfg.embedding_dim(dt)
        lags = [Lag(variable_name=cfg.data.variable_name, tau=-i)
                for i in range(d)]
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

        kurts_all, kurts_excl, seam_fracs = [], [], []
        for a in anchors:
            # Reproduce local_drift_and_diffusion's X/Y_df construction
            # exactly so we know the target_times aligned with resid rows.
            block = emb_intra.block
            blk = block.loc[block.index != a]
            X_df = blk.iloc[:-1]
            Y_df = blk.iloc[1:]
            keep = Y_df.index.hour != ah   # production day-anchor mask
            Y_kept = Y_df[keep]
            target_times = Y_kept.index    # the actual targets after masking
            _C, _S, _mu, _theta, resid = local_drift_and_diffusion(
                embedding=emb_intra, anchor=a,
            )
            r0 = resid[:, 0]
            if len(r0) != len(target_times):
                # Defensive: log and skip if alignment fails
                continue
            # First-of-month TARGET = target.day == 1 AND target.hour == 0
            is_seam = ((target_times.day == 1)
                       & (target_times.hour == 0))
            seam_fracs.append(float(is_seam.sum()) / len(is_seam))
            kurts_all.append(_excess_kurt(r0))
            kurts_excl.append(_excess_kurt(r0[~is_seam]))

        if not kurts_all:
            continue
        med_all = float(np.median(kurts_all))
        med_excl = float(np.median(kurts_excl))
        med_frac = float(np.median(seam_fracs))
        delta = med_all - med_excl
        print(f"{dt:>10s} {d:>2d} {len(kurts_all):>4d} "
              f"{med_all:>10.3f} {med_excl:>16.3f} "
              f"{delta:>+9.3f} {med_frac*100:>17.2f}%")

    print()
    print("Reading:")
    print("  - kurt_all~       : median per-anchor excess kurt (matches rebaseline's reported number)")
    print("  - kurt_seam_excl~ : median after dropping library transitions with first-of-month TARGETS")
    print("  - delta~          : seam contribution to per-anchor kurt")
    print("  - frac_seam_in_lib~ : median fraction of seam transitions in each anchor's library")
    print()
    print("If delta~ is large and positive (kurt_all > kurt_seam_excl by")
    print("more than ~3 units): rebaseline kurt is contaminated by the seam.")
    print("If delta~ is small: rebaseline finding is robust; forecast-cache")
    print("contamination is a separable issue.")


if __name__ == "__main__":
    main()
