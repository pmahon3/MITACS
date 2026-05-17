"""Disciplined production-path re-baseline of the core empirical facts.

Every reported number comes from a PRODUCTION function. This script
computes nothing itself except read-only summary stats (median/std) over
production outputs, plus ``np.linalg.cond`` (a standard metric) on the
production ``Sigma_j``. This is the rule that the earlier ad-hoc
diagnostics violated -- and every confident Ontario claim from a
reimplemented diagnostic this project made was wrong or inflated. The
VAR(1) gate never misled precisely because it calls production directly;
this script does the same.

Production sources
------------------
  - ``cfg.embedding_dim``                  -- embedding dimension
  - ``build_local_gaussian_semigroup``     -- C_j, Sigma_j, theta*
  - ``innovation_diagnostics``             -- non-Gaussianity (gate-validated)
  - ``interface.save_all`` / ``load_all``  -- artifact roundtrip
  - ``np.linalg.cond`` on production Sigma  -- conditioning (only computed
                                              metric; standard)

Dropped axis: r_hat / multi-mode. ``Sigma_j`` is rank-1 BY CONSTRUCTION of
the single-variable delay embedding (memory ``mitacs-rank1-structural``);
"multi-mode" is incoherent here and reporting it would relitigate a
settled artifact.

Controls
--------
  - Fixed shared anchor set per day type (valid under BOTH estimands), so
    full-vs-intra differences are the intended conditioning-set (Q)
    difference, not sampling noise.
  - A clean VAR(1) reference row through the identical production path:
    every Ontario number has a known-Gaussian baseline beside it.

Run::

    python -m processing.innovations.validation.rebaseline
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from edynamics.modelling_tools import Embedding, Lag

from config import load_config

from ..estimator import (
    build_local_gaussian_semigroup,
    innovation_diagnostics,
)
from ..interface import load_all, save_all
from ..spectral import diffusion_spectrum
from .synthetic import build_embedding, make_var1_params, simulate_var1


@dataclass
class BaselineRow:
    label: str
    estimand: str
    d: int
    n_anchors: int
    innov_var_med: float
    excess_kurt_med: float
    tail_ratio_med: float
    sigma_cond_med: float
    library: int


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


def _row(label, estimand, emb, anchors, d, day_anchor_hour, library):
    """All numbers here come from production functions only."""
    anchors = pd.DatetimeIndex([a for a in anchors if a in emb.block.index])
    est = build_local_gaussian_semigroup(
        embedding=emb, anchors=anchors, day_anchor_hour=day_anchor_hour
    )
    innov_var = est.covariances[:, 0, 0]                       # production
    conds = np.array([np.linalg.cond(S) for S in est.covariances])  # standard
    eks, trs = [], []
    for a in anchors:
        di = innovation_diagnostics(
            embedding=emb, anchor=a, day_anchor_hour=day_anchor_hour
        )
        if np.isfinite(di["excess_kurt"]):
            eks.append(di["excess_kurt"])
            trs.append(di["tail_ratio"])
    return BaselineRow(
        label, estimand, d, len(anchors),
        float(np.median(innov_var)),
        float(np.median(eks)) if eks else float("nan"),
        float(np.median(trs)) if trs else float("nan"),
        float(np.median(conds)),
        library,
    )


def _roundtrip_ok(est, tmp) -> bool:
    """interface.save_all/load_all production contract."""
    sp = diffusion_spectrum(est.eigvals)
    save_all(estimate=est, spectrum=sp, output_dir=tmp, run_tag="rb")
    back = load_all(tmp)
    return np.allclose(back["coeffs"], est.coefficients, atol=1e-5)


def rebaseline(n_anchors: int = 60, seed: int = 7) -> list[BaselineRow]:
    cfg = load_config()
    ah = cfg.data.day_anchor_hours
    rows: list[BaselineRow] = []

    # --- clean VAR(1) reference (identical production path) ---------------
    for d in (2, 4):
        A, Q = make_var1_params(d, 7)
        X = simulate_var1(A, Q, n=5000, burn=500, seed=8)
        emb, idx = build_embedding(X)
        interior = idx[d + 1 : -2]
        rng = np.random.default_rng(seed)
        an = interior[
            np.sort(rng.choice(len(interior), min(n_anchors, len(interior)), replace=False))
        ]
        rows.append(_row(f"VAR1_clean_d{d}", "n/a", emb, an, d, None, len(idx)))

    # --- Ontario: full_process vs intra_day, fixed shared anchors --------
    df = pd.read_csv(
        cfg.paths.clustered_csv, index_col=0, parse_dates=True
    ).asfreq("h")
    dayid = pd.Series((df.index.hour == ah).cumsum(), index=df.index)

    for dt in cfg.data.daytypes:
        d = cfg.embedding_dim(dt)
        lags = [Lag(variable_name=cfg.data.variable_name, tau=-i) for i in range(d)]
        fl_full = df.index[df["daytype"] == dt][d:-1]
        fl_intra = fl_full[_intraday_mask(fl_full, dayid, d)]

        emb_full = Embedding(data=df, observers=lags, library_times=fl_full)
        emb_full.compile()
        emb_intra = Embedding(data=df, observers=lags, library_times=fl_intra)
        emb_intra.compile()

        # fixed anchors valid under BOTH estimands
        rng = np.random.default_rng(seed)
        pool = fl_intra
        anchors = pool[
            np.sort(rng.choice(len(pool), min(n_anchors, len(pool)), replace=False))
        ]

        rows.append(
            _row(dt, "full_process", emb_full, anchors, d, ah, len(fl_full))
        )
        rows.append(
            _row(dt, "intra_day", emb_intra, anchors, d, ah, len(fl_intra))
        )
    return rows


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    rows = rebaseline()
    hdr = (
        f"{'label':14s} {'estimand':12s} {'d':>2s} {'n':>3s} "
        f"{'innov_var~':>10s} {'exc_kurt~':>9s} {'tail~':>6s} "
        f"{'Sig_cond~':>10s} {'library':>8s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r.label:14s} {r.estimand:12s} {r.d:2d} {r.n_anchors:3d} "
            f"{r.innov_var_med:10.5f} {r.excess_kurt_med:9.3f} "
            f"{r.tail_ratio_med:6.3f} {r.sigma_cond_med:10.4g} "
            f"{r.library:8d}"
        )

    # production save/load contract check (one Ontario estimate)
    cfg = load_config()
    df = pd.read_csv(
        cfg.paths.clustered_csv, index_col=0, parse_dates=True
    ).asfreq("h")
    dt = cfg.data.daytypes[0]
    d = cfg.embedding_dim(dt)
    lags = [Lag(variable_name=cfg.data.variable_name, tau=-i) for i in range(d)]
    fl = df.index[df["daytype"] == dt][d:-1]
    emb = Embedding(data=df, observers=lags, library_times=fl)
    emb.compile()
    rng = np.random.default_rng(0)
    an = pd.DatetimeIndex(np.sort(rng.choice(fl, 30, replace=False)))
    est = build_local_gaussian_semigroup(
        embedding=emb, anchors=an, day_anchor_hour=cfg.data.day_anchor_hours
    )
    with tempfile.TemporaryDirectory() as tmp:
        ok = _roundtrip_ok(est, Path(tmp))
    print()
    print("interface save/load roundtrip:", "PASS" if ok else "FAIL")
