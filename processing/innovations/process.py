"""Stage 6 — local Gaussian predictive-semigroup (Pi_Delta) estimator.

For each day type, builds the lag embedding, samples anchor times, fits the
per-anchor local drift ``C_j`` and raw-residual diffusion ``Sigma_j`` via
``estimator.build_local_gaussian_semigroup``, derives the aggregated
diffusion-spectrum diagnostic, and persists everything through
``interface.save_all``.

Operator naming/novelty defers to the Resolvent_Framework programme; this
module makes no such claims. Diffusion uses the raw-residual estimator
because the library's weighted-residual covariance is ~``w^2*Q`` (verified
by the VAR(1) validation gate).

Run standalone::

    python -m processing.innovations.process
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from edynamics.modelling_tools import Embedding, Lag

from config import PipelineConfig, load_config

from .estimator import build_local_gaussian_semigroup
from .interface import save_all
from .spectral import diffusion_spectrum

logger = logging.getLogger(__name__)


def main(cfg: PipelineConfig, daytype: str | None = None) -> None:
    daytypes = (daytype,) if daytype else cfg.data.daytypes

    df = pd.read_csv(cfg.paths.clustered_csv, index_col=0, parse_dates=True).asfreq("h")
    out_dir = cfg.paths.operator_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    theta_grid = np.linspace(
        cfg.theta.theta_min, cfg.theta.theta_max, cfg.theta.theta_count
    )
    sigma_grid = np.linspace(
        cfg.theta.sigma_min, cfg.theta.sigma_max, cfg.theta.sigma_count
    )

    for dt in daytypes:
        d = cfg.embedding_dim(dt)

        lags = [
            Lag(variable_name=cfg.data.variable_name, tau=-i) for i in range(d)
        ]
        # library: this day-type's times, skipping the first d (no embedding
        # before d samples) and the last (no t+1 target).
        full_lib = df.index[df["daytype"] == dt][d:-1]
        embedding = Embedding(data=df, observers=lags, library_times=full_lib)
        embedding.compile()

        rng = np.random.default_rng(cfg.sampling.random_seed)
        n_anchor = max(1, int(np.ceil(cfg.sampling.sample_frac * len(full_lib))))
        anchors = pd.DatetimeIndex(
            np.sort(rng.choice(full_lib, size=n_anchor, replace=False))
        )

        logger.info(
            "[%s] d=%d, library=%d, anchors=%d", dt, d, len(full_lib), len(anchors)
        )

        estimate = build_local_gaussian_semigroup(
            embedding=embedding,
            anchors=anchors,
            theta_grid=theta_grid,
            sigma_grid=sigma_grid,
            gl_penalty_C=cfg.theta.gl_penalty_C,
        )
        spectrum = diffusion_spectrum(estimate.eigvals)

        save_all(
            estimate=estimate,
            spectrum=spectrum,
            output_dir=out_dir,
            run_tag=dt,
        )
        logger.info(
            "[%s] saved -> %s  (r_hat=%d, energy@1=%.3f)",
            dt,
            out_dir,
            spectrum.r_hat,
            float(spectrum.energy_fraction[0]),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(load_config())
