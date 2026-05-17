"""Stage 4 — estimate embedding dimension per day type.

For each day type, sweeps embedding dimension via the edynamics
``dimensionality`` estimator and writes ``results_<daytype>.csv`` (whose
``idxmax`` gives the embedding dimension used downstream) plus a plot.

Run standalone::

    python -m processing.dimensions.process
"""
from __future__ import annotations

import multiprocessing

import matplotlib

# Headless-safe: the orchestration runner has no display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
from ray.util.multiprocessing import Pool

from edynamics.modelling_tools import Embedding, Lag
from edynamics.modelling_tools.estimators import dimensionality

from config import PipelineConfig, load_config


def main(cfg: PipelineConfig) -> None:
    params_dir = cfg.paths.dimensions_dir
    plots_dir = cfg.paths.dimensions_plots
    params_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not ray.is_initialized():
        ray.init(log_to_driver=False)
    pool = Pool(multiprocessing.cpu_count())

    # ── Load the day-type-tagged data (output of stage 3) ────────────────
    df = pd.read_csv(cfg.paths.clustered_csv, index_col=0)
    df.index = pd.DatetimeIndex(df.index)
    df = df.asfreq("h")

    for daytype in cfg.data.daytypes:
        library_indices = df.index[df["daytype"] == daytype][10:]
        sampled = np.random.choice(
            library_indices[:-1],
            size=int(0.2 * len(library_indices)),
            replace=False,
        )
        prediction_times = pd.DatetimeIndex(sampled)
        prediction_times = prediction_times[
            prediction_times.hour != (cfg.data.day_anchor_hours + 12) % 24
        ]

        embedding = Embedding(
            data=df,
            observers=[Lag(variable_name=cfg.data.variable_name, tau=0)],
            library_times=library_indices,
        )

        result = dimensionality(
            embedding=embedding,
            target=cfg.data.variable_name,
            steps=1,
            step_size=1,
            times=prediction_times,
            max_dimensions=cfg.embedding.max_dimensions,
            compute_pool=pool,
            verbose=False,
        )

        result.to_csv(params_dir / f"results_{daytype}.csv", index=True)

        plt.plot(result)
        plt.title(f"Dimensionality for {daytype.capitalize()}")
        plt.xlabel("K Nearest Neighbors Used")
        plt.ylabel("Pearson Correlation")
        plt.savefig(plots_dir / f"dimensionality_{daytype}.png")
        plt.cla()


if __name__ == "__main__":
    main(load_config())
