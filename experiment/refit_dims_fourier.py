"""Refit per-day-type embedding dimensions under Fourier preprocessing.

The dimensions in ``processing/dimensions/params/results_<daytype>.csv``
were computed on (month, hour) z-scored data. Under Fourier
preprocessing the rho-vs-d curve may peak at different dimensions; an
apples-to-apples backtest comparison wants each spec to use the dims
its preprocessing's stationarity surface supports.

This script writes a parallel
``processing/dimensions/params_fourier/results_<daytype>.csv`` set,
computed by the same production ``dimensionality`` estimator but on
Fourier-preprocessed z. ``experiment.freeze --climatology fourier``
reads from this directory.

Leakage guard: only ``<= cutoff`` data enters the fit, matching the
production discipline.

Run::

    python -m experiment.refit_dims_fourier

Outputs:
    processing/dimensions/params_fourier/results_weekday.csv
    processing/dimensions/params_fourier/results_saturday.csv
    processing/dimensions/params_fourier/results_sunday.csv
"""
from __future__ import annotations

import multiprocessing
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
from ray.util.multiprocessing import Pool

from edynamics.modelling_tools import Embedding, Lag
from edynamics.modelling_tools.estimators import dimensionality

from config import PROJECT_ROOT, load_config

from ._actuals import load_actuals, zscore_params, zscore_transform
from .freeze import DATA_CUTOFF


OUTPUT_DIR = PROJECT_ROOT / "processing" / "dimensions" / "params_fourier"
PLOTS_DIR = PROJECT_ROOT / "processing" / "dimensions" / "plots_fourier"


def _daytype_tag(idx: pd.DatetimeIndex, anchor_hours: int) -> np.ndarray:
    """Mirror processing.clustering.process: 07:00-shifted weekday."""
    offset = pd.Timedelta(hours=anchor_hours)
    shifted = idx - offset
    out = np.empty(len(idx), dtype=object)
    wd = shifted.weekday
    out[wd <= 4] = "weekday"
    out[wd == 5] = "saturday"
    out[wd == 6] = "sunday"
    return out


def main() -> None:
    cfg = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Build the Fourier-preprocessed z-series ──────────────────────────
    cutoff = DATA_CUTOFF
    print(f"loading actuals (cutoff = {cutoff})...", flush=True)
    raw = load_actuals(cutoff=None)
    print(f"  raw range: {raw.index[0]} to {raw.index[-1]} (n={len(raw)})",
          flush=True)

    print(f"fitting Fourier climatology (cutoff strictly <= {cutoff})...",
          flush=True)
    params = zscore_params(cutoff, method="fourier")
    z = zscore_transform(raw, params)
    print(f"  z mean={z.mean():+.6f}, std={z.std():.6f}", flush=True)

    # Match the production stationary_candidate range exactly
    # (2018-01-01 to 2024-12-31 23:00). The original dim-fit ran on
    # this restricted window; matching it is the apples-to-apples move
    # for the comparison.
    pipeline_start = pd.Timestamp("2018-01-01")
    z_restricted = z[(z.index >= pipeline_start) & (z.index <= cutoff)]
    print(f"  restricted to pipeline window: "
          f"{z_restricted.index[0]} to {z_restricted.index[-1]} "
          f"(n={len(z_restricted)})", flush=True)

    df = z_restricted.to_frame(cfg.data.variable_name)
    df.index = pd.DatetimeIndex(df.index)
    df = df.asfreq("h")
    df["daytype"] = _daytype_tag(df.index, cfg.data.day_anchor_hours)

    # ── Set up Ray pool (one init for the whole sweep) ───────────────────
    if not ray.is_initialized():
        ray.init(log_to_driver=False)
    pool = Pool(multiprocessing.cpu_count())

    # ── Per-day-type dimensionality sweep (mirrors processing.dimensions) ─
    for daytype in cfg.data.daytypes:
        print(f"\n[{daytype}] dimensionality sweep...", flush=True)
        library_indices = df.index[df["daytype"] == daytype][10:]
        rng = np.random.default_rng(7)  # deterministic anchor sample
        sampled = rng.choice(
            library_indices[:-1],
            size=int(0.2 * len(library_indices)),
            replace=False,
        )
        prediction_times = pd.DatetimeIndex(sampled)
        prediction_times = prediction_times[
            prediction_times.hour != (cfg.data.day_anchor_hours + 12) % 24
        ]
        print(f"  library: {len(library_indices)} pts; sampled "
              f"{len(prediction_times)} prediction times", flush=True)

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

        out_path = OUTPUT_DIR / f"results_{daytype}.csv"
        result.to_csv(out_path, index=True)
        print(f"  wrote {out_path}", flush=True)
        print(f"  curve head:\n{result.head().to_string()}", flush=True)

        plt.figure()
        plt.plot(result)
        plt.title(f"Dimensionality for {daytype.capitalize()} (Fourier)")
        plt.xlabel("K Nearest Neighbors Used")
        plt.ylabel("Pearson Correlation")
        plt.savefig(PLOTS_DIR / f"dimensionality_{daytype}.png")
        plt.close()

    print("\nDone. Fourier-derived dimension curves written to:")
    print(f"  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
