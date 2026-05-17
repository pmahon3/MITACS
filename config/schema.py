"""Typed, frozen schema for the MITACS pipeline configuration.

Every path field is an absolute ``Path`` once :func:`config.load_config` has
resolved it against the project root. Nothing here depends on the current
working directory.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PathsConfig:
    historical_csvs: Path
    pickled_dir: Path
    stationary_candidate: Path
    annual_params: Path
    preprocess_plots: Path
    clustered_csv: Path
    clustering_plots: Path
    dimensions_dir: Path
    dimensions_plots: Path
    pointwise_output_dir: Path
    global_output_dir: Path
    operator_output_dir: Path


@dataclass(frozen=True)
class DataConfig:
    year_range: tuple[int, int]
    variable_name: str
    daytypes: tuple[str, ...]
    day_anchor_hours: int


@dataclass(frozen=True)
class EmbeddingConfig:
    max_dimensions: int


@dataclass(frozen=True)
class ThetaGridConfig:
    theta_min: float
    theta_max: float
    theta_count: int
    sigma_min: float
    sigma_max: float
    sigma_count: int
    gl_penalty_C: float
    grid_spacing: str = "geometric"  # "geometric" | "linear"

    def _build(self, lo: float, hi: float, n: int) -> "np.ndarray":
        import numpy as np

        if self.grid_spacing == "geometric":
            if lo <= 0:
                raise ValueError("geometric grid requires min > 0")
            return np.geomspace(lo, hi, n)
        if self.grid_spacing == "linear":
            return np.linspace(lo, hi, n)
        raise ValueError(f"unknown grid_spacing {self.grid_spacing!r}")

    @property
    def theta_grid(self):
        return self._build(self.theta_min, self.theta_max, self.theta_count)

    @property
    def sigma_grid(self):
        return self._build(self.sigma_min, self.sigma_max, self.sigma_count)


@dataclass(frozen=True)
class SamplingConfig:
    sample_frac: float
    random_seed: int
    num_processes: int | None


@dataclass(frozen=True)
class PipelineConfig:
    paths: PathsConfig
    data: DataConfig
    embedding: EmbeddingConfig
    theta: ThetaGridConfig
    sampling: SamplingConfig
    project_root: Path
    profile: str = "full"

    # --- derived helpers (one place for the shared idioms) ---------------

    def dimensions_csv(self, daytype: str) -> Path:
        """Path to the embedding-dimension result CSV for a day type."""
        return self.paths.dimensions_dir / f"results_{daytype}.csv"

    def embedding_dim(self, daytype: str) -> int:
        """Embedding dimension for ``daytype`` (idxmax of its results CSV).

        This is the single source of truth for the ``idxmax()`` idiom that
        was previously duplicated across every ``process.py``.
        """
        df = pd.read_csv(self.dimensions_csv(daytype), index_col=0)
        return int(df.idxmax().iloc[0])
