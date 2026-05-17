"""Centralized, CWD-independent configuration for the MITACS pipeline.

Usage::

    from config import load_config
    cfg = load_config()
    df = pd.read_csv(cfg.paths.clustered_csv, index_col=0)

This replaces the old ``.env`` / ``data.convert_to_absolute_path`` mechanism
entirely. The project root is the directory containing this package, resolved
once from this file's location, so paths are correct no matter where a stage
is launched from.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from .schema import (
    DataConfig,
    EmbeddingConfig,
    PathsConfig,
    PipelineConfig,
    SamplingConfig,
    ThetaGridConfig,
)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "pipeline.yaml"


def _resolve(rel: str) -> Path:
    """Join a project-root-relative string to an absolute path.

    Rejects leading-slash / parent-relative values so a stray ``/foo`` or
    ``../foo`` in the YAML fails loudly here instead of silently reading the
    filesystem root (the class of bug that broke the old scripts).
    """
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(
            f"config path {rel!r} must be project-root-relative "
            f"(no leading '/' or '..')"
        )
    return (PROJECT_ROOT / p).resolve()


@lru_cache(maxsize=None)
def load_config(config_path: Path | str | None = None) -> PipelineConfig:
    """Load and cache the pipeline configuration.

    Parameters
    ----------
    config_path:
        Optional override for the YAML file. Defaults to
        ``config/pipeline.yaml`` at the project root.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)

    paths = PathsConfig(**{k: _resolve(v) for k, v in raw["paths"].items()})

    d = raw["data"]
    data = DataConfig(
        year_range=tuple(d["year_range"]),
        variable_name=d["variable_name"],
        daytypes=tuple(d["daytypes"]),
        day_anchor_hours=int(d["day_anchor_hours"]),
    )

    embedding = EmbeddingConfig(max_dimensions=int(raw["embedding"]["max_dimensions"]))

    t = raw["theta"]
    theta = ThetaGridConfig(
        theta_min=float(t["theta_min"]),
        theta_max=float(t["theta_max"]),
        theta_count=int(t["theta_count"]),
        sigma_min=float(t["sigma_min"]),
        sigma_max=float(t["sigma_max"]),
        sigma_count=int(t["sigma_count"]),
        gl_penalty_C=float(t["gl_penalty_C"]),
    )

    s = raw["sampling"]
    sampling = SamplingConfig(
        sample_frac=float(s["sample_frac"]),
        random_seed=int(s["random_seed"]),
        num_processes=(None if s["num_processes"] is None else int(s["num_processes"])),
    )

    return PipelineConfig(
        paths=paths,
        data=data,
        embedding=embedding,
        theta=theta,
        sampling=sampling,
        project_root=PROJECT_ROOT,
    )


__all__ = ["load_config", "PipelineConfig", "PROJECT_ROOT"]
