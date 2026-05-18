"""Shared pre-cutoff actuals loader + z-score derivation.

ONE implementation used by both ``freeze.py`` (to fingerprint the
pre-registered data) and ``predict.py`` (to derive the z-score
parameters at predict time). They must agree exactly, so they call the
same code -- not two parallel loaders. The hourly-index construction
mirrors ``pre_processing.year_wise_standardization`` (skiprows=3,
Date+Hour-1 -> timestamp, "Ontario Demand").

The cutoff is enforced as a HARD guard: any row dated strictly after it
is dropped before any statistic is computed, so post-cutoff data can
never enter the predictor's input representation (the leakage guard).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config


def _load_raw_actuals() -> pd.Series:
    """All hourly "Ontario Demand" from the git-tracked per-year
    ``PUB_Demand_<yr>.csv`` files (no cutoff). Single parser; mirrors
    ``pre_processing.year_wise_standardization``."""
    cfg = load_config()
    frames = []
    for f in sorted(Path(cfg.paths.historical_csvs).glob("PUB_Demand_*.csv")):
        df = pd.read_csv(f, skiprows=3).dropna(
            subset=["Date", "Hour", "Ontario Demand"]
        )
        df["Time"] = pd.to_datetime(
            df["Date"].astype(str)
            + " "
            + (df["Hour"] - 1).astype(int).astype(str)
            + ":00:00"
        )
        frames.append(df.set_index("Time")[["Ontario Demand"]])
    return pd.concat(frames).sort_index()["Ontario Demand"]


def load_actuals(cutoff: pd.Timestamp | None = None) -> pd.Series:
    """Hourly "Ontario Demand". With ``cutoff``, strictly ``<= cutoff``
    (the HARD leakage guard for fits / z-score params). Without, the full
    tracked series -- ONLY for building post-cutoff query states, never a
    fit."""
    hourly = _load_raw_actuals()
    if cutoff is not None:
        hourly = hourly[hourly.index <= cutoff]
        if hourly.empty:
            raise ValueError(f"no actuals <= {cutoff}")
    return hourly


def load_pre_cutoff_actuals(cutoff: pd.Timestamp) -> pd.Series:
    """Hourly "Ontario Demand" strictly ``<= cutoff`` (leakage guard)."""
    return load_actuals(cutoff)


def zscore_params(cutoff: pd.Timestamp) -> dict[str, pd.Series]:
    """Month+hour-of-day mean/std climatology from pre-cutoff actuals.

    This is the de-seasonalisation the embedded ``zscore`` variable uses
    (``mu_mh``/``sigma_mh`` in pre_processing). Derived strictly from
    ``<= cutoff`` data -- never recomputed against post-cutoff actuals.
    """
    h = load_pre_cutoff_actuals(cutoff)
    by = [h.index.month, h.index.hour]
    mu_mh = h.groupby(by).mean()
    sigma_mh = h.groupby(by).std()
    return {"mu_mh": mu_mh, "sigma_mh": sigma_mh}


def zscore_transform(
    raw: pd.Series, params: dict[str, pd.Series]
) -> pd.Series:
    """Apply frozen month+hour climatology to raw demand -> zscore."""
    idx = raw.index
    keys = list(zip(idx.month, idx.hour))
    mu = params["mu_mh"].reindex(keys).to_numpy()
    sd = params["sigma_mh"].reindex(keys).to_numpy()
    return pd.Series((raw.to_numpy() - mu) / sd, index=idx, name="zscore")


def actuals_fingerprint(cutoff: pd.Timestamp) -> str:
    """Content hash of the exact pre-cutoff actuals used.

    Deterministic over (timestamp_ns, demand) rows so any later change to
    the historical CSVs (re-scrape, revision) is detected by the frozen
    spec's hash.
    """
    h = load_pre_cutoff_actuals(cutoff)
    buf = np.ascontiguousarray(
        np.column_stack([h.index.view("int64"), h.to_numpy(dtype="float64")])
    )
    return hashlib.sha256(buf.tobytes()).hexdigest()


def zscore_params_fingerprint(cutoff: pd.Timestamp) -> str:
    """Content hash of the derived z-score climatology -- recorded with
    every forecast so post-hoc drift is detectable even though the values
    are not stored in the spec."""
    p = zscore_params(cutoff)
    parts = []
    for name in ("mu_mh", "sigma_mh"):
        s = p[name].sort_index()
        parts.append(name.encode())
        parts.append(np.ascontiguousarray(s.to_numpy(dtype="float64")).tobytes())
    return hashlib.sha256(b"".join(parts)).hexdigest()
