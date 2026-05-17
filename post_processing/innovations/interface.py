#!/usr/bin/env python3
"""
Post‐processing interface for Rose‐Operator runs.
Discovers and loads all artifacts written by save_all().
"""
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any


def _load_tensor(path: Path) -> np.ndarray:
    """Load a PyTorch‐saved tensor and return as NumPy array."""
    arr = torch.load(path)
    return arr.numpy() if hasattr(arr, "numpy") else arr


def _load_npy(path: Path) -> np.ndarray:
    """Load a NumPy .npy file."""
    return np.load(path, allow_pickle=False)


def _load_times(path: Path) -> np.ndarray:
    return np.load(path)


def load_all(run_dir: Path) -> Dict[str, Any]:
    r = list(run_dir.glob("residuals_*.pt")) or list(run_dir.glob("residuals_*.npy"))
    e = list(run_dir.glob("ensemble_*.pt"))
    c = list(run_dir.glob("coeffs_*.pt"))
    t = list(run_dir.glob("times_*.npy"))  # ← NEW
    if not (r and e and c and t):
        raise FileNotFoundError(f"Missing artefacts in {run_dir}")
    data = {
        "residuals": torch.load(r[0]).numpy(),
        "ensembles": torch.load(e[0]).numpy(),
        "coeffs": torch.load(c[0]).numpy(),
        "times": _load_times(t[0]),  # ← NEW
    }
    cov = list(run_dir.glob("covs_*.pt"))
    if cov:
        data["covs"] = torch.load(cov[0]).numpy()
    return data
