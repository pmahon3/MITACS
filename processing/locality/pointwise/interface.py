#!/usr/bin/env python3
"""
interface_processing.py

Provides functions to save & load four artifacts:
  • residuals_<tag>.pt   →  (N, K, d)     float32 torch tensor
  • coeffs_<tag>.pt      →  (N, K, d, d)  float32 torch tensor
  • times_<tag>.npy      →  (N,)          int64 nanoseconds array
  • thetas_<tag>.npy     →  (K,)          float64 array of θ values

         N = # of anchor‐times
         K = # of candidate θ’s
         d = embedding dimension
"""

from pathlib import Path
import numpy as np
import torch
from typing import Union, Dict, Any


def save_all(
    *,
    residuals: torch.Tensor,        # shape (N, K, d) or convertible
    coeffs: torch.Tensor,           # shape (N, K, d, d) or convertible
    anchor_times: Union[np.ndarray, list],  # length‐N of np.datetime64[ns] or int64(ns)
    thetas: Union[np.ndarray, list],        # length‐K floats
    output_dir: Path,
    run_tag: str,
) -> None:
    """
    Save four artifacts under output_dir with filenames:
        • residuals_<run_tag>.pt
        • coeffs_<run_tag>.pt
        • times_<run_tag>.npy
        • thetas_<run_tag>.npy

    residuals:   torch.Tensor of shape (N, K, d), dtype float32 (or castable).
    coeffs:      torch.Tensor of shape (N, K, d, d), dtype float32 (or castable).
    anchor_times: length‐N array of either np.datetime64[ns] or int64(ns).
    thetas:      length‐K array of floats.
    """

    # 1) Make sure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2) Cast residuals → float32 torch.Tensor
    if not isinstance(residuals, torch.Tensor):
        residuals = torch.as_tensor(residuals, dtype=torch.float32)
    else:
        residuals = residuals.to(dtype=torch.float32)

    # 3) Cast coeffs → float32 torch.Tensor
    if not isinstance(coeffs, torch.Tensor):
        coeffs = torch.as_tensor(coeffs, dtype=torch.float32)
    else:
        coeffs = coeffs.to(dtype=torch.float32)

    # 4) Convert anchor_times → int64‐ns array
    atimes_arr = np.asarray(anchor_times)
    if np.issubdtype(atimes_arr.dtype, np.datetime64):
        atimes_ns = atimes_arr.astype("int64")  # datetime64[ns] → int64(ns)
    else:
        atimes_ns = atimes_arr.astype("int64")

    # 5) Convert thetas → float64 array
    thetas_arr = np.asarray(thetas, dtype=np.float64)

    # 6) Build filenames
    fn_resid  = output_dir / f"residuals_{run_tag}.pt"
    fn_coeff  = output_dir / f"coeffs_{run_tag}.pt"
    fn_times  = output_dir / f"times_{run_tag}.npy"
    fn_thetas = output_dir / f"thetas_{run_tag}.npy"

    # 7) Save each artifact
    torch.save(residuals, fn_resid)
    torch.save(coeffs,   fn_coeff)
    np.save(fn_times,  atimes_ns)
    np.save(fn_thetas, thetas_arr)

    # (Optional) print or log
    # print(f"[✓] Saved residuals → {fn_resid}")
    # print(f"[✓] Saved coeffs    → {fn_coeff}")
    # print(f"[✓] Saved times     → {fn_times}")
    # print(f"[✓] Saved thetas    → {fn_thetas}")


def load_vector_residuals(run_dir: Path) -> Dict[str, Any]:
    """
    Load only the residual‐tensor, times, and thetas from run_dir.

    Looks for exactly one file matching each pattern:
      • residuals_*.pt
      • times_*.npy
      • thetas_*.npy

    Returns a dict:
      {
        "residuals": np.ndarray of shape (N, K, d), dtype=float32,
        "times":     np.ndarray of shape (N,), dtype=int64 (ns),
        "thetas":    np.ndarray of shape (K,), dtype=float64,
      }

    Raises FileNotFoundError if any required pattern is missing.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"{run_dir} is not a valid directory.")

    # Find matching files
    resid_files = sorted(run_dir.glob("residuals_*.pt"))
    times_files = sorted(run_dir.glob("times_*.npy"))
    thetas_files= sorted(run_dir.glob("thetas_*.npy"))

    if not resid_files:
        raise FileNotFoundError(f"No file matching 'residuals_*.pt' in {run_dir}")
    if not times_files:
        raise FileNotFoundError(f"No file matching 'times_*.npy' in {run_dir}")
    if not thetas_files:
        raise FileNotFoundError(f"No file matching 'thetas_*.npy' in {run_dir}")

    fn_resid  = resid_files[0]
    fn_times  = times_files[0]
    fn_thetas = thetas_files[0]

    # Load residuals (torch → NumPy)
    resid_tensor = torch.load(fn_resid)
    if hasattr(resid_tensor, "numpy"):
        resid_np = resid_tensor.numpy()
    else:
        resid_np = np.array(resid_tensor, dtype=np.float32)

    # Load times & thetas
    times_arr  = np.load(fn_times)    # int64 (ns)
    thetas_arr = np.load(fn_thetas)   # float64

    return {
        "residuals": resid_np,   # shape (N, K, d)
        "times":     times_arr,  # shape (N,) int64
        "thetas":    thetas_arr, # shape (K,) float64
    }


def load_coefficients(run_dir: Path) -> np.ndarray:
    """
    Load only the coefficient‐tensor from run_dir.

    Looks for exactly one file matching:
      • coeffs_*.pt

    Returns a NumPy array of shape (N, K, d, d), dtype=float32.

    Raises FileNotFoundError if no file matching 'coeffs_*.pt' is found.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"{run_dir} is not a valid directory.")

    coeffs_files = sorted(run_dir.glob("coeffs_*.pt"))
    if not coeffs_files:
        raise FileNotFoundError(f"No file matching 'coeffs_*.pt' in {run_dir}")

    fn_coeff = coeffs_files[0]
    coeff_tensor = torch.load(fn_coeff)
    if hasattr(coeff_tensor, "numpy"):
        coeff_np = coeff_tensor.numpy()
    else:
        coeff_np = np.array(coeff_tensor, dtype=np.float32)

    return coeff_np  # shape (N, K, d, d)


def load_all(run_dir: Path) -> Dict[str, Any]:
    """
    Convenience wrapper: load everything (residuals, coeffs, times, thetas) at once.
    Returns a dict with keys:
      {
        "residuals": np.ndarray (N,K,d),
        "coeffs":    np.ndarray (N,K,d,d),
        "times":     np.ndarray (N,),
        "thetas":    np.ndarray (K,),
      }
    """
    vec_data   = load_vector_residuals(run_dir)
    coeff_data = load_coefficients(run_dir)

    vec_data["coeffs"] = coeff_data
    return vec_data
