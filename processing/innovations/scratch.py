#!/usr/bin/env python3
"""
process.py

Given a precomputed list of “best θ” (one per anchor), run WLS exactly once per anchor using that θ
(pointwise mode), or—if --global-theta is set—use the median of those θ’s for all anchors (global mode).

For each anchor time t_i (loaded from times_<daytype>.npy) and its θ_i (from --thetas-input),
this script computes:
  • r(t_i, θ_i) = x_{t_i+1} – ŷ_{t_i+1}(θ_i), a vector of length d
  • C(t_i, θ_i) ∈ ℝ^{d×d}, the WLS coefficient matrix at θ_i.

Outputs (in <output-dir>):
  • residuals_<daytype>.pt   : torch.Tensor of shape (N, d)
  • coeffs_<daytype>.pt      : torch.Tensor of shape (N, d, d)
  • times_<daytype>.npy      : (N,) int64 array of anchor‐times (ns)  (copied from existing file)
  • thetas_<daytype>.npy     : (N,) float64 array of θ used for each anchor
  • [if --global-theta] global_theta.npy : (1,) float64 array (median of pointwise θ’s)

Usage:
    ./process.py [options]

Example (pointwise mode):
    ./process.py \
      --input-csv /path/to/data.csv \
      --variable-name zscore \
      --day-type saturday \
      --dimension-csv ../dimensions/params/results_saturday.csv \
      --thetas-input ./precomputed/best_theta_vals.npy \
      --times-input ./precomputed/times_saturday.npy \
      --output-dir ./outputs

Add --global-theta to use the median of θ’s for all anchors.
"""


from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import torch

from edynamics.modelling_tools import Lag, Embedding
from edynamics.modelling_tools.projectors import WeightedLeastSquares
from edynamics.modelling_tools.kernels import Exponential
from edynamics.modelling_tools.norms import Minkowski

from interface import save_all  # updated interface

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Globals for worker processes (populated by init_worker)
# ──────────────────────────────────────────────────────────────────────────────
embedding_global: Embedding


# ──────────────────────────────────────────────────────────────────────────────
# Worker initializer (called once per process)
# ──────────────────────────────────────────────────────────────────────────────
def init_worker(
    df_full: pd.DataFrame,
    embedding_dim: int,
    library_times: pd.DatetimeIndex,
    variable_name: str,
):
    """
    Build a shared Embedding object for all workers.
    """
    global embedding_global

    lags = [Lag(variable_name=variable_name, tau=-i) for i in range(embedding_dim)]
    embedding_global = Embedding(
        data=df_full,
        observers=lags,
        library_times=library_times,
    )
    embedding_global.compile()


# ──────────────────────────────────────────────────────────────────────────────
# Single‐anchor worker: compute residual and coefficient for one (t, θ) pair
# ──────────────────────────────────────────────────────────────────────────────
def compute_res_and_coef(args: tuple[pd.Timestamp, float]) -> tuple[pd.Timestamp, np.ndarray, np.ndarray]:
    """
    For a single anchor_time t and its θ:
      • Compute R = r(t, θ) ∈ ℝ^d
      • Compute C = C(t, θ) ∈ ℝ^{d×d}

    Returns:
      (anchor_time, R, C)
    """
    anchor_time, θ = args
    freq = embedding_global.frequency
    next_time = anchor_time + freq

    d = embedding_global.block.shape[1]

    # Default to NaNs
    R = np.full((d,), np.nan, dtype=float)
    C = np.full((d, d), np.nan, dtype=float)

    # Try to fetch “true” embedding x_{t+1}
    actual_blk = embedding_global.get_points(pd.DatetimeIndex([next_time]))
    if actual_blk.empty:
        return anchor_time, R, C

    true_vec = actual_blk.values[0]  # (d,)

    # Build WLS projector at θ
    proj = WeightedLeastSquares(norm=Minkowski(p=2), kernel=Exponential(theta=float(θ)))

    # Forecast one‐step at t → DataFrame
    pts = embedding_global.get_points(pd.DatetimeIndex([anchor_time]))
    wls_res = proj.project(
        embedding=embedding_global,
        points=pts,
        steps=1,
        step_size=1,
        leave_out=True,
        return_coefficients=True,
        use_innovations=False,
        rng=None,
    )

    # Extract coefficient C (shape (1,1,d,d))
    raw_coeff = wls_res.coefficients
    if isinstance(raw_coeff, torch.Tensor):
        arr = raw_coeff.cpu().numpy()  # (1,1,d,d)
        if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] == 1:
            Cmat = arr[0, 0, :, :]  # (d, d)
            C[:] = Cmat

    # Extract prediction ŷ_{t+1} and compute residual
    pred_df = wls_res.predictions
    if not pred_df.empty:
        try:
            pred_vec = pred_df.loc[(anchor_time, next_time)].values  # (d,)
            R[:] = true_vec - pred_vec
        except KeyError:
            pass

    return anchor_time, R, C


# ──────────────────────────────────────────────────────────────────────────────
# Main: parse arguments, load precomputed θs and times, compute res & coeffs
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute residuals and coefficients using precomputed per-anchor θ’s."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Hourly CSV with 'daytype' column and the target variable column",
    )
    parser.add_argument(
        "--variable-name",
        type=str,
        default="zscore",
        help="Column name to embed (e.g. 'zscore')",
    )
    parser.add_argument(
        "--day-type",
        type=str,
        required=True,
        help="Value of ‘daytype’ column (e.g. 'weekday', 'saturday', 'sunday')",
    )
    parser.add_argument(
        "--dimension-csv",
        type=Path,
        default=None,
        help=(
            "CSV (in '../dimensions/params/') whose idxmax gives embedding dimension. "
            "Default: '../dimensions/params/results_<day-type>.csv'"
        ),
    )
    parser.add_argument(
        "--thetas-input",
        type=Path,
        required=True,
        help="Path to precomputed per-anchor θ’s (1D .npy, length N).",
    )
    parser.add_argument(
        "--times-input",
        type=Path,
        required=True,
        help="Path to precomputed anchor‐times (1D .npy of int64 ns, length N).",
    )
    parser.add_argument(
        "--global-theta",
        action="store_true",
        help="If set, override all θ_i with the median of the input θ’s.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs"),
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of parallel workers (omit or 0 → serial)",
    )

    args = parser.parse_args()

    # ──────────────────────────────────────────────────────────────────────────
    # 1) Load CSV and select day‐type, build library_times for embedding
    # ──────────────────────────────────────────────────────────────────────────
    df = (
        pd.read_csv(args.input_csv, index_col=0, parse_dates=True)
        .asfreq("h")
    )
    if args.dimension_csv is None:
        dim_csv = Path("../dimensions/params") / f"results_{args.day_type}.csv"
    else:
        dim_csv = args.dimension_csv

    dims_df = pd.read_csv(dim_csv, index_col=0)
    embedding_dim = int(dims_df.idxmax()[0])

    full_lib = df.index[df["daytype"] == args.day_type]
    full_lib = full_lib[embedding_dim : -1]  # skip first d, drop last

    # ──────────────────────────────────────────────────────────────────────────
    # 2) Load precomputed anchor times (N,)
    # ──────────────────────────────────────────────────────────────────────────
    if not args.times_input.exists():
        raise FileNotFoundError(f"Cannot find times file: {args.times_input}")
    anchor_times_ns = np.load(args.times_input)  # int64 ns
    if anchor_times_ns.ndim != 1:
        raise ValueError(f"--times-input must be 1D, got shape {anchor_times_ns.shape}")
    N = anchor_times_ns.shape[0]
    # Convert to pandas.DatetimeIndex
    anchor_times = pd.to_datetime(anchor_times_ns)

    # ──────────────────────────────────────────────────────────────────────────
    # 3) Load precomputed per-anchor θ’s (length N)
    # ──────────────────────────────────────────────────────────────────────────
    if not args.thetas_input.exists():
        raise FileNotFoundError(f"Cannot find θ’s file: {args.thetas_input}")
    theta_profiles = np.load(args.thetas_input)  # shape (N,)
    if theta_profiles.ndim != 1 or theta_profiles.shape[0] != N:
        raise ValueError(
            f"--thetas-input must be length N={N}, but got shape {theta_profiles.shape}"
        )

    # find the best theta for each point
    best_theta = np.min(theta_profiles, axis=1)  # (N,)

    # ──────────────────────────────────────────────────────────────────────────
    # 4) If global‐θ mode: override all θ_i = median(best_theta)
    # ──────────────────────────────────────────────────────────────────────────
    if args.global_theta:
        # replace all entries in best_thetas with the average of best_thetas
        best_theta = np.full_like(best_theta, np.mean(best_theta))

    # ──────────────────────────────────────────────────────────────────────────
    # 5) Prepare output directory
    # ──────────────────────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 6) Build list of (anchor_time, θ_i) tuples
    # ──────────────────────────────────────────────────────────────────────────
    tasks: list[tuple[pd.Timestamp, float]] = [
        (anchor_times[i], float(best_theta[i])) for i in range(N)
    ]

    # ──────────────────────────────────────────────────────────────────────────
    # 7) Parallel or serial map over tasks to compute (R, C)
    # ──────────────────────────────────────────────────────────────────────────
    results: list[tuple[pd.Timestamp, np.ndarray, np.ndarray]] = []

    if args.num_processes and args.num_processes > 0:
        pool = Pool(
            processes=min(cpu_count(), args.num_processes),
            initializer=init_worker,
            initargs=(df, embedding_dim, full_lib, args.variable_name),
        )
        it = pool.imap_unordered(compute_res_and_coef, tasks)
        for (t, R, C) in tqdm(it, total=N, desc=f"{args.day_type}"):
            results.append((t, R, C))
        pool.close()
        pool.join()
    else:
        init_worker(df, embedding_dim, full_lib, args.variable_name)
        for task in tqdm(tasks, total=N, desc=f"{args.day_type} [serial]"):
            results.append(compute_res_and_coef(task))

    # ──────────────────────────────────────────────────────────────────────────
    # 8) Collect and sort by anchor_time
    # ──────────────────────────────────────────────────────────────────────────
    results.sort(key=lambda x: x[0])
    sorted_times = np.array([t.value for (t, _, _) in results], dtype=np.int64)  # (N,)
    d = embedding_dim

    # Build arrays: R_all (N, d), C_all (N, d, d)
    R_all = np.empty((N, d), dtype=float)
    C_all = np.empty((N, d, d), dtype=float)

    for i, (_t, R, C) in enumerate(results):
        R_all[i] = R
        C_all[i] = C

    # ──────────────────────────────────────────────────────────────────────────
    # 9) Save results:
    #    • residuals_<daytype>.pt   shape (N, d)
    #    • coeffs_<daytype>.pt      shape (N, d, d)
    #    • times_<daytype>.npy      shape (N,)
    #    • thetas_<daytype>.npy     shape (N,)
    #    • [if global] global_theta.npy  shape (1,)
    # ──────────────────────────────────────────────────────────────────────────

    res_tensor = torch.as_tensor(R_all, dtype=torch.float32)      # (N, d)
    coeffs_tensor = torch.as_tensor(C_all, dtype=torch.float32)   # (N, d, d)

    save_all(
        residuals=res_tensor,
        coefficients=coeffs_tensor,
        anchor_times=sorted_times,
        thetas=best_theta,
        output_dir=args.output_dir,
        run_tag=args.day_type
    )

    # Additionally, if global-theta flag was set, save the single scalar
    if args.global_theta:
        global_theta = best_theta[0]  # all entries are identical
        np.save(args.output_dir / "global_theta.npy", np.array([global_theta]))
        print(f"✔ Saved global_theta.npy = {global_theta:.6f}")

    print("✔ Outputs saved to:", args.output_dir.resolve())


if __name__ == "__main__":
    main()