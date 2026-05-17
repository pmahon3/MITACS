#!/usr/bin/env python3
"""
process.py

For each “anchor” time t in the chosen day‐type, and for each candidate θ,
this script computes both:

   • the d‐dimensional WLS residual vector
       r(t,θ) = x_{t+1} − ŷ_{t+1}(θ),
     where ŷ_{t+1}(θ) is the one‐step, deterministic WLS forecast of the
     embedded vector at time t+1 using leave‐one‐out.

   • the corresponding local linear coefficient matrix C(t,θ) ∈ ℝ^{d×d}
     from that same WLS fit.

Instead of pulling “true” x_{t+1} from `embedding_global.block`, we call
`embedding_global.get_points([next_time])` so that any valid embedding
is retrieved even if `block` does not contain next_time.

Outputs (written to <output‐dir>):
  • residuals_<daytype>.pt   : a torch Tensor of shape (N, |Θ|, d)
  • coeffs_<daytype>.pt      : a torch Tensor of shape (N, |Θ|, d, d)
  • times_<daytype>.npy      : N‐length array of anchor‐times (dtype=int64 nanoseconds)
  • thetas_<daytype>.npy     : |Θ|‐length array of θ values (float64)

Usage:
    ./process.py [options]

Example:
    ./process.py \
        --input-csv /path/to/stationary_candidate.csv \
        --variable-name zscore \
        --day-type saturday \
        --dimension-csv ../dimensions/params/results_saturday.csv \
        --theta-min 0.1 \
        --theta-max 5.0 \
        --theta-count 25 \
        --sample-frac 0.8 \
        --num-processes 4 \
        --output-dir ./outputs
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

from interface import save_all

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# 1) GLOBALS (populated by init_worker)
# ──────────────────────────────────────────────────────────────────────────────
embedding_global: Embedding
theta_grid_global: np.ndarray


# ──────────────────────────────────────────────────────────────────────────────
# 2) Worker initializer (called once per process)
# ──────────────────────────────────────────────────────────────────────────────
def init_worker(
    df_full: pd.DataFrame,
    thetas_arg: np.ndarray,
    embedding_dim: int,
    library_times: pd.DatetimeIndex,
    variable_name: str,
) -> None:
    """
    Build a single Embedding object (shared by all workers) and store the θ‐grid.
    """
    global embedding_global, theta_grid_global

    theta_grid_global = thetas_arg

    # Build lag embedding for [0, -1, -2, …, -(embedding_dim-1)]
    lags = [Lag(variable_name=variable_name, tau=-i) for i in range(embedding_dim)]
    embedding_global = Embedding(
        data=df_full,
        observers=lags,
        library_times=library_times,
    )
    embedding_global.compile()


# ──────────────────────────────────────────────────────────────────────────────
# 3) Single‐time worker: compute both residuals and coefficient‐matrices for all θ at anchor t
# ──────────────────────────────────────────────────────────────────────────────
def compute_vector_residuals(anchor_time: pd.Timestamp) -> tuple[pd.Timestamp, np.ndarray, np.ndarray]:
    """
    For a single anchor_time t, and for each θ in theta_grid_global, compute:

        r(t,θ) = x_{t+1} − ŷ_{t+1}(θ),
        C(t,θ): the fitted coefficient matrix from WLS at that θ.

    Instead of using embedding_global.block.loc[next_time],
    we call embedding_global.get_points([next_time]) to retrieve the true embedded
    vector (d-dimensional) if it exists.

    Returns:
        ( anchor_time,  R,  C )
    where:
      • R is an array of shape (|Θ|, d),  each row R[i,:] = r(t,θ_i).
      • C is an array of shape (|Θ|, d, d), each C[i,:,:] is the d×d matrix
        returned by the WLS fit at θ_i.

    If x_{t+1} or the WLS prediction fails, we fill with NaNs.
    """
    freq = embedding_global.frequency
    next_time = anchor_time + freq

    d = embedding_global.block.shape[1]
    K = theta_grid_global.size

    # Pre‐allocate outputs as NaNs
    R = np.full((K, d), np.nan, dtype=float)
    C = np.full((K, d, d), np.nan, dtype=float)

    # 1) Try to fetch “true” embedded vector at t+1 via get_points
    actual_blk = embedding_global.get_points(pd.DatetimeIndex([next_time]))
    if actual_blk.empty:
        # No true embedding at t+1 → leave R and C as NaNs
        return anchor_time, R, C

    true_vec = actual_blk.values[0]  # shape = (d,)

    # 2) Prepare the WLS projector
    proj = WeightedLeastSquares(kernel=Exponential(theta=0.0))

    for i, theta in enumerate(theta_grid_global):
        proj.kernel.theta = float(theta)

        # One‐step forecast from anchor_time → DataFrame of predictions
        wls_res = proj.project(
            embedding=embedding_global,
            points=embedding_global.get_points(pd.DatetimeIndex([anchor_time])),
            steps=1,
            step_size=1,
            leave_out=True,
            return_coefficients=True,
            use_innovations=False,
            rng=None,
        )

        # Extract coefficient matrix
        raw_coeff = wls_res.coefficients  # this is a torch.Tensor of shape (1, 1, d, d)
        coef_mat = None
        if isinstance(raw_coeff, torch.Tensor):
            arr = raw_coeff.cpu().numpy()  # now a NumPy array of shape (1, 1, d, d)
            if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] == 1 and arr.shape[2] == d and arr.shape[3] == d:
                coef_mat = arr[0, 0, :, :]    # squeeze to shape (d, d)
            else:
                coef_mat = None
        else:
            coef_mat = None

        if isinstance(coef_mat, np.ndarray) and coef_mat.shape == (d, d):
            C[i, :, :] = coef_mat
        # If coef_mat is None or wrong shape, C[i,:,:] remains NaN

        # --- Prediction extraction (unchanged) ---
        pred_df = wls_res.predictions
        if not pred_df.empty:
            try:
                pred_vec = pred_df.loc[(anchor_time, next_time)].values  # shape = (d,)
                R[i, :] = true_vec - pred_vec
            except KeyError:
                pass

    return anchor_time, R, C


# ──────────────────────────────────────────────────────────────────────────────
# 4) Main: parse arguments, sample anchors, run Pool/serial, save results
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute embedding‐vector residuals and coefficients for each (t,θ)"
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
        "--theta-min",
        type=float,
        default=0.1,
        help="Minimum θ for grid",
    )
    parser.add_argument(
        "--theta-max",
        type=float,
        default=5.0,
        help="Maximum θ for grid",
    )
    parser.add_argument(
        "--theta-count",
        type=int,
        default=25,
        help="Number of θ values in [theta-min, theta-max]",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.8,
        help="Fraction of library times to sample as anchors",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of parallel workers (omit or 0 → serial)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for random selection of anchors",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs"),
        help="Directory to save residuals_<daytype>.pt, coeffs_<daytype>.pt, times_<daytype>.npy, thetas_<daytype>.npy",
    )

    args = parser.parse_args()

    # ——————————————————————————————————————————
    # 4a) Load the CSV and select only the desired day‐type
    # ——————————————————————————————————————————
    df = (
        pd.read_csv(args.input_csv, index_col=0, parse_dates=True)
        .asfreq("h")
    )

    # 4b) Decide embedding dimension d
    if args.dimension_csv is None:
        dim_csv = Path("../dimensions/params") / f"results_{args.day_type}.csv"
    else:
        dim_csv = args.dimension_csv

    dims_df = pd.read_csv(dim_csv, index_col=0)
    embedding_dim = int(dims_df.idxmax()[0])

    # 4c) Build the full library times for that day‐type, skipping the first d hours
    full_lib = df.index[df["daytype"] == args.day_type]
    full_lib = full_lib[embedding_dim : -1]  # no embedding before first d samples; drop last

    # 4d) Sample a random fraction of library times as “anchor_times”
    rng = np.random.default_rng(args.random_seed)
    n_lib = len(full_lib)
    n_anchor = max(1, int(np.ceil(args.sample_frac * n_lib)))
    anchor_subset = rng.choice(full_lib, size=n_anchor, replace=False)
    anchor_times = pd.DatetimeIndex(anchor_subset)

    # 4e) Build the θ‐grid
    theta_grid = np.linspace(args.theta_min, args.theta_max, args.theta_count)

    # 4f) Prepare output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ——————————————————————————————————————————
    # 4g) Parallel or serial map over anchor_times
    # ——————————————————————————————————————————
    raw_results: list[tuple[pd.Timestamp, np.ndarray, np.ndarray]] = []

    if args.num_processes and args.num_processes > 0:
        pool = Pool(
            processes=min(cpu_count(), args.num_processes),
            initializer=init_worker,
            initargs=(df, theta_grid, embedding_dim, full_lib, args.variable_name),
        )
        it = pool.imap_unordered(compute_vector_residuals, anchor_times)
        for (t, R, C) in tqdm(it, total=len(anchor_times), desc=f"{args.day_type}"):
            raw_results.append((t, R, C))
        pool.close()
        pool.join()
    else:
        # Serial
        init_worker(df, theta_grid, embedding_dim, full_lib, args.variable_name)
        for t in tqdm(anchor_times, desc=f"{args.day_type} [serial]"):
            tt, R, C = compute_vector_residuals(t)
            raw_results.append((tt, R, C))

    # ——————————————————————————————————————————
    # 4h) Collect and sort the results
    # ——————————————————————————————————————————
    raw_results.sort(key=lambda x: x[0])  # sort by timestamp

    sorted_times = np.array([t.value for (t, _, _) in raw_results], dtype=np.int64)  # nanoseconds
    N = len(raw_results)
    K = theta_grid.size
    d = embedding_dim

    # Build arrays:
    residuals_arr = np.empty((N, K, d), dtype=float)
    coeffs_arr    = np.empty((N, K, d, d), dtype=float)

    for i, (_t, R, C) in enumerate(raw_results):
        residuals_arr[i, :, :] = R
        coeffs_arr[i, :, :, :] = C

    # 4i) Save to disk using the new save_all(...)
    res_tensor    = torch.as_tensor(residuals_arr, dtype=torch.float32)
    coeffs_tensor = torch.as_tensor(coeffs_arr,    dtype=torch.float32)

    save_all(
        residuals=res_tensor,        # shape (N, K, d)
        coeffs=coeffs_tensor,        # shape (N, K, d, d)
        anchor_times=sorted_times,   # ndarray length N of int64 (ns)
        thetas=theta_grid,           # ndarray length K of float64
        output_dir=args.output_dir,
        run_tag=args.day_type,
    )

    print(f"✔ Saved:")
    print(f"    - residuals_{args.day_type}.pt     shape = ({N}, {K}, {d})")
    print(f"    - coeffs_{args.day_type}.pt        shape = ({N}, {K}, {d}, {d})")
    print(f"    - times_{args.day_type}.npy         length = {N}")
    print(f"    - thetas_{args.day_type}.npy        length = {K}")
    print(f"  → Output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()