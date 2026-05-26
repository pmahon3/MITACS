#!/bin/bash
# SLURM submission for the W=2y climatology re-score on Fir (DRA).
#
# Usage on Fir:
#   sbatch scripts/fir/submit_rescore.sh
#
# Submits a single-CPU job that re-runs predictor 2 (mean-iteration,
# global OLS theta=0) over the full 500-day post-cutoff backtest with
# a 2-year climatology refit. Output: scratch/data/rescore_W2y/.
#
# Prereqs on Fir:
#   - python 3.10 venv at $SCRATCH/mitacs_venv (separate from $HOME
#     so DRA's small home quota isn't hit by torch/ray wheels):
#       module load python/3.10
#       python -m venv $SCRATCH/mitacs_venv
#       source $SCRATCH/mitacs_venv/bin/activate
#       python -m pip install --upgrade pip wheel
#       python -m pip install --no-index pandas scipy matplotlib \
#           seaborn pyyaml bs4 numpy tqdm ray torch python-dateutil pytz
#       python -m pip install --no-deps edynamics==0.4.0
#     (--no-index pulls from the DRA wheelhouse; --no-deps on edynamics
#     because its deps are already wheelhouse-satisfied and the
#     transitive PyPI lookups otherwise drag in rpds-py / maturin which
#     needs Rust, not available in the wheelhouse.)
#   - The data cache:  data/ontario/*.pkl    (or scraped to the same
#     locations).  load_actuals reads from there.
#
# Runtime: ~25 s of compute (per laptop profile post-optimization).
# 15-minute wall is generous; short jobs typically dispatch fast.

#SBATCH --job-name=rescore_W2y
#SBATCH --account=def-CHANGE_ME           # set your DRA RAPI account
#SBATCH --time=0:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/rescore_W2y_%j.out
#SBATCH --error=logs/rescore_W2y_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs scratch/data/rescore_W2y

# Activate the venv.  Look in $SCRATCH/mitacs_venv first (the DRA-
# friendly location, used by the standard setup in the header
# comment).  Fall back to a local .venv/ in the repo for laptop /
# personal-cluster setups.
VENV_PATH=""
if [ -n "${SCRATCH:-}" ] && [ -f "$SCRATCH/mitacs_venv/bin/activate" ]; then
    VENV_PATH="$SCRATCH/mitacs_venv"
elif [ -f .venv/bin/activate ]; then
    VENV_PATH=".venv"
fi
if [ -z "$VENV_PATH" ]; then
    echo "ERROR: no venv found at \$SCRATCH/mitacs_venv or ./.venv" >&2
    echo "       See header comment for setup instructions." >&2
    exit 1
fi
echo "[$(date)] activating venv: $VENV_PATH"
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

# Sanity check the edynamics install
python -c "from edynamics.modelling_tools import Embedding, Lag" \
    || { echo "ERROR: edynamics import failed.  Try: pip install --no-deps edynamics==0.4.0" >&2 ; exit 2; }

echo "[$(date)] starting rescore_W2y on $(hostname)"
python -m scripts.fir.rescore_climatology \
    --window-years 2 \
    --out scratch/data/rescore_W2y/cell_B_meaniter_global_W2y.pkl
echo "[$(date)] done"
