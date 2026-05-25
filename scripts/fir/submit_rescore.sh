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
#   - python 3.10 venv at $REPO/.venv with:
#       pip install numpy pandas scipy matplotlib seaborn pyyaml \
#                   torch ray bs4 dash plotly
#   - sibling EmpiricalDynamics repo cloned at
#       $REPO/../Takens_Whitney/EmpiricalDynamics
#     installed editable into the venv:
#       (.venv) $ pip install -e ../Takens_Whitney/EmpiricalDynamics
#   - The data cache:  data/ontario/*.pkl    (or scraped to the same
#     locations).  load_actuals reads from there.

#SBATCH --job-name=rescore_W2y
#SBATCH --account=def-CHANGE_ME           # set your DRA RAPI account
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/rescore_W2y_%j.out
#SBATCH --error=logs/rescore_W2y_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs scratch/data/rescore_W2y

# Activate the project venv. The local .venv layout (python3.10
# site-packages) is the assumed structure; see CLAUDE.md.
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    echo "ERROR: .venv not found.  See header comment for setup." >&2
    exit 1
fi

# Sanity check the editable edynamics install
python -c "from edynamics.modelling_tools import Embedding, Lag" \
    || { echo "ERROR: edynamics import failed.  pip install -e the sibling repo." >&2 ; exit 2; }

echo "[$(date)] starting rescore_W2y on $(hostname)"
python -m scripts.fir.rescore_climatology \
    --window-years 2 \
    --out scratch/data/rescore_W2y/cell_B_meaniter_global_W2y.pkl
echo "[$(date)] done"
