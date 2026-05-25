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
#       module load python/3.10
#       python -m venv .venv && source .venv/bin/activate
#       python -m pip install --no-index numpy pandas scipy tqdm \
#                                        torch ray wheel
#   - sibling EmpiricalDynamics repo cloned at $REPO/../EmpiricalDynamics
#       cd ..
#       git clone https://github.com/pmahon3/EmpiricalDynamics.git
#       cd MITACS
#       pip install -e ../EmpiricalDynamics --no-build-isolation
#     (--no-build-isolation skips the isolated build env so pip uses the
#      runtime deps already in the venv, avoiding the CVMFS slow-read
#      flakes that hit `setuptools`/`wheel` pulls on cold cache.)
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
