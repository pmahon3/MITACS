"""Eigenmode characterization of the v2 coherence defect.

Pure linear-algebra analysis of the existing factor pickle
(scratch/data/multiscale_factor/factors_v2_final.pkl); no new fits.

Implements the metric pre-registered in
``notes/preregistrations/2026-05-26_multiscale-defect-eigenmodes/phase_a.yaml``.

CLI::

    python -m experiment.multiscale_defect_eigenmodes \
        --in scratch/data/multiscale_factor/factors_v2_final.pkl \
        --out scratch/data/multiscale_factor/eigenmodes.pkl
"""
