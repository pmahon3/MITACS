"""Multiscale factor coherence: §6.1+§6.2 of the seed.

Direct h-step factor estimation (C_h, Sigma_h) and comparison to the
iterated 1-step factor (P_1^h, iterated-Sigma) on pre-cutoff Ontario
demand data, holding the embedding fixed at the registered spec.

Preregistered in notes/preregistrations/2026-05-26_multiscale-direct-h-step/.

CLI::

    python -m experiment.multiscale_factor_coherence \
        --out scratch/data/multiscale_factor/factors.pkl
"""
