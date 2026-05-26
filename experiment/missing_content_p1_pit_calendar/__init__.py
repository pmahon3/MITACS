"""P1 of the missing-content thread: conditional PIT analysis with
calendar candidates.

Pre-registered in
notes/preregistrations/2026-05-26_pit-calendar-fingerprint/phase_a.yaml.
Thread parent: 2026-05-26_missing-content-thread/thread.yaml (node P1).

Reads the v2 factor pickle (C_1, Sigma_1 per day_type) and computes
analytic PIT values for the iterated kappa_1 predictor at every
post-cutoff (delivery_day, hour_of_day). Stratifies by
(season, HE_bin, day_type). Scores calendar candidates by cell-
weighted partial correlation against the PIT residual, with paired-
day bootstrap CI and 1000-permutation null per candidate.

CLI::

    python -m experiment.missing_content_p1_pit_calendar \
        --in scratch/data/multiscale_factor/factors_v2_final.pkl \
        --out scratch/data/missing_content_p1/pit_calendar.pkl
"""
