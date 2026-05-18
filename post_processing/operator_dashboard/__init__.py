# Operator analysis dashboard (data <-> theory objects). Dash app entry
# point is post_processing.operator_dashboard.app; importing this
# package must not pull in Dash or start a server, so nothing is
# imported here. The pure data/figure layers (loader, figures) are
# importable directly for headless testing.
