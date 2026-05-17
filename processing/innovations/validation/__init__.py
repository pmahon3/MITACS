"""Synthetic ground-truth validation for the stochastic-operator stage.

`validation gates everything`: the VAR(1) recovery test here must pass before
any Ontario output (or any pre-existing `.pt` artifact) is trusted, because
the loaded `edynamics` projector API changed and existing artifacts may be
from a now-shadowed older library version.
"""
