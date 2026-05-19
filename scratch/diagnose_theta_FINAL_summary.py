"""#41 FINAL SUMMARY — the theta-localization investigation, resolved.

This file is the single durable record of where the theta-rail
investigation LANDED, after several intermediate framings that were
each corrected by stricter testing. Read this, not the intermediate
diagnose_theta_*.py narratives (each is annotated with what superseded
it). All claims here are production-path / bootstrap-verified.

CHAIN (each step corrected the previous over-reach):
1. Dashboard #28 surfaced per-anchor C_j ~constant to 4 decimals.
   Bug ruled out (distinct objects). Real degeneracy.
2. One-step LOO theta rails to the grid ceiling. First framed
   "selector broken"; corrected to: the OBJECTIVE (one-step pred error)
   is monotone in bandwidth -> no interior optimum to find.
3. Multi-step CV "restored an interior optimum + theta drifts with h"
   -> built a kappa_Q-vs-{Pi_t} layer theory on it. RETRACTED: that
   was logistic-map-specific, did not replicate on a 2nd system.
4. Re-framed as "system-dependent: degenerate on Ontario/logistic/
   VAR(1), not regime-switch (interior at h=1)". The "regime-switch is
   different" claim came from reading ARGMIN POSITION.
5. **FINAL (bootstrap SE across anchors, the discipline the earlier
   steps skipped):** on Ontario, multi-step CV is FLAT-WITHIN-NOISE at
   EVERY horizon (1..24) — no statistically separated interior theta
   optimum. AND the same rigor on regime-switch ALSO gives
   separated=False at every horizon. The "system-dependent" contrast
   in step 4 was itself an unverified-argmin artifact.

RESOLVED CONCLUSION (defensible, minimal, verified):
  Under proper statistical testing, NO prediction-error CV objective
  (one-step OR multi-step, h in {1,2,4,8,16,24}) yields a
  statistically separated interior theta optimum on ANY system tested
  (Ontario, logistic, VAR(1), regime-switch). This is a property of
  the CRITERION (prediction-error CV does not identify a localization
  bandwidth here), not of which data it sees. Consequently the
  estimator, as run, reduces to a single GLOBAL linear drift at the
  d=2 z-score-lag embedding; kappa_Q's structure is carried by the
  diffusion Sigma_j (the paper's §3 Gaussian-proxy / non-Gaussian
  innovation), which is therefore LOAD-BEARING, not a side caveat.

DECISION (advisor-concurred): CHARACTERIZE, do not engineer a
bandwidth selector. The next lever is the EMBEDDING (does a richer
d / added state produce a non-degenerate conditional mean?), which is
#29/#30's territory (the honing memo already concluded this; #41
independently confirms the leverage is there, not in the bandwidth
rule). #41 closes here.

DEPENDENT ACTIONS:
- #18 paper §2.1/§2.2/§3: replace "bandwidth selected by true LOO-CV"
  with an honest description (no statistically meaningful locality
  scale recoverable at this embedding; estimator reduces to a global
  linear drift; Sigma_j / non-Gaussian innovation carries kappa_Q's
  structure). §3 is STRENGTHENED (load-bearing), not weakened.
- #30/#31 remain blocked, but on the EMBEDDING question, not a
  bandwidth-rule fix.
- #28 stays the verification surface for any future embedding change.

PROVENANCE-GRADE: INSPECTION-ONLY summary of dev diagnostics; every
claim here is the bootstrap-/production-verified residue after
discarding the intermediate over-reaches. Not a result artifact; the
honest record of a resolved investigation.
"""
