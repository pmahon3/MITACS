<!--
OUTLINE STUB for a DEFERRED companion method paper. NOT being pursued
now (decision 2026-05-18). Captured so the structure + the load-bearing
requirement are not lost; likely easier to write AFTER the application
paper, which will have disciplined which theoretical properties are
actually load-bearing.

Register discipline is INHERITED unchanged: describe the operator as it
is; cite Resolvent_Framework@dbc7078 where concepts originate; NO
novelty assertion in either direction. Contribution-worthiness is still
an open question the user has not resolved — this outline does not
presume it.
-->

# Companion method paper — outline stub (deferred)

## Purpose

A standalone treatment of the local-Gaussian conditional-kernel
estimator and its relationship to the Resolvent_Framework `κ_Q`,
**demonstrated with purpose-built numerical examples on controlled
systems** — not the Ontario application (that is the separate
application paper, `writeup/tex/PAPER_DRAFT.tex`).

## The load-bearing requirement (do not lose this)

The application paper's synthetic gates (VAR(1) recovery /
non-Gaussianity / multi-step composition) are *validation* gates, not
*demonstrations*. A method paper needs examples **designed as
demonstrations** of the core elements:

- a system with a *known* non-Gaussian conditional law where the
  Gaussian-proxy gap (rank-1 `Σ_j` summarising a heavy-tailed scalar
  innovation) is shown and quantified against ground truth;
- a Dirac-limit (`Σ_j → 0`) example exhibiting the classical-Koopman
  collapse the programme names;
- a controlled case isolating drift `C_j` as the local linearisation
  of the minimal-sufficient factor;
- (if pursued) a constructed case probing the single-step → semigroup
  gap (Qualifier 3): where does iterated `Π_Δ` diverge from a
  CK-consistent `{Π_t}`?

These must be built fresh for demonstration; reusing the application's
gates would repeat the two-track confusion the split was made to avoid.

## Tentative structure (flat; fill after the application paper)

1. The estimator (full construction; the compressed §2 of the
   application paper expanded — WLS weighting, LOO-CV θ rule, the
   GL-degeneracy result in full).
2. Correspondence to `κ_Q` (the §2.2 material at full depth + the three
   qualifiers as the organising spine).
3. The rank-1 / Gaussian-proxy structural result, demonstrated.
4. Numerical demonstrations (the purpose-built examples above).
5. [open] Single-step vs semigroup — scope honestly; may stay a stated
   limitation rather than a contribution.
6. Relationship to the application paper (cite it for the real-data
   instantiation; this paper owes it no empirical claims).

## Why deferred, why after

- The application paper's evidence is done/provenance-locked; the
  theory paper's distinctive value needs new constructed examples.
- Writing the application first surfaces which theoretical properties
  are actually load-bearing (inverts the usual theory-proves-things-
  the-application-never-needed failure).
- Defers the still-open novelty/contribution-worthiness question to
  where it is answerable, on this paper's own terms.

## Soft-dependency mitigation (already in place)

The application paper does NOT forward-reference unwritten results: §2.2
positions the correspondence as *source-cited scoping*
(Resolvent_Framework@dbc7078), explicitly pointing here only for
*numerical demonstration*. The application paper stands alone; this
paper, if written, deepens it. No vapor dependency either direction.

Related project memory: mitacs-writeup-state, mitacs-theory-correspondence,
mitacs-rank1-structural, mitacs-rebaseline-facts.
