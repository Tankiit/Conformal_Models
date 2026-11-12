---
title: "Story - CLeaR Leakage Paper"
tags: [planning, story, clear, leakage, arc]
date: 2026-11-11
---

# 📖 Story: The CLeaR Paper (Leakage / Information Use)

> The narrative arc for the CLeaR arm, written down 2026-11-11 (accepted in chat).
> Companion arc: [[06 - Papers & Planning/Story - CVPR Collapse Paper|CVPR story]].
*Question this paper owns:* **does the predictor use information outside the
declared concepts?** (Paper Split §1.)

---

## Beat 1 — The promise and the blind spot

Concept-bottleneck models promise predictions that flow through declared
concepts, so explanations and interventions can be trusted. But a predictor can
use information **outside** the declared list — a hidden nuisance, a shortcut —
and the standard audits cannot see it:

> **Conformal coverage alone neither detects nor excludes leakage.**
> A coverage miss does not diagnose leakage; valid coverage does not certify
> its absence. ([[01 - Literature & Theory/Leakage One-Sided Claim|Leakage: the One-Sided Claim]] §1)

The accuracy-gap heuristics also failed here, twice over: the old B-minus-R gap
reading was training mismatch, not leakage (§5 of the theory note), and the
nn-leakage score had to be rebuilt twice (v1 sign-clip → v2 control → v3 null
distribution) before the statistic measured anything real.

## Beat 2 — The one-sided claim

The central claim is deliberately **one-sided**. Work under

$$Y = f(C), \qquad X = g(C, U)$$

with any $U$–$Y$ correlation created **only through training-data sampling**.
Run a controlled paired intervention — hold the true concepts $C$ fixed, flip
the nuisance $U$, re-render — and:

- output moves ⟹ the model uses information beyond the declared concepts ⟹ **leakage** (under $Y=f(C)$);
- if instead $Y = f(C, U)$, sensitivity means an **incomplete concept list**, not leakage — the two diagnoses stay distinct in the paper;
- a missed alarm is **not** proof of absence — no certificate in the clean direction. This asymmetry is stated, not hidden.

## Beat 3 — Why sets, not probabilities (the gate)

The step-3 gate: the **set-based test must detect $U$-dependence that an
ordinary probability/logit-sensitivity test misses**. The demonstration: a
residual CBM trained at $\rho_{\text{train}}=1$ (deterministic shortcut,
$U$ = parity of $Y$) shows conformal **set changes beyond its
$\rho_{\text{train}}=0$ null even when the marginal probability of $Y$ barely
moves**. If probability-change detects everything the sets do, the conformal
machinery is decoration and the paper does not exist.

## Beat 4 — The deployed-pipeline subtlety (what makes it a real paper)

Two test variants, and the choice of primary defines the headline (was **D5**):

- **Test 1** (ground-truth concepts fixed): a strict label head gives **exactly
  zero** set-change by construction — any change isolates an $x$-path (e.g. the
  residual channel). It is the *attribution control*.
- **Test 2** (predicted concepts — the deployed pipeline): the concept head sees
  the whole image, so $U$ can move *predicted* concepts and change sets without
  label-head leakage. **Zero set-change is not the leak-free signature here** —
  the null is the set-change rate measured at $\rho_{\text{train}}=0$ with the
  same pipeline. This is the **primary** test (settled 2026-11-11).

The shared-$u$ APS pairing (`ccm.interventions.paired_set_change`) guarantees
the measured changes are not randomization artifacts (u-pairing null ≡ 0).

## Beat 5 — The experiment

**Real official 3dshapes images** (`3dshapes.h5`, 480k grid — user direction
2026-11-11: main tables on real data; the procedural generator is the test
fixture). The settled design mapped onto the real grid
([[03 - Experiments/Shapes3D Paired-Nuisance Generator Spec|the spec]]):
floor_hue pinned to one level; $U$ = wall_hue at two palette levels, kept out
of $C$; $C$ = {`object_hue`, `scale`, `shape`, `orientation`}; $Y$ = composite
(`shape`, `scale`-bucket) tuned to 70–90% operating accuracy so conformal sets
are non-trivial; correlation knob
$P(U{=}1\mid Y{=}y) = \tfrac{1-\rho}{2} + \rho\,\mathbb{1}[y \bmod 2]$
implemented by per-class subsampling of the real grid, so marginal $Y = f(C)$
always holds. **U-flip partners are real images**: for every sampled image the
grid contains its same-C/other-wall twin — at $\rho{=}1$ that twin is outside
the training set by construction (a true counterfactual).
Main table: $\rho_{\text{train}} \in \{0,1\}$ × {CBM, CBM+residual,
black-box} at matched $\rho$; stress eval at decorrelated $\rho_{\text{test}}=0$.
Alongside: prediction-flip rate and the nn-leakage v3 audit on the same run.

## Status & guardrails

- **Why the generator had to exist:** the original `Shapes3DDataset` has label =
  concept column 4 — nothing *can* leak by construction — and saturates (100%
  task accuracy, singleton sets at 100% coverage).
- **Guardrail:** never claim a clean bill of health; the test is an alarm, not a
  certificate. Keep the $Y=f(C)$ vs $Y=f(C,U)$ distinction explicit.
- **Out of scope:** everything CUB / joint-uncertainty (that is the CVPR paper).

## 🔗 Related Notes
- [[06 - Papers & Planning/Paper Split - CLeaR vs CVPR|Paper Split]] · [[06 - Papers & Planning/Roadmap and Gates|Roadmap]] (step 3)
- [[01 - Literature & Theory/Leakage One-Sided Claim|Leakage: the One-Sided Claim]]
- [[03 - Experiments/Shapes3D Paired-Nuisance Generator Spec|Generator Spec]]
- [[06 - Papers & Planning/Story - CVPR Collapse Paper|CVPR story]]
