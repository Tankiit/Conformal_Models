---
title: "Story - CLeaR Leakage Paper"
tags: [planning, story, clear, leakage, arc]
date: 2026-11-11
---

# 📖 Story: The CLeaR Paper (Leakage / Information Use)

> The narrative arc for the CLeaR arm, written down 2026-11-11; **gate reframed 2026-11-13 (D7 settled)**.
> Companion arc: [[06 - Papers & Planning/Story - CVPR Collapse Paper|CVPR story]].
*Question this paper owns:* **does the predictor use information outside the
declared concepts — and when a nuisance intervention costs it the correct
answer, can a targeted concept correction restore it?** (Paper Split §1.)

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

## Beat 3 — The causal question (the gate, D7 settled 2026-11-13)

> **If we change only the wall colour, does the model still include the correct
> shape-and-scale label in its prediction set?**

The original gate ("sets detect what probabilities miss") is **retired**: APS
sets are computed from those probabilities, and the pilot's probability-shift
baseline separates as strongly as the sets. The informative question is the
**cost** — which probability movements cross the calibrated boundary and what
that does to coverage. That question is causal (known intervention, fixed
outcome), falsifiable, and has a guaranteed remedy to measure against.

Measured on the real grid (dose-response, 2026-11-13): at $\rho{=}1$ ordinary
single-render coverage is a mirage — **0.891 per render, but 0.086 under both
walls**: flipping the wall removes the correct label on **82%** of scenes, with
the same-shape/wrong-scale substitute entering 66% of those losses (at
$\rho{=}1$ the wall *is* the scale bucket). At $\rho{=}0.5$ the CBM does not
take the half-dose bait (1.8% loss). The counterfactual-invariance connection:
the decision under study is membership in a calibrated set, not an argmax.

## Beat 4 — The correction result (the paper's positive contribution)

The **four-image test** (wall {A,B} × size across the bucket boundary; all four
renders are real grid images; model, threshold, and the APS draw shared across
each group) separates three instruments:

1. **Paired-max calibration** (max-score threshold over a scene's two renders,
   $P[Y \in S(x_a) \cap S(x_b)] \ge 1-\alpha$): restores coverage-under-both-walls
   to 0.901 — at the guarantee — but at **4.93 labels/set** (vs 1.16 at
   $\rho{=}0$) and size discrimination collapses to 0.157 with 79% ambiguous
   sets. Calibration restores coverage **by admitting both sizes**.
2. **Targeted concept correction**: on a wall-induced failure, replace *only*
   the predicted **scale** concept with ground truth — the correct label returns
   on **59%** of failures. Controls: replacing the irrelevant **object-hue**
   concept restores 4%; clamping all concepts (Test 1) restores 77%. One
   semantically-relevant edit recovers most of the full clamp's power.
3. **The boundary**: the same corrections do **nothing (0%)** on the residual
   CBM — wall information flows through the residual channel, outside the
   bottleneck, and no concept edit can reach it. Test 1 attribution explains
   why: strict CBM T1 = 0 (its leakage enters through the concept head, which
   the correction fixes); residual T1 = 0.93 (the x-path bypasses the fix).

**Headline:** *a targeted concept correction restores coverage lost under a
nuisance intervention while preserving the model's ability to distinguish a
genuine change in object size* — paired calibration cannot do both at once.
The shortcut itself is a **confident error, not ambiguity** (size
discrimination 0.894 → 0.469, 35% confident errors, 0.2% ambiguous).

## Beat 5 — The deployed-pipeline subtlety and the experiment

Test variants (was **D5**, settled 2026-11-11): **Test 2 (predicted concepts,
the deployed pipeline) is primary** — the concept head sees the whole image, so
$U$ moves *predicted* concepts; the null is the $\rho_{\text{train}}{=}0$
measured rate. **Test 1** (ground-truth concepts fixed) is the attribution
control: exactly zero for a strict label head by construction; nonzero means an
$x$-path. The shared-$u$ APS pairing (`ccm.interventions.paired_set_change`)
guarantees measured changes are not randomization artifacts (u-pairing null ≡ 0,
verified exactly 0 in every cell).

**Real official 3dshapes images** (`3dshapes.h5`, 480k grid — user direction
2026-11-11: main tables on real data; the procedural generator is the test
fixture). The settled design mapped onto the real grid
([[03 - Experiments/Shapes3D Paired-Nuisance Generator Spec|the spec]]):
floor_hue pinned to one level; $U$ = wall_hue at two palette levels, kept out
of $C$; $C$ = {`object_hue`, `scale`, `shape`, `orientation`}; $Y$ = composite
(`shape`, `scale`-bucket) tuned to 70–90% operating accuracy; correlation knob
$P(U{=}1\mid Y{=}y) = \tfrac{1-\rho}{2} + \rho\,\mathbb{1}[y \bmod 2]$
implemented by **scene-level** per-class subsampling of the real grid (one item
per scene; partners never cross folds; matched scenes and split across doses,
2026-11-13), so marginal $Y = f(C)$ always holds. **U-flip partners are real
images**; at $\rho{=}1$ the partner is outside the training set by construction.
Tables: dose-response $\rho_{\text{train}} \in \{0, 0.5, 1\}$ (ordinary vs
paired-max, same fitted model) and the four-image 2×2 grid
$\rho \in \{0,1\}$ × {CBM, CBM+residual}; stress eval at decorrelated
$\rho_{\text{test}}=0$; nn-leakage v3 audit alongside (uninformative here —
reported as such). Run logs: [[04 - Results Logs/CLeaR_Paired_Pilot_clear_paired_shapes3d_real_20261113_193524|dose-response]],
[[04 - Results Logs/Four-Image_2x2_(wall_x_size)_20261113_200807|four-image]].

## Status & guardrails

- **Why the generator had to exist:** the original `Shapes3DDataset` has label =
  concept column 4 — nothing *can* leak by construction — and saturates (100%
  task accuracy, singleton sets at 100% coverage).
- **Guardrail:** never claim a clean bill of health; the test is an alarm, not a
  certificate. Keep the $Y=f(C)$ vs $Y=f(C,U)$ distinction explicit.
- **Guardrail (D7):** never frame the contribution as "sets detect what
  probabilities miss" — APS sets are functions of those probabilities. The
  contribution is the **causal cost** of the nuisance intervention and the
  **correction instrument**; the residual-model boundary (correction does
  nothing) is a finding, not a failure to hide.
- **Out of scope:** everything CUB / joint-uncertainty (that is the CVPR paper).

## 🔗 Related Notes
- [[06 - Papers & Planning/Paper Split - CLeaR vs CVPR|Paper Split]] · [[06 - Papers & Planning/Roadmap and Gates|Roadmap]] (step 3)
- [[01 - Literature & Theory/Leakage One-Sided Claim|Leakage: the One-Sided Claim]]
- [[03 - Experiments/Shapes3D Paired-Nuisance Generator Spec|Generator Spec]]
- [[06 - Papers & Planning/Story - CVPR Collapse Paper|CVPR story]]
