---
title: "Shapes3D Paired-Nuisance Generator Spec"
tags: [shapes3d, clear, leakage, spec, draft, d1, d2]
date: 2026-11-07
status: SETTLED 2026-11-11 (user sign-off) — D1, D2, D5 closed. Still open within the spec: the scale-bucket count (pilot-tunable), and the floor_hue fallback trigger.
---

# 🧪 Shapes3D Paired-Nuisance Generator — Spec (SETTLED)

> **Why this exists:** the current `Shapes3DDataset` cannot run the CLeaR paired test
> (Roadmap step 3). The label **is** concept column 4 (`labels = shapes`; `concepts[:,4] = shapes/3`),
> so nothing can leak by construction; there is no hidden nuisance $U$, no $Y=f(C)$ rule,
> no train-time correlation knob, and iid `random_split`. Accuracy saturates at 100%,
> making every gap diagnostic uninformative ([[06 - Papers & Planning/Results Snapshot 2026-11-06|Results Snapshot]] §1).

## 1. Design (proposal)

| Element | Proposal | Decision |
| :--- | :--- | :--- |
| Declared concepts $C$ | 4 of the 6 factors: `object_hue`, `scale`, `shape`, `orientation` | ✅ settled (2026-11-11) |
| Hidden nuisance $U$ | `wall_hue` binarized $\{0,1\}$ (two wall palettes); **not** in $C$ | ✅ **wall_hue** (D1, 2026-11-11); `floor_hue` = fallback if pilot shows non-exploitable |
| Task rule $Y = f(C)$ | $Y$ = `shape` class (4-way); $f$ defined on $C$ alone, $U \perp Y \mid C$ **at generation** | ✅ superseded by composite (below) |
| Correlation knob $\rho$ | $P(U{=}1 \mid Y{=}y) = \frac{1-\rho}{2} + \rho\cdot\mathbb{1}[y \bmod 2]$; $\rho=0$ independent, $\rho=1$ deterministic shortcut (U = parity of Y) | ✅ settled (D1) |
| $\rho$ regimes | train at $\rho_{\text{train}} \in \{0, 1\}$ (optionally 0.5); fix the eval regime before calibration — main tables at matched $\rho$, stress eval at decorrelated $\rho_{\text{test}}=0$ | ✅ settled (D2, 2026-11-11) |
| Rendering | **REAL data primary (user direction 2026-11-11):** official DeepMind 3dshapes grid (`3dshapes.h5`, 480k images), floor_hue pinned to one level, U = wall_hue at two palette levels, ρ by per-class subsampling of the real grid, U-flip partners = the real image with the same C and other wall level (`ccm.data.shapes3d_real.RealShapes3DPairedDataset`). The procedural renderer (`ccm.data.shapes3d` paired mode) is demoted to unit-test fixture / smoke runs | ✅ settled (2026-11-11) |
| De-saturation | composite rule: $Y$ = (`shape`, `scale`-bucket) → 8–12 classes, target operating accuracy 70–90% so conformal sets are non-trivial | ✅ **composite** (D1, 2026-11-11); bucket count pilot-tunable |
| Splits | same-distribution cal/test; **paired sampler**: $(x, x')$ identical $C$, flipped $U$, re-rendered deterministically (procedural generator makes this cheap) | — |

## 2. The paired test (what gets measured)

For each test pair $(x, x')$ with identical $C$ and flipped $U$ — scored through
`ccm.interventions.paired_set_change`, which **holds the APS randomization $u$ fixed
within a pair** (so set changes cannot be randomization artifacts; the $u$-pairing
null is exactly zero — verify with $(x, x)$ pairs before the main grid):

1. **Set-change rate**: fraction of pairs where the conformal label set changes $C(x) \neq C(x')$.
2. **Prediction-flip rate**: fraction where the argmax flips.
3. **nn-leakage audit (v3)** on the same run: signed gap judged against the null band
   of $z$ re-synthesized from $c$ alone (`ccm.metrics.nn_leakage_score`, 2026-11-07).

**What "leakage-free" reads as — depends on the variant (this is D5, not orthogonal):**

- **Test 1, ground-truth-concepts-fixed** (`use_gt_concepts=True`; both members scored
  through the label head with $c$ fixed to truth via the intervention interface): a
  strict label head $f(c)$ gives **exactly zero set-change by construction**; any
  change reveals an $x$-path into the prediction (e.g. the hybrid residual channel).
  Isolates the label head.
- **Test 2, predicted concepts** (`use_gt_concepts=False`; the whole deployed
  pipeline): the concept head sees the whole image, wall colour included, so $U$ can
  move the *predicted* concepts and change sets **without label-head leakage**. Zero
  set-change is **not** the leak-free signature here — the null is the set-change
  rate measured at $\rho_{\text{train}} = 0$ with the same pipeline (no shortcut
  available). Measure that null before the main grid.

**✅ DECIDED (D5, 2026-11-11):** Test 2 primary — the CLeaR claim is about the *deployed
predictor* (predicted concepts + optional residual channel), which is where
outside-concept information can actually be used; Test 1 is the attribution control
that separates label-head leakage from concept-head drift.
The $\rho_{\text{train}}=0$ null for Test 2 **must be measured before the main
grid** (it is the reference line every $\rho{=}1$ number is compared against).

The step-3 gate under **D7 (settled 2026-11-13, user sign-off)**: the **causal
coverage cost** of the $U$-flip — does the correct label stay in the set? — and
whether a **targeted concept correction** restores it (paired-max calibration as
the guaranteed-but-blunt alternative). The old "sets detect what
probability-sensitivity misses" framing is **retired** (APS sets are functions
of the probabilities); set-change rates remain as descriptive texture only.

## 3. Implementation status (2026-11-11)

- ✅ Real data: `ccm.data.shapes3d_real.RealShapes3DPairedDataset` +
  `get_real_paired_shapes3d_loaders` — grid subsampling ρ, real U-flip partners,
  exact per-class parity counts (classes shrink, never topped up, if a pool runs
  short). Pilot: `scripts/clear_paired.py --config configs/clear_paired_shapes3d_real.yaml`.
- ✅ Fixture/procedural: `Shapes3DDataset` paired mode (`u_source`, `rho`) +
  `get_paired_shapes3d_loaders` — same item interface; `configs/clear_paired_shapes3d.yaml`.
- Paired eval: `ccm.interventions.paired_set_change` (shared-$u$ APS via
  `APSScore(uniforms=...)`, both Test 1 / Test 2 variants).
- Grid: `rho_train: {0, 1}` × `model: {cbm, cbm+residual, black-box}` — the 2×3 grid
  behind the CLeaR paper's main table.

## 4. What remains open (post-sign-off)

- The **scale-bucket count** in the composite rule (drives operating accuracy).
  **Real-data pilot 2026-11-11**
  ([[04 - Results Logs/CLeaR_Paired_Pilot_clear_paired_shapes3d_real_20261111_094725|real run log]];
  procedural smoke superseded): buckets=2 on real 3dshapes puts the ρ=0 CBM at
  **acc 0.834, set size 1.36** — inside the 70–90% target with non-trivial sets.
  ρ=1 cells saturate (≥0.997) by design (U is a perfect feature). Residual and
  black-box saturate even at ρ=0 (they bypass the bottleneck's discretized
  concept head). Buckets likely stay at 2.
- The **floor_hue fallback trigger**: if the pilot shows wall_hue is not
  exploitable (set-change null at $\rho{=}1$ indistinguishable from
  $\rho{=}0$), switch $U$ to binarized `floor_hue` and re-pilot.
- D5 is **closed** (Test 2 primary) — see §2.

## 🔗 Related Notes
- [[01 - Literature & Theory/Leakage One-Sided Claim|Leakage: the One-Sided Claim]]
- [[06 - Papers & Planning/Roadmap and Gates|Roadmap and Gates]] (step 3)
- [[06 - Papers & Planning/Open Decisions|Open Decisions]] (D1, D2)
