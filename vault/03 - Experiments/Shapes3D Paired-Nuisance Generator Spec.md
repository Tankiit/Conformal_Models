---
title: "Shapes3D Paired-Nuisance Generator Spec"
tags: [shapes3d, clear, leakage, spec, draft, d1, d2]
date: 2026-11-07
status: DRAFT — proposes answers to D1/D2; not settled until the user signs off
---

# 🧪 Shapes3D Paired-Nuisance Generator — Spec (DRAFT)

> **Why this exists:** the current `Shapes3DDataset` cannot run the CLeaR paired test
> (Roadmap step 3). The label **is** concept column 4 (`labels = shapes`; `concepts[:,4] = shapes/3`),
> so nothing can leak by construction; there is no hidden nuisance $U$, no $Y=f(C)$ rule,
> no train-time correlation knob, and iid `random_split`. Accuracy saturates at 100%,
> making every gap diagnostic uninformative ([[06 - Papers & Planning/Results Snapshot 2026-11-06|Results Snapshot]] §1).

## 1. Design (proposal)

| Element | Proposal | Decision |
| :--- | :--- | :--- |
| Declared concepts $C$ | 4 of the 6 factors: `object_hue`, `scale`, `shape`, `orientation` | — |
| Hidden nuisance $U$ | `wall_hue` binarized $\{0,1\}$ (two wall palettes); **not** in $C$ | closes **D1** (alt: `floor_hue`) |
| Task rule $Y = f(C)$ | $Y$ = `shape` class (4-way); $f$ defined on $C$ alone, $U \perp Y \mid C$ **at generation** | closes **D1** |
| Correlation knob $\rho$ | $P(U{=}1 \mid Y{=}y) = \frac{1-\rho}{2} + \rho\cdot\mathbb{1}[y \bmod 2]$; $\rho=0$ independent, $\rho=1$ deterministic shortcut (U = parity of Y) | closes **D2** |
| $\rho$ regimes | train at $\rho_{\text{train}} \in \{0, 1\}$ (optionally 0.5); **fix the eval regime before calibration** — proposal: main tables at matched $\rho$, stress eval at decorrelated $\rho_{\text{test}}=0$ | closes **D2** |
| Rendering | wall color = function of $U$; everything else rendered from $C$ exactly as now | — |
| De-saturation | composite rule option: $Y$ = (`shape`, `scale`-bucket) → 8–12 classes, target operating accuracy 70–90% so conformal sets are non-trivial | open (recommend composite) |
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

**Recommendation (D5):** Test 2 primary — the CLeaR claim is about the *deployed
predictor* (predicted concepts + optional residual channel), which is where
outside-concept information can actually be used; Test 1 as the attribution control
that separates label-head leakage from concept-head drift.

The step-3 **gate** is unchanged either way: a model exploiting $U$ (e.g. the CBM
**with residual channel** at $\rho_{\text{train}}=1$) should show set changes beyond
its $\rho_{\text{train}}=0$ null **even when the marginal probability of $Y$ barely
moves** — the set-based test detects dependence that probability-sensitivity misses.

## 3. Implementation sketch

- Extend `Shapes3DDataset.__init__` with `declared_concepts`, `u_source="wall_hue"`,
  `rho_train`, `label_rule`; return `(x, c, y, u)` (4-tuple) with $U$ kept out of `c`.
- Paired eval: `ccm.interventions.paired_set_change` (exists since 2026-11-07:
  shared-$u$ APS via `APSScore(uniforms=...)`, both Test 1 / Test 2 variants).
- Config: `rho_train: {0, 1}` × `model: {cbm, cbm+residual, black-box}` — the 2×3 grid
  behind the CLeaR paper's main table.

## 4. What this intentionally does NOT decide

- Whether `floor_hue` is a better $U$ than `wall_hue` (rendering prominence matters for
  how exploitable the shortcut is; needs one pilot).
- D5 **recommendation recorded above** (Test 2 primary), but the final call shapes the
  CLeaR paper's headline claim — user sign-off required; see Open Decisions.

## 🔗 Related Notes
- [[01 - Literature & Theory/Leakage One-Sided Claim|Leakage: the One-Sided Claim]]
- [[06 - Papers & Planning/Roadmap and Gates|Roadmap and Gates]] (step 3)
- [[06 - Papers & Planning/Open Decisions|Open Decisions]] (D1, D2)
