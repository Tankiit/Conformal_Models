---
title: "Results Snapshot 2026-11-13"
tags: [results, cub, shapes3d, paired, snapshot, planning, canonical]
date: 2026-11-13
---

# 📸 Results Snapshot — 2026-11-13 (CANONICAL)

> **Provenance:** supersedes [[06 - Papers & Planning/Results Snapshot 2026-11-06|Results Snapshot 2026-11-06]] (kept as archive; its CUB numbers are unchanged and still the source of record for the CVPR arm). Raw per-run logs: [[04 - Results Logs/README|Results Logs]]. Test suite: **46/46 passing** (`torch-multimodal` env, 2026-11-13). Both deadlines are **16 Nov 2026 AoE — 3 days out**.

---

## 1. CUB (CVPR arm) — unchanged since the gate run

All numbers live in [[04 - Results Logs/cub_gate_20261106|cub_gate_20261106]] and are re-stated in [[06 - Papers & Planning/Story - CVPR Collapse Paper|the CVPR story]]: the collapse (induced 8.5 ambiguous vs 55.3/57.8 at matched 0.90 joint; induced joint coverage ≡ label coverage, 0.900 vs 0.898) and the raw-312 $(m{+}1)$ frontier (306.5 → 104.2 → 30.6 ambiguous at $m = 0/16/40$, guarantee exact at every $m$). No new CUB runs since. The gate machinery is now **in the library** (`ccm.conformal.m_plus_one_score`, `conformal_quantile_per_column`, `induced_concept_sets`, `calibrate_concepts(joint_m=…)`).

**Open robustness item:** one raw-annotation second dataset (e.g. CelebA-style per-image attributes) to show the frontier generalizes — the largest remaining review risk for the CVPR arm.

## 2. Shapes3D paired-nuisance grid (CLeaR arm) — first real-data grid, 2026-11-11

[[04 - Results Logs/CLeaR_Paired_Pilot_clear_paired_shapes3d_real_20261111_094725|Real 3dshapes run]] (480k official HDF5, $U$ = binarized `wall_hue`, $Y$ = shape × scale-bucket composite, $n = 4800$, matched-$\rho$ eval per D2, Test 2 primary per D5):

| Model | $\rho$ | Acc | Cov | Set size | T2 set-change (vs own $\rho{=}0$ null) | T1 set-change |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| cbm | 0 | 0.834 | 0.905 | 1.36 | **0.056** (this IS the null) | 0.000 |
| cbm | 1 | 0.997 | 0.901 | 0.97 | **0.952** | 0.000 |
| residual | 0 | 1.000 | 0.890 | 0.89 | 0.006 (null) | 0.006 |
| residual | 1 | 1.000 | 0.891 | 0.89 | **0.928** | **0.928** |
| blackbox | 0 | 1.000 | 0.884 | 0.89 | 0.002 (null) | — |
| blackbox | 1 | 1.000 | 0.878 | 0.88 | **0.915** | — |

- **Machinery validated:** sanity $(x,x)$ null **exactly 0.000** in all 6 cells (shared-$u$ APS holds); Test-1 attribution textbook — plain CBM T1 = 0 everywhere (its $\rho{=}1$ leakage enters through the concept head), residual T1 = 0.006 → 0.928 (the residual channel *is* the x-path).
- **De-saturation achieved where it matters:** the $\rho{=}0$ CBM cell runs at 0.834 acc with sets of 1.36 — non-trivial conformal sets. $\rho{=}1$ cells saturate by design ($U$ = parity of $Y$ is a perfect feature); null-relative claims are unaffected.
- **nn audit (v3):** `gap_vs_null_z` ≤ 0.56 in every cell — within the null band throughout the pilot.
- **Step-3 gate still OPEN:** at $\rho{=}1$ the probability-shift baseline separates (L1 1.4–2.0) as strongly as the sets, so "sets detect what probabilities miss" is not yet demonstrated. Encouraging detail: cbm @ $\rho{=}1$ set-change 0.952 > flip rate 0.860 — ~9% of pairs change the set while the argmax doesn't flip. **Next run: $\rho = 0.5$ dose-response** (partial-shortcut regime) and/or a concept-loss-weight sweep to soften the concept head.

## 2b. Dose-response + four-image 2×2 (2026-11-13, later same day)

**Dose-response** ([[04 - Results Logs/CLeaR_Paired_Pilot_clear_paired_shapes3d_real_20261113_193524|run log]]; plain CBM, ordinary vs paired-max calibration, matched scenes/seeds across doses):

| $\rho$ | cov | size | dir-loss | cov-both (ord) | subst\|lost | cov-both (paired) | paired size |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 0.891 | 1.08 | 0.015 | 0.891 | 0.00 | 0.913 | 1.16 |
| 0.5 | 0.895 | 1.15 | 0.018 | 0.877 | 0.00 | 0.905 | 1.24 |
| 1 | 0.891 | 0.99 | **0.819** | **0.086** | 0.656 | **0.901** | **4.93** |

At $\rho{=}1$ the wall flip removes the correct label on **82%** of scenes (with the same-shape/wrong-scale substitute entering 66% of losses); **paired-max restores cov-both to 0.901 — at the guarantee — but at 4.93 labels/set** (~4× the $\rho{=}0$ cost): restoration by calibration = admitting both sizes. $\rho{=}0.5$ is mild (dir-loss 1.8%) — the CBM does not take the half-dose bait.

**Four-image 2×2** (wall {A,B} × size across the bucket boundary; [[04 - Results Logs/Four-Image_2x2_(wall_x_size)_20261113_200807|run log]]; 400 held-out groups; model/threshold/shared-u fixed per group):

| model | $\rho$ | thr | own-cov | nuis loss A→B / B→A | size-disc | conf-err | scale-fix | hue-fix | clamp-fix |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| cbm | 0 | ord | 0.894 | .018 / .052 | **0.894** | 0.000 | 0.154* | 0.385* | 0.615* |
| cbm | 1 | ord | 0.471 | .905 / .817 | **0.469** | **0.351** | **0.593** | 0.042 | 0.769 |
| cbm | 1 | paired | 0.946 | .175 / .040 | **0.157** | 0.054 | — | — | — |
| residual | 1 | ord | 0.474 | .842 / .873 | 0.467 | 0.329 | **0.000** | 0.000 | 0.000 |

*$\rho{=}0$ correction rates are n=13 noise. The three findings, exactly as the 2026-11-13 review predicted the design could show:

1. **The shortcut is a confident error, not ambiguity**: at $\rho{=}1$ size-discrimination collapses (0.894 → 0.469) with 35% confident errors and only 0.2% "both sizes" sets — the model reads the size off the wall.
2. **A single relevant concept correction repairs the strict CBM**: replacing only the predicted *scale* concept restores the lost label on **59%** of wall-induced failures (irrelevant hue correction: 4%; full clamp: 77%) — most of the clamp's power from one targeted edit.
3. **The boundary**: the SAME corrections do nothing (0.000) on the residual model — wall information flows through the residual channel, outside the bottleneck, and no concept edit can reach it. And paired-max calibration restores coverage (0.892 cov-both) only by destroying size discrimination (0.157). **Targeted correction is the only instrument that restores coverage while preserving the size readout.**

**D7 SETTLED (2026-11-13, user sign-off):** the coverage-cost framing is adopted; the CLeaR story's beats 3–5 were rewritten around the causal question and the correction result ("sets detect what probabilities miss" retired as the gate). Step-3 gate: **passed**.

## 3. Repo / code state (review rounds 2026-11-07, commits `994b9f2` → `0267949`)

- `fill_empty` defaults to **False** everywhere (D3); intervention sweep reports no-fill.
- Mondrian small-group fallback warns + records `fallback_groups` (stopgap; merge rule = **D6, still open**); `assign_strata` refuses to fit on its input.
- `nn_leakage_score` v3: signed gap, stratified folds, fold-fitted scaling, **embedding-null calibration** (`gap_vs_null_z`, resolution-floored denominator for saturated regimes). The v2 single-projection subtraction reduced to `acc_proj − acc_c` and was not a null — documented in the docstring.
- `paired_set_change` + `APSScore(uniforms=…)` implement the shared-$u$ paired evaluation (Test 1 / Test 2 variants).
- **Invalidated logs** (do not cite): 4 × `cvpr_pilot_cub` (undertrained from-scratch, tagged INVALID), 2 × `clear_shapes3d_benchmark` (synthetic, superseded), 3 × `nn_agreement` (saturated + sign-clip caveat, pre-v1-fix). The [[00 - Dashboards/CVPR Benchmark Leaderboard|CVPR leaderboard]] is banner-flagged as a placeholder mock-up.

## 4. Open items this snapshot leaves

1. **Step-3 gate: ✅ PASSED under D7 (settled 2026-11-13)** — §2b is the evidence. Remaining for CLeaR: **write the paper** (~3 days).
2. **MPI3D-real as the next dataset (feasibility checked 2026-11-13):** 460,800 photos, 64×64, factors include **background colour (3)**, camera height (3), object size (2), shape (6) — maps directly onto this design ($U$ = background; the third background enables the calibrate-on-two / test-on-held-out-third transfer question). Official rr-learning download link is broken; working mirror: `mpi3d_real.npz` ≈ **12.8 GB** at [huggingface.co/datasets/waleedgondal/mpi3d](https://huggingface.co/datasets/waleedgondal/mpi3d). One-time download; the Shapes3D machinery (scene sampling, paired/four-image eval, `calibrate_paired`) ports directly.
3. **CVPR registration status unrecorded** — confirm before investing further in the CVPR arm; if unregistered, CLeaR is the only live 16 Nov target.
4. **D6** (Mondrian merge rule) — open; blocks any group-conditional coverage claim.
5. CVPR second-dataset robustness run (§1).
6. Paper writing: both stories 3 days from deadline; CLeaR story beat 3 must be rewritten if D7 is accepted. EditCLEVR stays follow-up work (question-selective response to edits).

---

## 🔗 Related Notes
- [[06 - Papers & Planning/Results Snapshot 2026-11-06|Previous snapshot (2026-11-06, archive)]]
- [[06 - Papers & Planning/Story - CLeaR Leakage Paper|CLeaR story]] · [[06 - Papers & Planning/Story - CVPR Collapse Paper|CVPR story]]
- [[06 - Papers & Planning/Roadmap and Gates|Roadmap and Gates]] · [[06 - Papers & Planning/Open Decisions|Open Decisions]]
