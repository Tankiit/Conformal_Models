---
title: "Open Decisions"
tags: [planning, decisions, design]
date: 2026-11-06
---

# 🧭 Open Decisions

Design decisions that are still open. Each blocks a specific roadmap step — do not start that step without closing the decision.

| # | Decision | Blocks | Notes |
| :---: | :--- | :--- | :--- |
| **D1** | ✅ **SETTLED (2026-11-11, user sign-off):** $U$ = binarized `wall_hue` (floor_hue stays a fallback if the pilot shows it is not exploitable); $C$ = {`object_hue`, `scale`, `shape`, `orientation`}; $Y$ = **composite** (`shape`, `scale`-bucket) with bucketing tuned by one pilot to 70–90% operating accuracy | CLeaR paired test (Roadmap step 3) | Fixes the generative story of [[01 - Literature & Theory/Leakage One-Sided Claim\|the leakage claim]]. Spec: [[03 - Experiments/Shapes3D Paired-Nuisance Generator Spec\|the spec]] |
| **D2** | ✅ **SETTLED (2026-11-11, user sign-off):** train at $\rho_{\text{train}} \in \{0, 1\}$; **main tables at matched $\rho$** (fix eval regime before calibration), **stress eval at decorrelated $\rho_{\text{test}}=0$** | CLeaR paired test (step 3) | Spec §1 |
| **D3** | ✅ **SETTLED (2026-11-06):** any same-distribution calibration pool; report **no-fill** APS; state the label source. Optional temperature scaling (exact coverage, ~30% smaller sets). See [[04 - Results Logs/cub_gate_20261106\|CUB Gate Run]] | — | Resolved by check 1, differently than hypothesized (fill, not shift) |
| **D4** | Form of the joint concept claim: all right / at most $m$ wrong / expected fraction | CUB joint comparison (step 2) | **Urgent after gate failure**: on denoised labels the all-right claim *is* the label claim (induced dominates 8.5 vs 55+); on raw labels the $(m{+}1)$-th largest is exact but expensive (m=40 @ α=0.10 → 30.6 ambiguous of 312). Pick per setting. Design space: [[01 - Literature & Theory/Joint Concept Coverage\|Joint Concept Coverage]] §2 |
| **D5** | ✅ **SETTLED (2026-11-11, user sign-off):** **Test 2 primary** (whole deployed pipeline — predicted concepts + optional residual), Test 1 as attribution control. Under Test 2 the leak-free null is the **measured set-change rate at $\rho_{\text{train}}=0$** (measure it before the main grid); under Test 1 a strict label head gives exactly zero by construction | CLeaR paper framing (step 3) | See [[01 - Literature & Theory/Leakage One-Sided Claim\|Leakage: the One-Sided Claim]] §2 and [[03 - Experiments/Shapes3D Paired-Nuisance Generator Spec\|the spec]] §2. Headline claim lives in [[06 - Papers & Planning/Story - CLeaR Leakage Paper\|the CLeaR story]] |
| **D6** | Mondrian small-group rule: enforce a **minimum group size at stratification time** (merge small strata into the parent/global pool *before* calibration, rule fixed in advance) vs keep the current global-threshold fallback (now loudly warned + recorded in `fallback_groups`, 2026-11-07) | Any group-conditional coverage claim (CLeaR audit / difficulty strata) | Silent fallback voided the conditional claim for small groups; the warning is a stopgap, the merge rule is the real fix |

---

## 🔗 Related Notes
- [[06 - Papers & Planning/Roadmap and Gates|Roadmap and Gates]] — where each decision lands
- [[06 - Papers & Planning/Paper Split - CLeaR vs CVPR|Paper Split]]
