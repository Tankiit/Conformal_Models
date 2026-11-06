---
title: "Open Decisions"
tags: [planning, decisions, design]
date: 2026-11-06
---

# 🧭 Open Decisions

Design decisions that are still open. Each blocks a specific roadmap step — do not start that step without closing the decision.

| # | Decision | Blocks | Notes |
| :---: | :--- | :--- | :--- |
| **D1** | Which background hue is the Shapes3D nuisance, and what rule defines $Y = f(C)$? | CLeaR paired test (Roadmap step 3) | Fixes the generative story of [[01 - Literature & Theory/Leakage One-Sided Claim\|the leakage claim]] |
| **D2** | Should calibration and test share a **decorrelated** $U$, or follow the **training correlation**? | CLeaR paired test (step 3) | Two regimes to run regardless: training correlation 0 and 1 |
| **D3** | ✅ **SETTLED (2026-11-06):** any same-distribution calibration pool; report **no-fill** APS; state the label source. Optional temperature scaling (exact coverage, ~30% smaller sets). See [[04 - Results Logs/cub_gate_20261106\|CUB Gate Run]] | — | Resolved by check 1, differently than hypothesized (fill, not shift) |
| **D4** | Form of the joint concept claim: all right / at most $m$ wrong / expected fraction | CUB joint comparison (step 2) | **Urgent after gate failure**: on denoised labels the all-right claim *is* the label claim (induced dominates 8.5 vs 55+); on raw labels the $(m{+}1)$-th largest is exact but expensive (m=40 @ α=0.10 → 30.6 ambiguous of 312). Pick per setting. Design space: [[01 - Literature & Theory/Joint Concept Coverage\|Joint Concept Coverage]] §2 |
| **D5** | Which CLeaR test is **primary**: ground-truth-concepts-fixed (Test 1) vs predicted-concepts-fixed (Test 2) | CLeaR paper framing (step 3) | Test 2 applies to residual models only — see [[01 - Literature & Theory/Leakage One-Sided Claim\|Leakage: the One-Sided Claim]] §2 |

---

## 🔗 Related Notes
- [[06 - Papers & Planning/Roadmap and Gates|Roadmap and Gates]] — where each decision lands
- [[06 - Papers & Planning/Paper Split - CLeaR vs CVPR|Paper Split]]
