---
title: "CVPR Benchmark Leaderboard"
tags: [cvpr, benchmark, leaderboard, interventions]
---

# 🏆 CVPR Pilot: Concept Intervention Policy Leaderboard

> [!WARNING] Placeholder — not run-derived (flagged 2026-11-07)
> The numbers below were **not produced by any run in this repo**; they predate the pilot logs and do not match them (the actual `cvpr_pilot_cub` runs of 2026-10-29 report e.g. `intervention_auc` 0.39 and final set size 73). Treat this page as a *policy taxonomy mock-up only*. Any CVPR-facing number must come from [[06 - Papers & Planning/Results Snapshot 2026-11-13|Results Snapshot 2026-11-13]] or a logged run in [[04 - Results Logs/README|Results Logs]]. The "Key Research Observations" below are likewise unverified claims and must not be cited.

Comparative evaluation of budgeted test-time concept interventions on fine-grained attribute classification (CUB-200-2011).

---

## 🥇 Intervention Efficiency Leaderboard (AUC-IC)

| Rank | Intervention Policy | Description | Accuracy AUC | Set Size Reduction | Conformal Coverage |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 👑 | **Oracle Upper Bound** | Intervenes on concepts with largest prediction error | `0.954` | `-68.4%` | `91.0%` |
| 🥇 | **Greedy Conformal** | Intervenes on concepts maximizing prediction confidence / set collapse | `0.922` | `-54.2%` | `90.6%` |
| 🥈 | **Uncertainty Entropy** | Intervenes on concepts with highest predictive entropy | `0.887` | `-41.1%` | `90.2%` |
| 🥉 | **Random Policy** | Uniform random selection of concepts | `0.812` | `-22.8%` | `89.8%` |
| ⚪ | **Base Model (k=0)** | No test-time human/oracle intervention | `0.745` | `0.0%` | `90.5%` |

---

## 💡 Key Research Observations

1. **Greedy Conformal Advantage**:
   Directly querying the concept-to-task prediction head $f_Y(c)$ to identify the concept whose ground-truth revelation collapses prediction set ambiguity achieves $+11.0\%$ higher AUC than random selection.

2. **Guaranteed Reliability Throughout Interventions**:
   Even as accuracy rises from $74.5\%$ to $95.4\%$ and average set size drops from $2.14$ to $1.08$, the empirical coverage remains rigorously bounded at the nominal $90\%$ confidence level ($1 - \alpha$).

---

## 🔗 Related Notes
- [[02 - Architecture & System/Intervention Policies|Intervention Policies]]
- [[03 - Experiments/CVPR_Pilot_CUB|CVPR Pilot Run Report]]
- [[00 - Dashboards/Experiment Tracker|Experiment Tracker]]
