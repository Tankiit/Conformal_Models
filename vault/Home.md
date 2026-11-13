---
title: "Conformal Concept Models - Research Wiki & Vault"
tags: [conformal-prediction, concept-bottleneck-models, research, wiki]
date: 2026-10-29
---

# 🌌 Conformal Concept Models (CCM) Research Vault

> **Welcome to the Conformal Concept Models (CCM) Research Hub.**
> This vault tracks theoretical foundations, architecture designs, benchmark scorecards, and live automated run logs for distribution-free conformal prediction on Concept Bottleneck Models (CBMs).

## 📌 Program Status & Deadlines (updated 2026-11-11)

| Deadline | Date |
| :--- | :--- |
| ~~CVPR 2027~~ | ❌ **not submitting** (2026-11-13) |
| 🔴 **CLeaR 2027 submission** | **16 Nov 2026, 11:59 pm AoE** (verified 2026-11-11 — no extension) |

The program is now **single-paper: CLeaR 2027 only** (2026-11-13: NOT submitting to CVPR — never registered; collapse/CUB work parked as a follow-up). **State of play (2026-11-13, evening — deadline 16 Nov AoE):** D1–D7 **all settled** (D7 by user sign-off: the causal coverage-cost gate). The CLeaR **step-3 gate PASSED** on the real 3dshapes grid — the wall flip costs the correct label on 82% of ρ=1 scenes; a targeted scale-concept correction restores 59% (control 4%) on the strict CBM and 0% on the residual model; paired-max restores coverage only by giving up size discrimination. Story beats 3–5 rewritten around the causal question + correction result. **Remaining: finish the CLeaR paper — draft at `paper/main.tex`.** Canonical numbers: [[06 - Papers & Planning/Results Snapshot 2026-11-13|Results Snapshot 2026-11-13]]. Stories: [[06 - Papers & Planning/Story - CLeaR Leakage Paper|CLeaR]] · [[06 - Papers & Planning/Story - CVPR Collapse Paper|CVPR]]. Planning: [[06 - Papers & Planning/Roadmap and Gates|Roadmap and Gates]] → [[06 - Papers & Planning/Results Snapshot 2026-11-13|Results Snapshot]] → [[06 - Papers & Planning/Open Checks|Open Checks]] → [[06 - Papers & Planning/Open Decisions|Open Decisions]].

---

## 🧭 Navigation & Maps of Content (MOC)

| Directory | Core Purpose | Direct Links |
| :--- | :--- | :--- |
| **00 - Dashboards** | Live experiment tables, audit scorecards, and pilot leaderboards | [[00 - Dashboards/Experiment Tracker\|Experiment Tracker]]<br>[[00 - Dashboards/Audit Dashboard\|Audit Dashboard]]<br>[[00 - Dashboards/CVPR Benchmark Leaderboard\|CVPR Leaderboard]] |
| **01 - Literature & Theory** | Mathematical formulations and peer-reviewed methodology | [[01 - Literature & Theory/Conformal Prediction Foundations\|Conformal Prediction]]<br>[[01 - Literature & Theory/Concept Bottleneck Models\|CBMs & Bottlenecks]]<br>[[01 - Literature & Theory/CLEAR Methodology\|CLEAR Audit Framework]] |
| **02 - Architecture & System** | Deep dives into pipelines, score functions, and intervention engines | [[02 - Architecture & System/Pipeline Architecture\|Pipeline Architecture]]<br>[[02 - Architecture & System/Nonconformity Scores\|Nonconformity Scores]]<br>[[02 - Architecture & System/Intervention Policies\|Intervention Policies]] |
| **03 - Experiments** | Benchmark definitions for Shapes3D, CUB-200-2011, and Leakage audits | [[03 - Experiments/Shapes3D Benchmarks\|Shapes3D Benchmark]]<br>[[03 - Experiments/CUB-200-2011 Benchmarks\|CUB Benchmark]] |
| **04 - Results Logs** | Live automated notes created per script execution | See latest run reports in `04 - Results Logs/` |
| **05 - Templates** | Reusable templates for logging runs and audits | [[05 - Templates/New Experiment Run\|New Run Template]] |
| **06 - Papers & Planning** | Two-paper program: paper split, results snapshots, open checks/decisions, roadmap & gates | [[06 - Papers & Planning/Paper Split - CLeaR vs CVPR\|Paper Split: CLeaR vs CVPR]]<br>[[06 - Papers & Planning/Results Snapshot 2026-11-13\|Results Snapshot]]<br>[[06 - Papers & Planning/Roadmap and Gates\|Roadmap & Gates]] |

---

## 🔬 Core Research Questions & Objectives

1. **Finite-Sample Validity on Concepts**:
   Can we provide finite-sample, distribution-free marginal coverage guarantees $P(Y \in C(X)) \ge 1 - \alpha$ for downstream tasks while also constructing valid prediction sets $C_j(X)$ for each intermediate concept?

2. **Concept Leakage Auditing (CLEAR)**:
   Does the latent representation $z$ leak task-predictive signals not captured by the human-interpretable concept bottleneck $c$? How do nearest-neighbor agreement metrics quantify this leakage?

3. **Conformal-Guided Interventions**:
   When human or oracle intervention budget is constrained ($k \ll C$), does prioritizing interventions using conformal uncertainty (predictive entropy reduction / set compression) outperform standard uncertainty and random interventions?

---

## 🚀 Quick Commands & Reproducibility

```bash
# Activate conda environment
conda activate torch-multimodal

# 0. Run the CLeaR paired-nuisance grid on REAL 3dshapes images
python scripts/clear_paired.py --config configs/clear_paired_shapes3d_real.yaml

# 1. Train CBM on Shapes3D and calibrate conformal sets
python scripts/clear_train.py --config configs/clear_shapes3d.yaml

# 2. Run CLEAR Reliability & Subgroup Audit
python scripts/clear_audit.py --config configs/clear_shapes3d.yaml

# 3. Run CVPR Pilot Intervention Benchmark
python scripts/cvpr_pilot.py --config configs/cvpr_cub.yaml --budget 6

# 4. Run Nearest-Neighbor Concept Leakage Diagnostic
python scripts/nn_agreement.py --config configs/clear_shapes3d.yaml

# 5. Run test suite
pytest tests/ -v
```

*Note: All script runs automatically append structured Markdown logs and update tables inside this vault.*
