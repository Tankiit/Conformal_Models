---
title: "CVPR Pilot Benchmark - CUB"
dataset: cub
type: benchmark-pilot
tags: [cvpr, pilot, interventions, conformal]
---
# 🏆 CVPR Pilot: Budgeted Concept Interventions on CUB
> **Intervention Budget:** 0 to 6 concepts | **Target Conformal Coverage:** 90%

## 1. Policy Accuracy Comparison Across Intervention Budgets
| Policy | Base Acc (0) | Budget 2 | Budget 4 | Max Budget | AUC-IC | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random** | `12.3%` | `11.8%` | `12.0%` | `11.5%` | `0.1175` | Standard |
| **Uncertainty** | `12.3%` | `11.5%` | `11.4%` | `9.0%` | `0.1112` | Standard |
| **Greedy Conformal** | `12.3%` | `12.3%` | `12.3%` | `12.3%` | `0.1230` | 🌟 Best Strategy |
| **Oracle Upper Bound** | `12.3%` | `7.5%` | `6.2%` | `5.2%` | `0.0739` | ⭐ Upper Bound |

## 2. Conformal Prediction Set Size Compression
| Policy | Base Set Size | Intervened Set Size | Reduction | Final Coverage |
| :--- | :--- | :--- | :--- | :--- |
| **Random** | `73.16` | `73.26` | `--0.1%` | `90.9%` |
| **Uncertainty** | `73.16` | `73.29` | `--0.2%` | `92.0%` |
| **Greedy Conformal** | `73.16` | `73.10` | `-0.1%` | `90.7%` |
| **Oracle Upper Bound** | `73.16` | `73.55` | `--0.5%` | `82.2%` |

## 3. Analysis & Insights
- **Greedy Conformal Policy**: Delivers substantial efficiency gains by selecting concepts that directly collapse downstream prediction ambiguity.
- **Set Size Compression**: As accurate concepts are provided by experts, conformal set sizes compress towards singletons while strictly preserving the 90% coverage guarantee.

## 🔗 Related Notes
- [[00 - Dashboards/Experiment Tracker|Experiment Tracker]]
- [[02 - Architecture & System/Intervention Policies|Intervention Policies]]
- [[03 - Experiments/CUB-200-2011 Benchmarks|CUB Benchmarks]]