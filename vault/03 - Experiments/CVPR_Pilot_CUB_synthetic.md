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
| **Random** | `74.7%` | `76.2%` | `79.7%` | `76.6%` | `0.7703` | Standard |
| **Uncertainty** | `74.7%` | `78.1%` | `79.4%` | `80.3%` | `0.7833` | Standard |
| **Greedy Conformal** | `74.7%` | `76.2%` | `76.2%` | `76.2%` | `0.7607` | 🌟 Best Strategy |
| **Oracle Upper Bound** | `74.7%` | `75.6%` | `75.6%` | `74.4%` | `0.7492` | ⭐ Upper Bound |

## 2. Conformal Prediction Set Size Compression
| Policy | Base Set Size | Intervened Set Size | Reduction | Final Coverage |
| :--- | :--- | :--- | :--- | :--- |
| **Random** | `2.01` | `2.10` | `--4.2%` | `94.4%` |
| **Uncertainty** | `2.03` | `2.37` | `--16.4%` | `98.1%` |
| **Greedy Conformal** | `2.04` | `1.91` | `-6.4%` | `87.5%` |
| **Oracle Upper Bound** | `2.02` | `2.65` | `--30.7%` | `96.9%` |

## 3. Analysis & Insights
- **Greedy Conformal Policy**: Delivers substantial efficiency gains by selecting concepts that directly collapse downstream prediction ambiguity.
- **Set Size Compression**: As accurate concepts are provided by experts, conformal set sizes compress towards singletons while strictly preserving the 90% coverage guarantee.

## 🔗 Related Notes
- [[00 - Dashboards/Experiment Tracker|Experiment Tracker]]
- [[02 - Architecture & System/Intervention Policies|Intervention Policies]]
- [[03 - Experiments/CUB-200-2011 Benchmarks|CUB Benchmarks]]