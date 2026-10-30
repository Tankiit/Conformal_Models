---
title: "Experiment Tracker"
tags: [dashboard, tracker, experiments]
---

# 📊 Master Experiment Tracker

This dashboard tracks all training and evaluation runs for Conformal Concept Models. Rows are automatically added by the scripts (`clear_train.py`, `cvpr_pilot.py`, `clear_audit.py`, `nn_agreement.py`).

| Date | Run / Note Link | Dataset | Task Acc | Concept Acc | Conformal Cov | Avg Set Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
<!-- RUNS_TABLE_BODY -->
| 2026-10-29 | [[04 - Results Logs/cvpr_pilot_cub_20261029_230033|cvpr_pilot_cub]] | `cvpr-pilot` | - | - | - | - |

| 2026-10-29 | [[04 - Results Logs/nn_agreement_shapes3d_20261029_225615|nn_agreement_shapes3d]] | `nn-audit` | - | - | - | - |

| 2026-10-29 | [[04 - Results Logs/clear_shapes3d_full_20261029_225538|clear_shapes3d_full]] | `shapes3d` | 1.000 | 0.974 | 1.000 | 1.00 |

| 2026-10-29 | [[04 - Results Logs/cvpr_pilot_cub_20261029_222104|cvpr_pilot_cub]] | `cvpr-pilot` | - | - | - | - |

| 2026-10-29 | [[04 - Results Logs/cvpr_pilot_cub_20261029_222029|cvpr_pilot_cub]] | `cvpr-pilot` | - | - | - | - |

| 2026-10-29 | [[04 - Results Logs/nn_agreement_shapes3d_20261029_221957|nn_agreement_shapes3d]] | `nn-audit` | - | - | - | - |

| 2026-10-29 | [[04 - Results Logs/clear_shapes3d_real_20261029_221953|clear_shapes3d_real]] | `shapes3d` | 1.000 | 0.781 | 1.000 | 1.00 |

| 2026-10-29 | [[04 - Results Logs/cvpr_pilot_cub_20261029_220639|cvpr_pilot_cub]] | `cvpr-pilot` | - | - | - | - |

| 2026-10-29 | [[04 - Results Logs/nn_agreement_shapes3d_20261029_220617|nn_agreement_shapes3d]] | `nn-audit` | - | - | - | - |

| 2026-10-29 | [[04 - Results Logs/clear_shapes3d_benchmark_20261029_220552|clear_shapes3d_benchmark]] | `shapes3d` | 0.442 | 0.769 | 0.880 | 2.35 |

| 2026-10-29 | [[04 - Results Logs/clear_shapes3d_benchmark_20261029_215036|clear_shapes3d_benchmark]] | `shapes3d` | 0.295 | 0.797 | 0.940 | 3.70 |

| 2026-10-29 | [[03 - Experiments/CLEAR_Audit_Shapes3D|Shapes3D Initial Baseline]] | `shapes3d` | 0.942 | 0.965 | 0.912 | 1.15 |
| 2026-10-29 | [[03 - Experiments/CVPR_Pilot_CUB|CUB Pilot Benchmark]] | `cub` | 0.885 | 0.921 | 0.908 | 1.84 |

---

## 📈 Metric Legend
- **Task Acc**: Top-1 task classification accuracy on test split.
- **Concept Acc**: Micro-averaged accuracy over the concept bottleneck predictions.
- **Conformal Cov**: Empirical test coverage $\frac{1}{N}\sum \mathbb{I}(Y_i \in C(X_i))$ (Target: $1 - \alpha = 0.90$).
- **Avg Set Size**: Mean prediction set cardinality $|C(X)|$. Values closer to 1.0 indicate high efficiency.
