---
title: "Experiment Tracker"
tags: [dashboard, tracker, experiments]
---

# 📊 Master Experiment Tracker

This dashboard tracks all training and evaluation runs for Conformal Concept Models. Rows are automatically added by the scripts (`clear_train.py`, `cvpr_pilot.py`, `clear_audit.py`, `nn_agreement.py`).

> [!NOTE] Provenance: the canonical numbers are in [[06 - Papers & Planning/Results Snapshot 2026-11-13|Results Snapshot 2026-11-13]] (CUB detail: the `cub-gate` row; Shapes3D paired grid: the 2026-11-11 `clear-paired-pilot` rows). Rows below it are early pilot/synthetic runs — the four `cvpr_pilot_cub` runs are **tagged INVALID** in their logs (undertrained from-scratch heads; base acc 0.12–0.75 vs probe reference 0.877), the two `clear_shapes3d_benchmark` runs are synthetic procedural pilots (superseded), and the two hand-added rows at the bottom are not from logs. Do not cite any row except `cub-gate` and the paired pilots.

| Date | Run / Note Link | Dataset | Task Acc | Concept Acc | Conformal Cov | Avg Set Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
<!-- RUNS_TABLE_BODY -->
| 2026-11-13 | [[04 - Results Logs/Four-Image_2x2_(wall_x_size)_20261113_200807|Four-Image 2x2 (wall x size)]] | `four-image` (ρ=1 cbm, ord) | 0.942* | 0.952 | **0.471** (own-cov) | 0.99 |

| 2026-11-13 | [[04 - Results Logs/CLeaR_Paired_Pilot_clear_paired_shapes3d_real_20261113_193524|CLeaR Paired Pilot clear_paired_shapes3d_real]] | `clear-paired-pilot` (ρ=0/0.5/1 cbm) | 0.900/0.881/0.942 | 0.952 | 0.891/0.895/0.891 | 1.08/1.15/0.99 |

| 2026-11-11 | [[04 - Results Logs/CLeaR_Paired_Pilot_clear_paired_shapes3d_real_20261111_094725|CLeaR Paired Pilot clear_paired_shapes3d_real]] | `clear-paired-pilot` (ρ=0/1, 3 models) | 0.834–1.000 | 0.952 | 0.878–0.905 | 0.88–1.36 |

| 2026-11-11 | [[04 - Results Logs/CLeaR_Paired_Pilot_clear_paired_shapes3d_20261111_092746|CLeaR Paired Pilot clear_paired_shapes3d]] | `clear-paired-pilot` | - | - | - | - |

| 2026-11-06 | [[04 - Results Logs/cub_gate_20261106|cub_gate (check1 + step2)]] | `cub-gate` | 0.872 | 0.952 | 0.901 | 2.63 |
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

*Four-image row: Task Acc is the underlying model's single-render accuracy (ρ=1 cell); the run's own metrics are the 2×2 readouts (own-cov 0.471, size-disc 0.469, scale-fix 0.593 vs hue 0.042, residual 0.000) — see the log. Dose-response row: cov-both collapses 0.891→0.086 at ρ=1; paired-max restores 0.901 at 4.93 labels/set.*

---

## 📈 Metric Legend
- **Task Acc**: Top-1 task classification accuracy on test split.
- **Concept Acc**: Micro-averaged accuracy over the concept bottleneck predictions.
- **Conformal Cov**: Empirical test coverage $\frac{1}{N}\sum \mathbb{I}(Y_i \in C(X_i))$ (Target: $1 - \alpha = 0.90$).
- **Avg Set Size**: Mean prediction set cardinality $|C(X)|$. Values closer to 1.0 indicate high efficiency.
