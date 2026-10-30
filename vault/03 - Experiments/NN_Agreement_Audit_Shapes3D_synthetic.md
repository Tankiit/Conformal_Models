---
title: "Nearest Neighbor Agreement Audit - SHAPES3D"
dataset: shapes3d
type: leakage-diagnostic
tags: [nn-agreement, concept-leakage, fidelity]
---
# 🔍 Nearest-Neighbor Concept Leakage Diagnostic: SHAPES3D
> **Audited on:** `400` samples | **Neighborhood Size:** `k=5` | **Assessment:** 🟢 Minimal Leakage

## 1. Leakage & Agreement Scorecard
| Metric | Observed Value | Theoretical Ideal | Status |
| :--- | :--- | :--- | :--- |
| **k-NN Accuracy (Latent $z$)** | `0.5250` | Equal to $c$ | Latent Fidelity |
| **k-NN Accuracy (Concepts $c$)** | `0.5500` | High | Bottleneck Sufficiency |
| **Leakage Gap ($z - c$)** | `+0.0000` | `0.0000` | 🟢 Minimal Leakage |
| **Leakage Ratio ($z / c$)** | `0.9545` | `1.0000` | Multiplicative Bias |
| **Latent vs Concept Agreement** | `70.00%` | `100.0%` | Decision Consistency |

## 2. Interpretation & Conformal Implications
- **Information Bottleneck Integrity**: When the leakage gap is low, downstream conformal sets calibrated on concepts accurately reflect true prediction uncertainty without unmodeled latent side-channels.
- **Intervention Reliability**: High NN agreement guarantees that intervening on human concepts will produce predictable, faithful shifts in model behavior rather than being overridden by leaked latent variables.

## 🔗 Related Notes
- [[00 - Dashboards/Audit Dashboard|Audit Dashboard]]
- [[01 - Literature & Theory/Concept Bottleneck Models|CBM Theory]]
- [[00 - Dashboards/Experiment Tracker|Experiment Tracker]]