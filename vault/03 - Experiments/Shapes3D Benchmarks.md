---
title: "Shapes3D Benchmarks"
tags: [shapes3d, benchmarks, disentanglement, clear]
---

# 🧊 Shapes3D Benchmark Specification

The 3D Shapes dataset is a canonical synthetic benchmark used to assess concept learning, disentanglement, and conformal reliability.

---

## 🎯 Dataset Attributes & Factors

| Factor | Cardinality / Range | Role in Bottleneck |
| :--- | :--- | :--- |
| **`floor_hue`** | 10 values in $[0, 1]$ | Environment Context Concept |
| **`wall_hue`** | 10 values in $[0, 1]$ | Environment Context Concept |
| **`object_hue`**| 10 values in $[0, 1]$ | Foreground Object Concept |
| **`scale`** | 8 discrete values | Geometric Property Concept |
| **`shape`** | 4 classes (Cube, Cylinder, Sphere, Pill) | **Primary Target Task $Y$** |
| **`orientation`**| 15 values in $[-\pi, \pi]$ | Continuous Spatial Factor |

---

## 🔬 Benchmark Experiments

1. **CLEAR Conformal Audit**:
   Multi-alpha empirical coverage verification on Shapes3D shape classification:
   ```bash
   python scripts/clear_audit.py --config configs/clear_shapes3d.yaml
   ```

2. **Nearest-Neighbor Leakage Audit**:
   Verifies whether the latent space $z$ learns background shortcuts (wall/floor color) to predict shape:
   ```bash
   python scripts/nn_agreement.py --config configs/clear_shapes3d.yaml
   ```

---

## 🔗 Related Notes
- [[03 - Experiments/CLEAR_Audit_Shapes3D|Shapes3D Live Audit Report]]
- [[00 - Dashboards/Experiment Tracker|Experiment Tracker]]
- [[01 - Literature & Theory/CLEAR Methodology|CLEAR Methodology]]
