---
title: "Shapes3D Benchmarks"
tags: [shapes3d, benchmarks, disentanglement, clear]
---

# 🧊 Shapes3D Benchmark Specification

The 3D Shapes dataset is a canonical synthetic benchmark used to assess concept learning, disentanglement, and conformal reliability.

---

## 🎯 Dataset Attributes & Factors

Official DeepMind 3dshapes: 480,000 images, full factorial grid. Real data on
this Mac: `/Users/tanmoy/research/data/3dshapes.h5` (gzip HDF5, 255MB;
downloaded 2026-11-11 — the `/Users/cril/...` cache paths in older configs are
stale). Level counts verified: 10/10/10/8/4/15, column order
[floor, wall, obj, scale, shape, orient].

**Roles under the settled CLeaR paired design** (2026-11-11; see
[[03 - Experiments/Shapes3D Paired-Nuisance Generator Spec|the spec]]):

| Factor | Cardinality / Range | Role |
| :--- | :--- | :--- |
| **`floor_hue`** | 10 values in $[0, 1]$ | **Pinned to one level** (undeclared, non-nuisance) |
| **`wall_hue`** | 10 values in $[0, 1]$ | **Hidden nuisance $U$** — binarized to two palette levels (default 1, 8), NOT a concept |
| **`object_hue`** | 10 values in $[0, 1]$ | Declared concept |
| **`scale`** | 8 discrete values | Declared concept (+ bucket enters $Y$) |
| **`shape`** | 4 classes | Declared concept (enters $Y$) |
| **`orientation`** | 15 values in $[-30°, 30°]$ | Declared concept |
| **$Y$** | 8 classes | Composite $Y = f(C)$ = (`shape`, `scale`-bucket), buckets = 2 |

---

## 🔬 Benchmark Experiments

1. **CLeaR Paired-Nuisance Grid (REAL data, primary)** — the step-3 experiment:
   ```bash
   python scripts/clear_paired.py --config configs/clear_paired_shapes3d_real.yaml
   ```
   First run 2026-11-11: [[04 - Results Logs/CLeaR_Paired_Pilot_clear_paired_shapes3d_real_20261111_094725|real-data run log]]
   (ρ=0 CBM at **acc 0.834, set size 1.36** — de-saturated with non-trivial sets;
   Test-2 signal 0.9+ vs nulls ≤ 0.056; sanity null exactly 0).

2. **CLEAR Conformal Audit** (legacy protocol):
   ```bash
   python scripts/clear_audit.py --config configs/clear_shapes3d.yaml
   ```

3. **Nearest-Neighbor Leakage Audit** (nn-leakage v3 also runs inside the paired pilot):
   ```bash
   python scripts/nn_agreement.py --config configs/clear_shapes3d.yaml
   ```

---

## 🔗 Related Notes
- [[03 - Experiments/Shapes3D Paired-Nuisance Generator Spec|Paired-Nuisance Generator Spec]]
- [[03 - Experiments/CLEAR_Audit_Shapes3D|Shapes3D Live Audit Report]]
- [[00 - Dashboards/Experiment Tracker|Experiment Tracker]]
- [[01 - Literature & Theory/CLEAR Methodology|CLEAR Methodology]]
