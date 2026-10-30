---
title: "CUB-200-2011 Benchmarks"
tags: [cub, fine-grained, benchmark, cvpr]
---

# 🦅 CUB-200-2011 Benchmark Specification

Caltech-UCSD Birds-200-2011 is the gold standard benchmark for Concept Bottleneck Models (Koh et al., 2020), featuring 200 bird species and fine-grained visual attributes.

---

## 🎯 Dataset Attributes & Properties

- **Number of Classes**: 200 fine-grained bird species.
- **Number of Images**: 11,788 photos with varied backgrounds and poses.
- **Concept Bottleneck**: 312 binary attribute questions (canonical subsets range from 28 to 112 visual attributes: bill shape, wing color, throat pattern, etc.).

---

## 🔬 Benchmark Experiments

1. **CVPR Budgeted Intervention Pilot**:
   Simulates human-in-the-loop expert corrections across budgets $k \in \{0, 1, \dots, 8\}$:
   ```bash
   python scripts/cvpr_pilot.py --config configs/cvpr_cub.yaml --budget 6
   ```

2. **RAPS Set Cardinality Compression**:
   Evaluates how Regularized Adaptive Prediction Sets prevent set explosion while preserving 90% coverage on 10 to 200 classes.

---

## 🔗 Related Notes
- [[03 - Experiments/CVPR_Pilot_CUB|CUB Pilot Run Report]]
- [[00 - Dashboards/CVPR Benchmark Leaderboard|Intervention Leaderboard]]
- [[02 - Architecture & System/Intervention Policies|Intervention Policies]]
