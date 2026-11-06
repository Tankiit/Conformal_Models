---
title: "Nonconformity Score Functions"
tags: [scores, aps, raps, softmax, nonconformity]
---

# 🧮 Nonconformity Score Functions in CCM

The choice of non-conformity score function governs the geometric shape, adaptiveness, and average cardinality of the resulting conformal prediction sets.

---

## 📊 Summary of Implemented Score Functions

| Score Function | Formula | Best Suited For | Key Properties |
| :--- | :--- | :--- | :--- |
| **`SoftmaxScore`** | $s(x, y) = 1 - \hat{p}(y \mid x)$ | Few-class classification | Simple, fast, unregularized |
| **`APSScore`** | $s(x, y) = \sum_{k \prec y} \hat{p}_{(k)} + U \cdot \hat{p}(y)$ | General classification | Adaptive, exact marginal coverage |
| **`RAPSScore`** | $s_{\text{APS}} + \lambda \max(0, \text{rank}(y) - k_{\text{reg}})$ | Fine-grained tasks (e.g. CUB) | Prevents runaway prediction sets |
| **`MarginScore`** | $\max_{y' \neq y} \hat{p}(y') - \hat{p}(y)$ | Binary & low-class classification | Emphasizes decision boundary gap |
| **`NormalizedScore`**| $s(x, y) / (\sigma(x) + \epsilon)$ | Heteroscedastic data | Adaptively widens on harder samples |
| **`ConceptResidual`**| $1 - \hat{p}(c_j \mid x)$ | Multi-label concept bottleneck | Per-concept prediction intervals |

---

## ⚙️ Hyperparameters for RAPS
- `lambda_reg` ($\lambda \in [0.001, 0.05]$): Controls the penalty strength per excess rank.
- `k_reg` ($k_{\text{reg}} \in \{1, \dots, 5\}$): Rank threshold above which penalty begins accumulating.

---

## 🔗 Related Notes
- [[01 - Literature & Theory/Conformal Prediction Foundations|Conformal Prediction Theory]]
- [[01 - Literature & Theory/Joint Concept Coverage|Joint Concept Coverage]] — which score constructions give which joint guarantee
- [[02 - Architecture & System/Pipeline Architecture|Pipeline Architecture]]
- [[00 - Dashboards/Audit Dashboard|Audit Dashboard]]
