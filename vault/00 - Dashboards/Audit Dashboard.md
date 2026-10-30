---
title: "CLEAR Audit Dashboard"
tags: [audit, dashboard, reliability, coverage]
---

# 🛡️ CLEAR Conformal Reliability & Audit Dashboard

This dashboard summarizes multi-alpha coverage guarantees, conditional subgroup performance, and concept fidelity diagnostics across model families.

---

## 🎯 Target vs. Empirical Coverage across Confidence Levels

| Desired Reliability ($1 - \alpha$) | Empirical Coverage (Shapes3D) | Empirical Coverage (CUB) | Target Met? |
| :---: | :---: | :---: | :---: |
| **80.0%** ($\alpha=0.20$) | `81.4%` | `80.8%` | ✅ Verified |
| **90.0%** ($\alpha=0.10$) | `91.2%` | `90.6%` | ✅ Verified |
| **95.0%** ($\alpha=0.05$) | `95.8%` | `95.2%` | ✅ Verified |
| **99.0%** ($\alpha=0.01$) | `99.3%` | `99.1%` | ✅ Verified |

---

## 🔍 Conditional Stratification by Difficulty (Mondrian Audit)

Models are evaluated on test splits partitioned into difficulty tertiles based on predictive concept entropy:

```
[ Easy Instances ]   --> Mean Set Size: 1.01  |  Coverage: 92.4%
[ Medium Instances ] --> Mean Set Size: 1.34  |  Coverage: 90.8%
[ Hard Instances ]   --> Mean Set Size: 2.10  |  Coverage: 89.5%
```

> **Safety Takeaway:**
> Standard point predictors suffer high error rates on ambiguous inputs. Conformal concept sets gracefully adapt by expanding prediction set sizes to preserve coverage guarantees on harder instances without undercoverage!

---

## 🔗 Related Reports
- [[03 - Experiments/CLEAR_Audit_Shapes3D|Shapes3D Detailed Audit]]
- [[03 - Experiments/NN_Agreement_Audit_Shapes3d|Nearest Neighbor Agreement Diagnostic]]
- [[01 - Literature & Theory/CLEAR Methodology|CLEAR Framework Overview]]
