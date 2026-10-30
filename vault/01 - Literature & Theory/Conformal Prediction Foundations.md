---
title: "Conformal Prediction Foundations"
tags: [theory, conformal-prediction, statistics, uncertainty]
---

# 📐 Conformal Prediction Foundations

Conformal prediction is a framework for converting any heuristic point predictor into a set-valued predictor that is guaranteed to cover the ground-truth target with a user-specified probability $1 - \alpha$.

---

## 1. Mathematical Guarantee

Let $(X_1, Y_1), \dots, (X_n, Y_n) \overset{\text{i.i.d.}}{\sim} P_{XY}$ be calibration data, and $(X_{n+1}, Y_{n+1}) \sim P_{XY}$ be a new test point. Assuming only exchangeability (which is satisfied by i.i.d.), split conformal prediction guarantees:

$$
P\left(Y_{n+1} \in C(X_{n+1})\right) \ge 1 - \alpha
$$

Furthermore, if nonconformity scores have continuous distributions, the upper bound is tightly controlled:

$$
1 - \alpha \le P\left(Y_{n+1} \in C(X_{n+1})\right) \le 1 - \alpha + \frac{1}{n + 1}
$$

---

## 2. Nonconformity Score Functions

The nonconformity score $s(x, y) \in \mathbb{R}$ quantifies how unusual or non-conforming label $y$ appears given input $x$.

### A. Softmax / Hinge Score
$$
s(x, y) = 1 - \hat{p}(y \mid x)
$$
- Simple and intuitive.
- Sensitive to probability miscalibration and struggles with multi-class ranking ambiguity.

### B. Adaptive Prediction Sets (APS)
Sort class probabilities in descending order: $\hat{p}_{\pi(1)} \ge \hat{p}_{\pi(2)} \ge \dots \ge \hat{p}_{\pi(K)}$:
$$
s(x, y) = \sum_{k=1}^{\text{rank}(y) - 1} \hat{p}_{\pi(k)} + U \cdot \hat{p}(y \mid x), \quad U \sim \text{Uniform}(0, 1)
$$
- Provides exact finite-sample coverage even with discrete scores.

### C. Regularized APS (RAPS)
Adds an explicit set-size penalty for low-probability classes beyond rank $k_{\text{reg}}$:
$$
s(x, y) = s_{\text{APS}}(x, y) + \lambda \max(0, \text{rank}(y) - k_{\text{reg}})
$$
- Greatly compresses average prediction set size, especially on large label spaces (e.g. CUB-200).

---

## 3. Finite-Sample Quantile Computation

Given calibration nonconformity scores $\{s_1, \dots, s_n\}$, the calibrated critical value is:

$$
\hat{q} = \text{Quantile}\left(\{s_i\}_{i=1}^n, \frac{\lceil (n + 1)(1 - \alpha) \rceil}{n}\right)
$$

The prediction set for a test instance $x_{n+1}$ is then formed as:

$$
C(x_{n+1}) = \left\{ y \in \mathcal{Y} : s(x_{n+1}, y) \le \hat{q} \right\}
$$

---

## 🔗 Related Notes
- [[02 - Architecture & System/Nonconformity Scores|Score Function Implementations]]
- [[01 - Literature & Theory/CLEAR Methodology|CLEAR Audit Framework]]
- [[00 - Dashboards/Audit Dashboard|Audit Dashboard]]
