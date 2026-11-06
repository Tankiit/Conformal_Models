---
title: "Concept Bottleneck Models (CBMs)"
tags: [theory, cbms, interpretability, interventions]
---

# 🧩 Concept Bottleneck Models (CBMs)

Concept Bottleneck Models (Koh et al., ICML 2020) structure a deep neural network into two sequential, interpretable stages:

$$
x \xrightarrow{f_x} c \xrightarrow{g_y} y
$$

1. **Input-to-Concept Model ($x \to c$)**: Maps raw input $x \in \mathcal{X}$ to a vector of predefined, human-understandable concepts $c \in [0, 1]^K$ (e.g. wing color, shape, scale).
2. **Concept-to-Task Model ($c \to y$)**: Maps concept representations $c$ to final task predictions $y \in \mathcal{Y}$.

---

## 🛠️ Training Paradigms

1. **Joint Training**:
   Both components are trained simultaneously via a joint loss function:
   $$
   \mathcal{L}_{\text{joint}} = \mathcal{L}_{\text{task}}(g_y(f_x(x)), y) + \lambda \sum_{j=1}^K \mathcal{L}_{\text{concept}}(f_x(x)_j, c_j)
   $$
   - *Advantage*: High task accuracy.
   - *Risk*: Risk of concept leakage where $f_x(x)$ encodes task shortcuts rather than pure concept semantics.

2. **Sequential Training**:
   $f_x$ is trained exclusively on concept targets $c$. Once converged, $f_x$ is frozen, and $g_y$ is trained purely on the predicted concepts $\hat{c}$.
   - *Advantage*: Eliminates gradient flow from $y$ to $f_x$.

3. **Hybrid / Residual Bottleneck Models**:
   Augments the concept bottleneck with a bounded residual latent space:
   $$
   [c; \beta z_{\text{res}}] \to y
   $$
   - Used explicitly in benchmarking to audit the trade-off between concept fidelity and model capacity.

---

## 🧑‍⚕️ Test-Time Interventions

A defining feature of CBMs is the ability for human experts or automated oracles to inspect and intervene on intermediate concepts:

$$
\hat{c}_j \gets c^*_j
$$

When erroneous concepts are corrected, the downstream prediction $g_y(c)$ shifts accordingly. Key research challenges include:
- Budgeted selection: Which subset of $k$ concepts yields the largest downstream error reduction?
- Uncertainty propagation: How does error in non-intervened concepts propagate to the task prediction sets?

---

## 🔗 Related Notes
- [[02 - Architecture & System/Intervention Policies|Intervention Policies]]
- [[03 - Experiments/NN_Agreement_Audit_Shapes3d|Leakage Diagnostics]]
- [[01 - Literature & Theory/CLEAR Methodology|CLEAR Methodology]]
- [[01 - Literature & Theory/Label-Set Guarantee - Scope and Limits|Label-Set Guarantee: Scope and Limits]] — interventions, residual CBMs, and what stays valid
- [[06 - Papers & Planning/Paper Split - CLeaR vs CVPR|Paper Split: CLeaR vs CVPR]]
