---
title: "Intervention Policies & Selection Strategies"
tags: [interventions, policies, active-learning, cbm]
---

# 🎯 Concept Intervention Policies

When human experts or automated verification systems inspect a model, their bandwidth is strictly budgeted. Selecting the most informative subset of $k$ concepts is critical.

---

## 📋 Implemented Intervention Policies

### 1. `RandomInterventionPolicy`
- Selects $k$ concepts uniformly at random without replacement.
- Represents the uninformed baseline.

### 2. `UncertaintyInterventionPolicy`
- Measures binary prediction entropy for each concept:
  $$
  H(p_j) = -p_j \log(p_j) - (1 - p_j) \log(1 - p_j)
  $$
- Intervenes on the top-$k$ concepts with highest predictive entropy (closest to $0.5$).
- Highly effective when concept uncertainty is well-calibrated.

### 3. `GreedyConformalInterventionPolicy`
- Evaluates the effect of candidate concept interventions directly on the downstream task prediction distribution $f_Y(c)$.
- Selects the concept $j^*$ that maximizes task prediction confidence (or equivalently minimizes task set size / entropy):
  $$
  j^* = \arg\max_{j \notin S} \max_{y \in \mathcal{Y}} P\left(Y = y \mid c_{S \cup \{j\}}\right)
  $$
- Runs in greedy forward selection order up to budget $k$.

### 4. `OracleOptimalInterventionPolicy`
- Intervenes on the concepts where the model's prediction error $|p_j - c_j^*|$ is strictly largest.
- Provides the theoretical upper bound on intervention efficiency.

---

## 🔗 Related Notes
- [[00 - Dashboards/CVPR Benchmark Leaderboard|Intervention Leaderboard]]
- [[03 - Experiments/CUB-200-2011 Benchmarks|CUB Benchmark Notes]]
- [[01 - Literature & Theory/Concept Bottleneck Models|CBM Theory]]
