---
title: "Label-Set Guarantee - Scope and Limits"
tags: [theory, conformal-prediction, guarantees, interventions, residual-cbm]
date: 2026-11-06
---

# 🎯 What the Label-Set Guarantee Does and Does Not Say

Split conformal prediction treats the whole pipeline — image → concepts → label scores — as **one black box**. The guarantee attaches to the *set*, not to any point prediction, and it is **marginal**.

> **Settled (do not re-litigate):** everything on this page is agreed scope for both papers. See [[06 - Papers & Planning/Paper Split - CLeaR vs CVPR|the paper split]] for how it is used in each.

---

## 1. What is guaranteed

$$
P\bigl(Y \in C(X)\bigr) \;\ge\; 1 - \alpha \quad \text{(marginally, over the data distribution)}
$$

- Holds for the full CBM pipeline scored as a black box; no assumption on the interior (concept head, task head, residuals).
- See [[01 - Literature & Theory/Conformal Prediction Foundations|Conformal Prediction Foundations]] for the general framework.

## 2. What is NOT guaranteed

- ❌ The guarantee says **nothing about top-1 accuracy**. The true label being in the set with probability $1-\alpha$ does not mean the argmax is correct.
- ❌ **Singleton sets get no accuracy guarantee.** A singleton prediction set can still be wrong.
- ✅ **Reporting rule:** always report **singleton frequency** and **singleton error** as separate metrics. Never fold singletons into an accuracy claim.

## 3. Stochastic / probabilistic CBMs

Stochastic or probabilistic concept models are **compatible** with the guarantee, under one condition:

> Calibration and test must use the **identical sampling procedure** (same seeds policy, same number of samples, same scoring path). Any randomization inside the pipeline is treated as part of the black box.

APS score randomization is an example — it is part of the score, used identically at calibration and test. (It is also the only thing the 40 seeds of the CUB pilot varied; see [[06 - Papers & Planning/Open Checks|Open Checks]].)

## 4. Interventions

Test-time interventions are **fine** for the label-set guarantee when:

1. The intervention protocol is **the same at calibration and test**, and
2. The protocol **never uses the label** $Y$ (no leakage of the target into the procedure).

⚠️ **Each intervention regime needs its own calibration.** A threshold calibrated under "no interventions" does not cover a pipeline running under "budget-$k$ interventions", and thresholds are not comparable across regimes. See [[02 - Architecture & System/Intervention Policies|Intervention Policies]].

## 5. Residual CBMs

CBMs with a residual channel (see [[01 - Literature & Theory/Concept Bottleneck Models|CBMs]], hybrid paradigm):

- ✅ **Keep the label-set guarantee** — the score still treats the pipeline as a black box.
- ⚠️ **Interpretation cost:** set size now reflects *concepts + residual information combined*. Large sets can no longer be read as "concepts were ambiguous"; they may be driven by the residual pathway. Any set-size narrative in the papers must account for this.

## 6. Exact order statistic (implemented)

Use the **exact conformal order statistic**, not an interpolated quantile:

$$
\hat q = s_{(k)}, \qquad k = \lceil (n+1)(1-\alpha) \rceil
$$

If $k > n$ — the required rank exceeds the calibration set size — the only valid return is $+\infty$ (a **vacuous but honest** set), **not** the maximum calibration score. Returning $\max$ silently breaks validity.

✅ **Fixed in code:** [`compute_conformal_quantile`](../../src/ccm/conformal.py) in `src/ccm/conformal.py` implements the exact rank rule and returns `float("inf")` when $k > n$.

---

## 🔗 Related Notes
- [[01 - Literature & Theory/Conformal Prediction Foundations|Conformal Prediction Foundations]]
- [[01 - Literature & Theory/Joint Concept Coverage|Joint Concept Coverage]] — the concept-side analogue and its pitfalls
- [[01 - Literature & Theory/Leakage One-Sided Claim|Leakage: the One-Sided Claim]] — why coverage alone says nothing about leakage
- [[06 - Papers & Planning/Paper Split - CLeaR vs CVPR|Paper Split: CLeaR vs CVPR]]
