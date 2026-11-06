---
title: "Leakage - The One-Sided Claim"
tags: [theory, leakage, interventions, clear, one-sided]
date: 2026-11-06
---

# 🚱 Leakage: The One-Sided Claim

The CLeaR paper's central claim about leakage is deliberately **one-sided**. Getting its logic right is a prerequisite for everything in [[06 - Papers & Planning/Paper Split - CLeaR vs CVPR|the CLeaR arm]].

---

## 1. What conformal coverage can and cannot do

> **Conformal coverage alone neither detects nor excludes leakage.**

- A miss (coverage below nominal) does not *diagnose* leakage — many things break coverage.
- A hit (valid coverage) does not *certify* the absence of leakage.

What *can* reveal leakage: a **controlled intervention test** showing the predictor's output depends on information **outside the declared concepts**. And even then:

> ⚠️ **A missed alarm is not proof of absence.** The test has no certificate in the "clean" direction. This asymmetry must be stated in the paper.

## 2. The two intervention tests

| Test | Held fixed | Varied | Applies to |
| :--- | :--- | :--- | :--- |
| **Test 1** | Ground-truth concepts $C$ | Image nuisance $U$ (regenerate the image) | Full pipeline |
| **Test 2** | Predicted concept values $\hat c$ | The residual channel | **Residual CBMs only** |

Test 1: keep ground-truth concepts fixed, vary the nuisance, run the full image → concept → label pipeline. If outputs move, the pipeline tracks $U$, i.e. information beyond the declared concepts.

Test 2: keep predicted concept values fixed, vary the residual — isolates the residual pathway of a residual CBM.

## 3. Generative setup and interpretation rules

Work under:

$$
Y = f(C), \qquad X = g(C, U)
$$

with any correlation between $U$ and $Y$ created **only through training-data sampling** (the sampler favors certain $(C, U)$ co-occurrences; marginally $Y$ is a function of $C$ alone).

Interpretation:

- Under $Y = f(C)$: sensitivity to $U$ ⟹ the model uses information outside the declared concepts ⟹ **leakage**.
- If instead $Y = f(C, U)$ (the label genuinely depends on $U$): sensitivity to $U$ shows an **incomplete concept list**, *not* leakage. The two diagnoses must be kept distinct in the paper.

## 4. ❓ OPEN — what does conformalization add?

The set-based test must be compared against **ordinary probability / logit sensitivity**: feed the same paired images through and measure how much the softmax probabilities or logits move.

> **The paper must answer:** what does the *set-based* version detect that the probability-change version does not? If nothing, the conformal machinery is decoration. This is the CLeaR gate (Roadmap step 3).

## 5. 🪦 Corrected interpretation: the B-minus-R gap

An earlier reading attributed the **B-minus-R** gap (benchmark minus residual-arm difference) to leakage. **That interpretation was wrong:**

> Both arms receive ground-truth concepts, so the gap reflects **training mismatch** between the arms — not leakage.

Do not resurrect this claim; audit notes predating 2026-11-06 that say otherwise are superseded.

---

## 🔗 Related Notes
- [[01 - Literature & Theory/CLEAR Methodology|CLEAR Methodology]] — the broader audit framework (NN-agreement leakage metrics etc.)
- [[01 - Literature & Theory/Label-Set Guarantee - Scope and Limits|Label-Set Guarantee]] — why coverage is silent on leakage
- [[03 - Experiments/Shapes3D Benchmarks|Shapes3D Benchmarks]] / [[03 - Experiments/NN_Agreement_Audit_Shapes3D|NN Agreement Audit]]
- [[06 - Papers & Planning/Roadmap and Gates|Roadmap]] — step 3 gate
