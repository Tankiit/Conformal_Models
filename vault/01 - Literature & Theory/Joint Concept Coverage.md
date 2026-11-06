---
title: "Joint Concept Coverage"
tags: [theory, conformal-prediction, concepts, multiple-testing, risk-control]
date: 2026-11-06
---

# 🧱 Joint Concept Coverage

Per-concept conformal sets are easy. **Per-concept guarantees do not combine.** If each of $K$ concepts is covered at $1-\alpha$ marginally, the joint event "all concepts covered" holds at roughly $(1-\alpha)^K$ — e.g. about $0.9^{6} \approx 0.53$ for six concepts, which is exactly what the Shapes3D run observed ([[06 - Papers & Planning/Results Snapshot 2026-11-06|Results Snapshot]]).

This note is the design space for the **CVPR paper's** core question: *can we quantify uncertainty over the whole concept vector efficiently?* Status of each piece below; the efficiency gate lives in [[06 - Papers & Planning/Roadmap and Gates|Roadmap and Gates]].

---

## 1. The problem

Per-concept marginal sets $C_1(x), \dots, C_K(x)$ each satisfy $P(c^*_j \in C_j(x)) \ge 1-\alpha$, but jointly:

$$
P\bigl(\forall j:\; c^*_j \in C_j(x)\bigr) \;\approx\; (1-\alpha)^K \xrightarrow{K \to 312} 0
$$

Empirically: **uncorrected joint coverage is 0.000 at $K = 312$ concepts** on CUB (see snapshot).

## 2. Candidate forms of the joint claim — ❓ OPEN (Decision D4)

| Form | Statement | Mechanism | Status |
| :--- | :--- | :--- | :--- |
| **(a) All concepts right** | $P(\forall j: c^*_j \in C_j(x)) \ge 1-\alpha$ | Bonferroni or global max-score | Baseline; expensive at large $K$ |
| **(b) At most $m$ wrong** | $P(\#\{\text{uncovered concepts}\} \le m) \ge 1-\alpha$ | Sample score $t(x) = (m{+}1)$-th largest per-concept score, calibrated at $1-\alpha$ | Strong candidate; flexible $m$ |
| **(c) Expected fraction** | $\mathbb{E}\bigl[\tfrac{1}{K}\sum_j \mathbb{1}\{c^*_j \notin C_j(x)\}\bigr] \le \alpha$ | Conformal risk control (bounds expected fraction wrong) | Strong candidate; weakest-looking claim, easiest to satisfy |

For (b): the event "at most $m$ concepts uncovered" is exactly the event "$\le m$ scores exceed $\hat q$", i.e. the $(m{+}1)$-th largest score $\le \hat q$ — so calibrating the $(m{+}1)$-th largest score as a single nonconformity score gives the guarantee by split conformal directly.

## 3. Baseline suite

Every claim form must be compared against:

1. **Uncorrected per-concept** — the straw man (0.000 joint at $K=312$).
2. **Bonferroni** — per-concept level $\alpha/K$; valid but vacuous at small calibration sizes ($n = 600$ on CUB).
3. **Global max-over-concepts score** — $t(x) = \max_j s_j(x)$, single threshold.
4. **$(m{+}1)$-th largest score** — the mechanism behind claim (b).
5. **Induced concept sets from label sets** — project the conformal *label* set back through the concept-to-label map. Required baseline especially because class-denoised concept labels are a function of the class, so "all concepts right" is nearly "right class" ([[06 - Papers & Planning/Open Checks|Open Checks]] #2).
6. **Proposed local score** (nearest-neighbour) — the method contribution; build only if the global score leaves a gap (Roadmap step 4).

## 4. Locally adaptive (nearest-neighbour) sets — validity conditions ⚠️

Locally adaptive concept sets are **only exactly valid** under one of these two constructions:

1. **Reference-then-calibrate:** build the local (NN) score from *reference data*, then calibrate its threshold **globally on an independent calibration set**. The locality enters through the score; the quantile stays exchangeable.
2. **Frozen partition + Mondrian:** freeze the partition (e.g. of input space or difficulty strata) **before** looking at calibration data, then calibrate per-cell with [[02 - Architecture & System/Pipeline Architecture|Mondrian calibration]] (`MondrianConformalCalibrator`).

Any scheme that both *defines* neighborhoods and *calibrates* on the same data destroys exchangeability and the guarantee with it.

---

## 🔗 Related Notes
- [[01 - Literature & Theory/Label-Set Guarantee - Scope and Limits|Label-Set Guarantee: Scope and Limits]]
- [[02 - Architecture & System/Nonconformity Scores|Nonconformity Scores]] — the score functions behind each baseline
- [[06 - Papers & Planning/Paper Split - CLeaR vs CVPR|Paper Split]] — this is the CVPR paper's question
- [[06 - Papers & Planning/Results Snapshot 2026-11-06|Results Snapshot]] — current joint-coverage numbers on CUB
- [[06 - Papers & Planning/Open Decisions|Open Decisions]] — D4 (claim form)
