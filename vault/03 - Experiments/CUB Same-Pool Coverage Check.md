---
title: "CUB Same-Pool Coverage Check"
dataset: cub
type: experiment-design
tags: [cvpr, cub, coverage, validation, design]
date: 2026-11-06
---

# 🔬 CUB Same-Pool Coverage Check

**Purpose:** explain the anomalous **95.2% APS coverage at $\alpha = 0.10$** ([[06 - Papers & Planning/Results Snapshot 2026-11-06|Results Snapshot]] §2). This is Roadmap step 1 and blocks the CUB split protocol ([[06 - Papers & Planning/Open Decisions|D3]]).

---

## ✅ Outcome (2026-11-06 — run report: [[04 - Results Logs/cub_gate_20261106|CUB Gate Run]])

> **Resolved — and the hypothesis was wrong in the best way.** Same-pool coverage did *not* drop to ~90.5% under the original scoring: it stayed at 0.96, exactly like the official split. The cause of the 95.2% (and of the same-pool 0.96) is the **empty-set fill**: randomized APS yields empty sets on ~6.2% of confident images, and filling them with the top-1 class (correct 97% of the time given empty) adds **+6.0 coverage points** under *any* split. With the fill removed, every protocol is nominal: same-pool n=600 **0.9014 ± 0.0128** (Beta sd 0.0122 — the theory hint confirmed), same-pool n=3600 **0.8995 ± 0.0085**, official val-cal **0.9071 ± 0.0088**, official train-holdout **0.8948 ± 0.0083**. Residual train/test shift ≤ +0.7 pt, within noise. Bonus: temperature scaling (fit on cal) keeps coverage exact and shrinks sets 2.57 → 1.81.

**D3 (split protocol) settled:** any same-distribution calibration pool; report **no-fill** coverage; state the label source. The pre-registered table below predates the run and is kept for the record.

## Hypothesis (pre-registered)

The 40 seeds of the original run varied **only the APS randomization**, which cannot move marginal coverage by +4.7 points. If drawing calibration *and* evaluation from the **same pool** yields mean coverage ≈ **90.5%**, then the **official train/test shift** is the cause of the anomaly. If not, suspect the calibration code path — start from [`compute_conformal_quantile`](../../src/ccm/conformal.py) and the score in `src/ccm/scores.py`.

## Design

Many random splits of one pool into fit / calibration / evaluation; identical distribution across the three roles by construction. Per split: refit heads on fit only, exact conformal quantile on calibration, coverage on evaluation. Report mean and standard deviation of coverage across draws.

```python
def coverage_over_draws(pool, n_draws, n_cal, alpha):
    covs = []
    for seed in range(n_draws):
        # TODO: random split of pool into fit / cal / eval (same distribution)
        # TODO: fit heads on fit only; exact conformal quantile on cal
        # TODO: coverage on eval
        covs.append(...)
    # TODO: mean and sd; what sd do you expect for n_cal = 600 vs 3600?
```

## Pre-registration of outcomes

| Outcome | Reading |
| :--- | :--- |
| Mean ≈ **0.905** | Official train/test shift explains 95.2%; coverage is sound within-distribution. Fix D3 accordingly (calibrate within split). |
| Mean ≫ 0.905 | The anomaly is in the pipeline, not the split. Audit the quantile and score code before anything else. |
| sd larger than expected at $n_{\text{cal}} = 600$ | Score distribution heavy-tailed / discrete near the quantile; consider $n_{\text{cal}} = 3600$ for the paper. |

Also settle here which label source (raw per-image vs class-denoised, 91.9% agreement) each coverage figure uses — [[06 - Papers & Planning/Open Checks|Open Checks]] #3.

---

## 🔗 Related Notes
- [[06 - Papers & Planning/Open Checks|Open Checks]] #1 · [[06 - Papers & Planning/Roadmap and Gates|Roadmap]] step 1
- [[01 - Literature & Theory/Conformal Prediction Foundations|Conformal Prediction Foundations]]
- [[03 - Experiments/CUB-200-2011 Benchmarks|CUB Benchmarks]]
