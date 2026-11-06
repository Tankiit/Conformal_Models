---
title: "Results Snapshot 2026-11-06"
tags: [results, shapes3d, cub, snapshot, planning]
date: 2026-11-06
---

# 📸 Results Snapshot — 2026-11-06

> **Provenance:** all numbers below come from runs in this codebase (`scripts/`) and were **not re-run** for the 2026-11-06 planning pass. Raw per-run logs: [[04 - Results Logs/README|Results Logs]]. Older rows in the [[00 - Dashboards/Experiment Tracker|Experiment Tracker]] are earlier pilot/synthetic runs and are **superseded** by this snapshot for planning purposes.
>
> **🔁 Update (same day):** a local full replication now exists — [[04 - Results Logs/cub_gate_20261106|CUB Gate Run]] (DINOv2 features re-extracted on this machine; probe 0.894, CBM-312 0.872; check 1 resolved: **the 95.2% was the empty-set fill, not shift**; step-2 gate fails — induced baseline dominates). Where this snapshot and the Gate Run disagree, the Gate Run's protocol (no-fill, same-pool) is the one to use.

---

## 1. Shapes3D (all 480k images) — 🔴 SATURATED

| Metric | Value |
| :--- | :--- |
| Task accuracy | 100% |
| Concept accuracy | 97.4% |
| Label coverage | 100%, **singleton sets** |
| Joint concept coverage | 52.6% (≈ $0.9^6$), no hard concepts |

**Verdict: Shapes3D as currently run cannot test either paper.** Everything is at ceiling; label sets are all singletons at 100% coverage. The joint-coverage figure behaving like the independent product $0.9^6$ confirms there is no structure to exploit. The paired intervention test at training correlation 0 and 1 (Roadmap step 3) is what Shapes3D is still for.

⚠️ Caveat: this concept coverage was computed for concepts **thresholded at 0.5**, not for categorical factor values ([[06 - Papers & Planning/Open Checks|Open Checks]] #4).

## 2. CUB — frozen DINOv2-base, 312 attributes, sequential CBM

| Metric | Value |
| :--- | :--- |
| Test accuracy | 82.1% (linear probe: 87.7%) |
| APS coverage @ $\alpha = 0.10$, $n_{\text{cal}} = 600$ | **95.2%** ⚠️ unexplained — see Open Checks #1 |
| Avg classes per set | 3.49 |

### 2.1 Joint concept coverage ($n = 3{,}600$ calibration)

| Concept set × method | Joint coverage | Ambiguous concepts / image |
| :--- | :---: | :---: |
| 312 raw, Bonferroni | 0.947 | 291.6 of 312 |
| 312 raw, global max | 0.884 | 247 of 312 |
| 112 denoised, Bonferroni | 0.955 | 51.5 of 112 |
| 112 denoised, global max | 0.910 | 42.6 of 112 |

- Uncorrected joint coverage: **0.000** at 312 concepts.
- Bonferroni is **vacuous at $n = 600$** (sets that big / thresholds that crude say nothing).

Read with [[01 - Literature & Theory/Joint Concept Coverage|Joint Concept Coverage]] for what each row means.

### 2.2 Label-head train/test mismatch grid

| Label head trained on ＼ tested on | Predicted concepts | True concepts |
| :--- | :---: | :---: |
| **Predicted concepts** | **61.2%** | 32.2% |
| **True concepts** | 54.3% | **38.8%** |

- The **38.8% corner is an annotation-noise ceiling** (true-concept labels are noisy at the per-image level).
- **Train/test mismatch costs ≈ 6.9 points** (61.2% → 54.3% when the head is trained on true concepts but tested on predicted ones; the mirrored direction loses ≈ 6.6).
- The **Oracle intervention policy halving accuracy is most likely this mismatch, not a bug.** Before debugging code, check mismatch first.

## 3. Every coverage number is *relative to its labels*

Raw per-image attribute labels and class-denoised labels **agree on only 91.9% of entries**. Coverage computed against one is not coverage against the other — always state which label source a number was calibrated against ([[06 - Papers & Planning/Open Checks|Open Checks]] #3).

---

## 🔗 Related Notes
- [[06 - Papers & Planning/Paper Split - CLeaR vs CVPR|Paper Split]]
- [[06 - Papers & Planning/Open Checks|Open Checks]]
- [[03 - Experiments/CUB-200-2011 Benchmarks|CUB Benchmarks]] · [[03 - Experiments/Shapes3D Benchmarks|Shapes3D Benchmarks]]
- [[00 - Dashboards/Experiment Tracker|Experiment Tracker]]
