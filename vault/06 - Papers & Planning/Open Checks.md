---
title: "Open Checks"
tags: [planning, checks, coverage, cub, validation]
date: 2026-11-06
---

# ✅ Open Checks

Empirical loose ends that must be resolved before numbers go into either paper. Ordered by urgency; #1 blocks the CUB split protocol (Roadmap step 1).

---

## 1. ✅ RESOLVED (2026-11-06): the 95.2% coverage was the empty-set fill, NOT shift

Same-pool calibration did **not** fix it (0.96 under the original scoring) — the cause is the **fill-empty-with-top-1 step** in `ConformalCalibrator.predict`: randomized APS produces empty sets on ~6.2% of confident images, and filling them (top-1 correct 97% of the time given empty) adds **+6.0 points under any split**. Without the fill, coverage is nominal everywhere: same-pool 0.901/0.899 (n=600/3600, empirical sd matching the Beta sd at n=600), official val-cal 0.907, official train-holdout 0.895. Residual shift ≤ +0.7 pt. Full report: [[04 - Results Logs/cub_gate_20261106|CUB Gate Run]]; D3 settled there. **Action for code:** report no-fill APS (fill is coverage-safe but destroys exactness; say which one you use).

## 2. ⚠️ CONFIRMED, and it decides the gate (2026-11-06): class-denoised concepts are a function of the class

CUB class-denoised attribute labels are constant per class, so **"all concepts right" IS "right class"**: the induced baseline's joint coverage equals its label-set coverage exactly (0.900 vs 0.898 measured). At matched 90% joint coverage the induced baseline needs **8.5 ambiguous concepts/image vs 55.3 (Bonferroni) / 57.8 (global max)** — the step-2 gate fails as framed. See [[04 - Results Logs/cub_gate_20261106|CUB Gate Run]]. Consequence: the meaningful concept-vector setting is **raw per-image annotations**, where induced is not viable (class prototypes ≠ per-image truth on ~19% of entries) and the $(m{+}1)$ frontier is the honest instrument.

## 3. Every coverage number is relative to the labels it was calibrated against

Raw per-image vs class-denoised labels agree on only **91.9%** of entries. Any table mixing label sources is comparing incomparable quantities. **Action:** annotate every coverage figure with its label source; never mix within a comparison.

## 4. Shapes3D concept coverage used 0.5-thresholded concepts — ✅ CLOSED by design change (2026-11-11)

The 52.6% joint figure was computed for concepts **thresholded at 0.5**, not for categorical factor values. **D1 is now settled** and the paired CLeaR test is **label-set based** — it never computes joint concept coverage, so the 52.6% figure stays retired with the legacy protocol (see [[04 - Results Logs/CLeaR_Paired_Pilot_clear_paired_shapes3d_real_20261111_094725|the real-data run]]). Reopens only if a concept-coverage claim returns to the paper.

---

## 🔗 Related Notes
- [[06 - Papers & Planning/Results Snapshot 2026-11-06|Results Snapshot]] — the numbers these checks refer to
- [[06 - Papers & Planning/Open Decisions|Open Decisions]] — design choices, not checks
- [[06 - Papers & Planning/Roadmap and Gates|Roadmap and Gates]]
