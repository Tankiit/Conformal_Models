---
title: "Story - CVPR Collapse Paper"
tags: [planning, story, cvpr, collapse, joint-coverage, arc]
date: 2026-11-11
---

# 📖 Story: The CVPR Paper (Concept-Set Collapse / Raw-Annotation Regime)

> [!IMPORTANT] PARKED (2026-11-13) — not part of the CLeaR submission
> The user is **not submitting to CVPR 2027** (never registered). This arm is a **follow-up paper**; the story below and all CUB results stay archived and citable when that paper is revived. Nothing from here goes into the CLeaR submission.

> The narrative arc for the CVPR arm, written down 2026-11-11 (recommended in
> chat 2026-11-06, accepted 2026-11-11). Companion arc:
> [[06 - Papers & Planning/Story - CLeaR Leakage Paper|CLeaR story]].
> One line: **concept uncertainty in CBMs is either redundant or hard, and the
> label source decides which.**

---

## Beat 1 — The promise

Quantify uncertainty over the *whole concept vector* with conformal guarantees:
calibrate once, output per-concept sets whose joint correctness is controlled.
The natural framing everyone reaches for: "an efficient joint score that beats
the baselines."

## Beat 2 — The blind spot

Class-denoised concept labels (CUB-112: attributes made constant within class)
are **functions of the class**: $C = g(Y)$. Under denoised labels, "all concepts
right" *is* "right class" — joint concept uncertainty is label uncertainty
wearing a costume. Any method built to beat the label set is fighting arithmetic,
not uncertainty.

## Beat 3 — The collapse, measured

Gate run 2026-11-06 ([[04 - Results Logs/cub_gate_20261106|run report]]), 112
denoised concepts, matched 90% realized joint coverage:

- **Induced from label sets: 8.5 ambiguous concepts/image** vs **55.3**
  (Bonferroni) and **57.8** (global max). The "baselines to beat" win by 6.5×.
- Induced's joint coverage ≡ its label APS coverage (0.900 vs 0.898) — the
  collapse is exact, not approximate.
- Dominance holds down the whole direct frontier: even the weaker at-most-$m$
  claim ($m{=}8$: 5.4 ambiguous at P(≤8 wrong) = 0.899) does not escape
  induced's *all-right* 8.5.

The step-2 gate ("beat induced") **fails as framed** — and that failure is the
finding.

## Beat 4 — The real regime: raw annotations

On **raw per-image 312 labels** (where $C \neq g(Y)$ — annotations carry
instance-level noise, 91.9% agreement with denoised):

- Induced is **not viable** (no function of the label set covers per-image
  concepts); uncorrected joint coverage is **0.000**; Bonferroni is vacuous at
  practical $n_{\text{cal}}$.
- The **$(m{+}1)$-th-largest calibration is exact at every $m$**:
  $P(\le m \text{ wrong}) \ge 1-\alpha$, a dial from all-right to mostly-right —
  306.5 → 104.2 → 30.6 ambiguous as $m$ goes 0 → 16 → 40 (α = 0.10,
  exact realized coverage ≈ 0.90 at every setting).

Framing: **(a) the collapse finding** as the headline, with **(b) the
raw-annotation $(m{+}1)$ frontier** as the constructive half; the evaluation
methodology (mismatch grid, ceilings) demoted to a section.

## Beat 5 — The protocol

Every coverage number is *relative to its label source* — state it. Report
no-fill APS (the 95.2% anomaly was the empty-set fill: +6.0 pts ≈ 6.2% empty ×
97% top-1-correct, not shift); temperature scaling keeps coverage exact and
shrinks sets 2.57 → 1.81.

## Guardrails

- **Never** frame as efficiency / "beats baselines" — the gate failure is the
  story; a re-run claiming victory would be self-refuting.
- Do not oversell $(m{+}1)$ novelty — cite the order-statistics conformal
  lineage; the contribution is the *regime* (raw annotations), not the statistic.
- Biggest review risk: **one dataset / one backbone** (CUB + DINOv2). A second
  raw-annotation dataset (e.g. CelebA-style attributes) is the strongest
  de-risking move.
- Keep all leakage material out — that is the
  [[06 - Papers & Planning/Story - CLeaR Leakage Paper|CLeaR paper]].

## Title candidates (from 2026-11-06)

1. *Concept Sets Collapse: When Concept Uncertainty Is Label Uncertainty in Bottleneck Models*
2. *Do Concept Sets Say More Than Label Sets? The Label Source Decides*
3. *Exact At-Most-m Concept Set Guarantees for CBMs*

## ⚠️ Open items

- **Registration status for CVPR 2027 is not recorded in the vault.** The
  registration date (10 Nov 2026) has passed; if the paper was not registered,
  the 16 Nov full submission is moot and this story's vehicle becomes the next
  available deadline (CLeaR?). Confirm and record.

## 🔗 Related Notes
- [[04 - Results Logs/cub_gate_20261106|CUB Gate Run]] · [[06 - Papers & Planning/Results Snapshot 2026-11-13|Results Snapshot (canonical)]]
- [[01 - Literature & Theory/Joint Concept Coverage|Joint Concept Coverage]]
- [[06 - Papers & Planning/Paper Split - CLeaR vs CVPR|Paper Split]] · [[06 - Papers & Planning/Roadmap and Gates|Roadmap]]
