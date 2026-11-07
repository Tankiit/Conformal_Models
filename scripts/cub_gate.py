#!/usr/bin/env python3
"""
CUB gate experiment for the CVPR paper (Roadmap steps 1-2).

Stage `heads`  : sequential CBM heads on frozen DINOv2-base features (112 denoised, 312 raw, linear probe).
Stage `check1` : same-pool coverage check - explains the 95.2% APS anomaly (Open Checks #1 / D3).
Stage `gate`   : joint concept comparison - uncorrected, Bonferroni, global max, (m+1)-th largest,
                 induced-from-label-sets, compared at MATCHED realized joint coverage.
Stage `table`  : reproduces the existing CUB joint table under official-shift vs same-pool protocols.

Everything is deterministic given --seed. Results are appended to data_cache/gate_results.json
and summarized for the Obsidian vault.
"""

from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from ccm.conformal import (
    compute_conformal_quantile,
    conformal_quantile_per_column,
    induced_concept_sets,
    m_plus_one_score,
)
from ccm.scores import APSScore
from ccm.utils import set_seed

# Library functions are the single source of truth; local names keep call sites stable.
quantile_per_column = conformal_quantile_per_column
image_score_topm = m_plus_one_score

CACHE = PROJECT_ROOT / "data_cache"
RESULTS_JSON = CACHE / "gate_results.json"
REF = {  # summary-doc checkpoints used to validate the replication
    "cbm312_test_acc": 0.821,
    "probe_test_acc": 0.877,
    "labelhead112_on_pred_acc": 0.612,
    "aps_coverage_official_n600": 0.952,
    "aps_set_size_official_n600": 3.49,
    "raw_denoised_agreement": 0.919,
}

ALPHA_GRID = [0.02, 0.03, 0.05, 0.07, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30]
M_GRID = [0, 1, 2, 4, 8, 16, 28]
AT_MOST_GRID = [0, 1, 2, 4, 8, 16, 28, 56]


# ─────────────────────────────── data ───────────────────────────────

def load_split(split: str) -> dict:
    d = torch.load(CACHE / f"cub_dinov2b14_{split}.pt", map_location="cpu", weights_only=False)
    return d


def raw312_labels(d: dict) -> torch.Tensor:
    """Raw per-image binary labels: CUB is_present on all 312 attributes."""
    return d["present312"]


class Data:
    """All splits, both label sources, standardized features."""

    def __init__(self):
        self.tr, self.va, self.te = load_split("train"), load_split("val"), load_split("test")
        mu = self.tr["features"].mean(0, keepdim=True)
        sd = self.tr["features"].std(0, keepdim=True).clamp_min(1e-6)
        for d in (self.tr, self.va, self.te):
            d["X"] = (d["features"] - mu) / sd
            d["c112"] = d["concepts112"].float()                      # class-denoised
            d["c312"] = raw312_labels(d)                              # raw per-image
        self.class_concept_112 = self._class_concept_table()

    def _class_concept_table(self) -> torch.Tensor:
        """200 x 112 class -> class-denoised concept vector (constant per class, verified)."""
        y, c = self.tr["labels"], self.tr["c112"]
        M = torch.full((int(y.max()) + 1, c.shape[1]), -1.0)
        for cls in torch.unique(y):
            rows = c[y == cls]
            assert (rows == rows[0]).all(), "class-denoised labels not constant within class"
            M[int(cls)] = rows[0]
        assert (M >= 0).all()
        return M


# ─────────────────────────────── heads ───────────────────────────────

def fit_linear_classifier(X: torch.Tensor, Y: torch.Tensor, out_dim: int, binary: bool,
                          pos_weight: bool = True, epochs: int = 1500, lr: float = 0.05,
                          seed: int = 0, verbose: bool = False) -> torch.nn.Linear:
    """Full-batch Adam on a single linear layer. X: (N,D) standardized, Y: (N,out) targets."""
    set_seed(seed)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    X, Y = X.to(device), Y.to(device)
    W = torch.nn.Linear(X.shape[1], out_dim).to(device)
    pos_w = None
    if binary and pos_weight:
        p, n = Y.sum(0).clamp(1), (1 - Y).sum(0).clamp(1)
        pos_w = (n / p).clamp(0.1, 10.0)
    opt = torch.optim.Adam(W.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        opt.zero_grad()
        logits = W(X)
        if binary:
            loss = F.binary_cross_entropy_with_logits(logits, Y, pos_weight=pos_w)
        else:
            loss = F.cross_entropy(logits, Y)
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (ep + 1) % 300 == 0:
            print(f"      epoch {ep+1}: loss {loss.item():.4f}")
    return W.cpu().eval()


def train_heads(data: Data, seed: int = 0, n_head_train: int | None = None, with_probe: bool = True) -> dict:
    """Sequential CBM heads. If n_head_train is set, heads use only that many shuffled train
    rows; heads['cal_pool_idx'] holds the remaining rows as a clean calibration pool."""
    heads = {}
    if n_head_train is not None:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(data.tr["labels"]), generator=g)
        head_idx, rest_idx = perm[:n_head_train], perm[n_head_train:]
    else:
        head_idx = torch.arange(len(data.tr["labels"]))
        rest_idx = None
    Xtr = data.tr["X"][head_idx]
    print(f"[*] Training heads on {len(head_idx)} rows (sequential CBM on frozen DINOv2-base) ...")
    for K, key in [(112, "c112"), (312, "c312")]:
        print(f"    concept head K={K} ({'class-denoised' if K==112 else 'raw per-image'})")
        ch = fit_linear_classifier(Xtr, data.tr[key][head_idx], K, binary=True, seed=seed)
        with torch.no_grad():
            cprob = torch.sigmoid(ch(Xtr))
        print(f"    label head on predicted concepts K={K}")
        lh = fit_linear_classifier(cprob, data.tr["labels"][head_idx], 200, binary=False, seed=seed + 1)
        heads[K] = {"concept": ch, "label": lh}
    heads["cal_pool_idx"] = rest_idx

    if with_probe:
        print("    linear probe (features -> 200)")
        heads["probe"] = fit_linear_classifier(Xtr, data.tr["labels"][head_idx], 200,
                                               binary=False, seed=seed + 2)
        report = {}
        with torch.no_grad():
            probe_logits = heads["probe"](data.te["X"])
            report["probe_test_acc"] = (probe_logits.argmax(1) == data.te["labels"]).float().mean().item()
            for K in (112, 312):
                key = "c112" if K == 112 else "c312"
                cprob_te = torch.sigmoid(heads[K]["concept"](data.te["X"]))
                cbm_logits = heads[K]["label"](cprob_te)
                report[f"cbm{K}_test_acc"] = (cbm_logits.argmax(1) == data.te["labels"]).float().mean().item()
                report[f"concept{K}_micro_acc"] = ((cprob_te > 0.5).float() == data.te[key]).float().mean().item()
        print(f"    probe {report['probe_test_acc']:.3f} (ref {REF['probe_test_acc']}) | "
              f"CBM-312 {report['cbm312_test_acc']:.3f} (ref {REF['cbm312_test_acc']}) | "
              f"CBM-112 {report['cbm112_test_acc']:.3f} (ref labelhead-on-pred {REF['labelhead112_on_pred_acc']})")
        torch.save({f"c{K}": heads[K]["concept"].state_dict() for K in (112, 312)}
                   | {f"y{K}": heads[K]["label"].state_dict() for K in (112, 312)}
                   | {"probe": heads["probe"].state_dict()},
                   CACHE / "cub_heads.pt")
        return report
    return heads


def load_heads(data: Data) -> dict:
    sd = torch.load(CACHE / "cub_heads.pt", map_location="cpu", weights_only=False)
    heads = {}
    for K in (112, 312):
        ch = torch.nn.Linear(data.tr["X"].shape[1], K)
        ch.load_state_dict(sd[f"c{K}"])
        lh = torch.nn.Linear(K, 200)
        lh.load_state_dict(sd[f"y{K}"])
        heads[K] = {"concept": ch.eval(), "label": lh.eval()}
    return heads


# ─────────────────────── conformal machinery ───────────────────────
# quantile_per_column / image_score_topm: imported from ccm.conformal (aliased above).


def concept_metrics(concept_probs: torch.Tensor, truth: torch.Tensor, thresholds) -> dict:
    """thresholds: per-concept tensor (len K) or scalar (single-threshold methods).
    Sets: keep value v if s(v) <= q. s(1) = 1-p, s(0) = p."""
    if not torch.is_tensor(thresholds):
        thresholds = torch.as_tensor(thresholds)
    q = thresholds.view(1, -1) if thresholds.numel() == concept_probs.shape[1] else thresholds.view(1, 1)
    keep1 = (1.0 - concept_probs) <= q
    keep0 = concept_probs <= q
    miss = torch.where(truth > 0.5, ~keep1, ~keep0)
    n_miss = miss.sum(1).float()
    ambiguous = (keep0 & keep1).sum(1).float()
    empty = (~keep0 & ~keep1).float().mean(0)  # per-concept empty rate; reported as mean
    return {
        "joint_cov": (n_miss == 0).float().mean().item(),
        "avg_misses": n_miss.mean().item(),
        "ambiguous": ambiguous.mean().item(),
        "empty_rate": empty.mean().item(),
        "p_at_most": {m: (n_miss <= m).float().mean().item() for m in AT_MOST_GRID},
    }


def aps_label_coverage_and_sets(label_logits_cal, y_cal, label_logits_eval, y_eval, alpha, seed):
    """Calibrate APS on cal, return eval coverage, avg set size, and class masks.
    NO empty-set filling: randomized APS exactness requires it (filling with top-1
    inflates coverage by ~P(empty)*P(top1 correct | empty) ~= +5 points)."""
    score = APSScore(from_logits=True, randomized=True)
    s_cal = score(label_logits_cal, y_cal, random_seed=seed)
    q = compute_conformal_quantile(s_cal, alpha)
    s_all = score(label_logits_eval, None, random_seed=seed + 10_000)
    mask = s_all <= q  # (N_eval, 200)
    cov = mask[torch.arange(len(y_eval)), y_eval].float().mean().item()
    size = mask.sum(1).float().mean().item()
    return cov, size, mask, q


def aps_empty_set_diagnostic(label_logits_cal, y_cal, label_logits_eval, y_eval, alpha, seed):
    """Decompose the empty-set fill effect: P(empty) and P(top1 correct | empty)."""
    score = APSScore(from_logits=True, randomized=True)
    s_cal = score(label_logits_cal, y_cal, random_seed=seed)
    q = compute_conformal_quantile(s_cal, alpha)
    s_all = score(label_logits_eval, None, random_seed=seed + 10_000)
    mask = s_all <= q
    empty = mask.sum(1) == 0
    p_empty = empty.float().mean().item()
    if empty.any():
        top1_correct = (label_logits_eval[empty].argmax(1) == y_eval[empty]).float().mean().item()
    else:
        top1_correct = float("nan")
    cov_nofill = mask[torch.arange(len(y_eval)), y_eval].float().mean().item()
    cov_fill = cov_nofill + p_empty * top1_correct
    return {"p_empty": p_empty, "top1_acc_given_empty": top1_correct,
            "cov_nofill": cov_nofill, "cov_with_fill_approx": cov_fill}


def induced_sets(mask: torch.Tensor, class_concept: torch.Tensor):
    """mask: (N,200) label-set mask; class_concept: (200,K) in {0,1}.
    Thin wrapper over ccm.conformal.induced_concept_sets keeping the gate script's
    (ambiguous, any1, any0) return convention."""
    cps = induced_concept_sets(mask, class_concept)
    any1, any0 = cps.binary_mask[..., 1], cps.binary_mask[..., 0]
    ambiguous = (any1 & any0).sum(1).float()
    return ambiguous, any1, any0


# ───────────────────────────── stages ─────────────────────────────

def stage_check1(data: Data, heads: dict, n_draws: int = 100, out: dict | None = None):
    print("\n" + "=" * 70)
    print("STAGE check1: same-pool APS coverage (explains the 95.2% anomaly)")
    print("=" * 70)
    res = {}

    label_logits_te = heads[312]["label"](torch.sigmoid(heads[312]["concept"](data.te["X"])))
    y_te = data.te["labels"]

    # (0) empty-set fill decomposition on one draw (the suspected cause of the 95.2% anomaly)
    set_seed(0)
    perm0 = torch.randperm(len(y_te)); cal0, ev0 = perm0[:3600], perm0[3600:]
    diag = aps_empty_set_diagnostic(label_logits_te[cal0], y_te[cal0],
                                    label_logits_te[ev0], y_te[ev0], 0.10, 0)
    print(f"  empty-set diagnostic: P(empty)={diag['p_empty']:.4f}, "
          f"P(top1 correct | empty)={diag['top1_acc_given_empty']:.4f}, "
          f"cov no-fill={diag['cov_nofill']:.4f}, with-fill~{diag['cov_with_fill_approx']:.4f}")
    res["empty_set_diagnostic"] = diag

    # (a) same-pool: cal + eval both from official test
    pool_logits, pool_y = label_logits_te, y_te
    for n_cal in (600, 3600):
        covs, sizes = [], []
        for seed in range(n_draws):
            set_seed(seed)
            perm = torch.randperm(len(pool_y))
            cal, ev = perm[:n_cal], perm[n_cal:]
            cov, size, _, _ = aps_label_coverage_and_sets(
                pool_logits[cal], pool_y[cal], pool_logits[ev], pool_y[ev], 0.10, seed)
            covs.append(cov); sizes.append(size)
        k = math.ceil((n_cal + 1) * 0.9)
        beta_sd = math.sqrt(k * (n_cal + 1 - k) / ((n_cal + 1) ** 2 * (n_cal + 2)))
        res[f"same_pool_n{n_cal}"] = {
            "mean_cov": float(np.mean(covs)), "sd_cov": float(np.std(covs)),
            "beta_sd": beta_sd, "mean_size": float(np.mean(sizes)),
        }
        print(f"  same-pool n_cal={n_cal}: coverage {np.mean(covs):.4f} ± {np.std(covs):.4f} "
              f"(Beta sd {beta_sd:.4f}) | set size {np.mean(sizes):.2f}")

    # (b) official-shift: cal = fixed random 600 of val (the '600 calibration rows'), eval on test
    label_logits_va = heads[312]["label"](torch.sigmoid(heads[312]["concept"](data.va["X"])))
    y_va = data.va["labels"]
    g = torch.Generator().manual_seed(123)
    va600 = torch.randperm(len(y_va), generator=g)[:600]
    covs, sizes = [], []
    for seed in range(40):  # vary only APS randomization, mirroring the original 40 seeds
        cov, size, _, _ = aps_label_coverage_and_sets(
            label_logits_va[va600], y_va[va600], label_logits_te, y_te, 0.10, seed)
        covs.append(cov); sizes.append(size)
    res["official_val_cal_n600"] = {"mean_cov": float(np.mean(covs)), "sd_cov": float(np.std(covs)),
                                    "mean_size": float(np.mean(sizes))}
    print(f"  official (cal=600 from val, eval=test): coverage {np.mean(covs):.4f} ± {np.std(covs):.4f} "
          f"| set size {np.mean(sizes):.2f}   [original anomaly: {REF['aps_coverage_official_n600']} "
          f"@ size {REF['aps_set_size_official_n600']}]")

    # (c) official-style n=600: heads retrained on 4196 rows, cal = clean 600-row holdout, eval = test
    print("    retraining heads on 4196 rows for the clean n=600 official-style condition ...")
    hB = train_heads(data, seed=0, n_head_train=4196, with_probe=False)
    cal_idx = hB["cal_pool_idx"]
    logits_cal = hB[312]["label"](torch.sigmoid(hB[312]["concept"](data.tr["X"][cal_idx])))
    y_cal = data.tr["labels"][cal_idx]
    logits_ev = hB[312]["label"](torch.sigmoid(hB[312]["concept"](data.te["X"])))
    covs, sizes = [], []
    for seed in range(40):
        cov, size, _, _ = aps_label_coverage_and_sets(logits_cal, y_cal, logits_ev, y_te, 0.10, seed)
        covs.append(cov); sizes.append(size)
    res["official_trainholdout_cal_n600"] = {"mean_cov": float(np.mean(covs)), "sd_cov": float(np.std(covs)),
                                             "mean_size": float(np.mean(sizes))}
    print(f"  official-style (heads on 4196, cal=600 held-out train, eval=test): "
          f"coverage {np.mean(covs):.4f} ± {np.std(covs):.4f} | size {np.mean(sizes):.2f}")

    if out is not None:
        out["check1"] = res
    return res


def _gate_run(data, heads, K, cal_idx, ev_idx, alphas, m_grid, seed):
    """One split: returns per-method list of (alpha, metrics)."""
    key = "c112" if K == 112 else "c312"
    truth_all = data.te[key]
    probs_all = torch.sigmoid(heads[K]["concept"](data.te["X"]))
    label_logits_all = heads[K]["label"](probs_all)
    y_all = data.te["labels"]

    pc, pe = probs_all[cal_idx], probs_all[ev_idx]
    tc, te_ = truth_all[cal_idx], truth_all[ev_idx]
    s_true = torch.where(tc > 0.5, 1.0 - pc, pc)          # (n_cal, K) true-value scores
    s_true_ev = torch.where(te_ > 0.5, 1.0 - pe, pe)

    rows = []
    for alpha in alphas:
        # uncorrected (per-concept alpha)
        q_unc = quantile_per_column(s_true, alpha)
        rows.append(("uncorrected", alpha, concept_metrics(pe, te_, q_unc)))
        # Bonferroni (per-concept alpha/K)
        q_bon = quantile_per_column(s_true, alpha / K)
        rows.append(("bonferroni", alpha, concept_metrics(pe, te_, q_bon)))
        # global max & (m+1)-th largest (single threshold)
        for m in m_grid:
            t = image_score_topm(s_true, m)
            q = compute_conformal_quantile(t, alpha)
            rows.append((f"topm_m{m}", alpha, concept_metrics(pe, te_, q)))
        # induced from label APS set
        cov, size, mask, _ = aps_label_coverage_and_sets(
            label_logits_all[cal_idx], y_all[cal_idx],
            label_logits_all[ev_idx], y_all[ev_idx], alpha, seed)
        amb, any1, any0 = induced_sets(mask, data.class_concept_112 if K == 112 else data.class_concept_112)
        if K != 112:
            pass  # induced only defined for class-denoised (constant per class)
        miss = torch.where(te_ > 0.5, ~any1, ~any0) if K == 112 else torch.full_like(te_, float("nan"))
        n_miss = miss.sum(1).float()
        rows.append(("induced", alpha, {
            "joint_cov": (n_miss == 0).float().mean().item() if K == 112 else float("nan"),
            "avg_misses": n_miss.mean().item() if K == 112 else float("nan"),
            "ambiguous": amb.mean().item(),
            "empty_rate": 0.0,
            "label_cov": cov, "label_size": size,
            "p_at_most": {m: (n_miss <= m).float().mean().item() for m in AT_MOST_GRID} if K == 112 else {},
        }))
    return rows


def stage_gate(data: Data, heads: dict, n_draws: int = 20, out: dict | None = None):
    print("\n" + "=" * 70)
    print("STAGE gate: joint concept comparison on 112 class-denoised (n_cal=3600, same-pool)")
    print("=" * 70)
    K = 112
    n_cal = 3600
    all_rows = []  # (draw, method, alpha, metrics)
    for seed in range(n_draws):
        set_seed(seed)
        perm = torch.randperm(len(data.te["labels"]))
        cal_idx, ev_idx = perm[:n_cal], perm[n_cal:]
        rows = _gate_run(data, heads, K, cal_idx, ev_idx, ALPHA_GRID, M_GRID, seed)
        for method, alpha, met in rows:
            all_rows.append({"draw": seed, "method": method, "alpha": alpha, **met})
        if (seed + 1) % 5 == 0:
            print(f"    draw {seed+1}/{n_draws} done")

    # aggregate: mean ± sd over draws per (method, alpha)
    agg = {}
    for r in all_rows:
        agg.setdefault((r["method"], r["alpha"]), []).append(r)
    table = []
    for (method, alpha), rs in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        jc = [x["joint_cov"] for x in rs]
        amb = [x["ambiguous"] for x in rs]
        table.append({"method": method, "alpha": alpha,
                      "joint_cov": float(np.mean(jc)), "joint_cov_sd": float(np.std(jc)),
                      "ambiguous": float(np.mean(amb)), "ambiguous_sd": float(np.std(amb)),
                      "label_cov": float(np.mean([x.get("label_cov", float("nan")) for x in rs])),
                      "label_size": float(np.mean([x.get("label_size", float("nan")) for x in rs])),
                      "p_at_most_8": float(np.mean([x["p_at_most"].get(8, float("nan")) for x in rs]))})

    # matched-coverage interpolation: ambiguous at realized joint coverage = 0.90
    def frontier(method):
        pts = sorted([(t["joint_cov"], t["ambiguous"]) for t in table if t["method"] == method])
        return pts

    def amb_at(method, target=0.90):
        pts = frontier(method)
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        if not xs:
            return None
        return float(np.interp(target, xs, ys))

    print(f"\n  {'method':<14} {'alpha':>5} {'joint_cov':>9} {'ambig':>7} {'P(≤8 wrong)':>11}")
    for t in table:
        print(f"  {t['method']:<14} {t['alpha']:>5.2f} {t['joint_cov']:>9.3f} "
              f"{t['ambiguous']:>7.1f} {t['p_at_most_8']:>11.3f}")

    print("\n  Ambiguous concepts per image at MATCHED joint coverage 0.90:")
    matched = {m: amb_at(m, 0.90) for m in ["bonferroni", "topm_m0", "topm_m1", "topm_m2",
                                            "topm_m4", "topm_m8", "induced"]}
    for m, v in matched.items():
        print(f"    {m:<14} {v if v is None else round(v, 1)}")

    if out is not None:
        out["gate"] = {"table": table, "matched_at_090": matched}
    return table, matched


def stage_table(data: Data, heads: dict, out: dict | None = None):
    """Existing 4-row joint table under (a) official-shift cal=train-subset, (b) same-pool."""
    print("\n" + "=" * 70)
    print("STAGE table: existing CUB joint table — official-shift vs same-pool (n_cal=3600)")
    print("=" * 70)
    set_seed(0)
    res = {}
    # Official-style condition: heads trained on 1196 rows, calibration on the clean
    # 3600-row remainder (train distribution), eval on the shifted official test.
    print("    retraining heads on 1196 rows for the official-style condition ...")
    hC = train_heads(data, seed=0, n_head_train=1196, with_probe=False)
    cal_off_idx = hC["cal_pool_idx"]  # 3600 rows, disjoint from head training
    for K in (312, 112):
        key = "c112" if K == 112 else "c312"
        probs_all = torch.sigmoid(heads[K]["concept"](data.te["X"]))
        truth_all = data.te[key]
        # official-style: heads + cal on train distribution, eval = official test
        off_cal_p = torch.sigmoid(hC[K]["concept"](data.tr["X"][cal_off_idx]))
        off_cal_t = data.tr[key][cal_off_idx]
        off_ev_p = torch.sigmoid(hC[K]["concept"](data.te["X"]))
        off_ev_t = data.te[key]
        s_cal = torch.where(off_cal_t > 0.5, 1.0 - off_cal_p, off_cal_p)
        # same-pool: cal = 3600 from test pool, eval = rest
        perm = torch.randperm(len(truth_all))
        cal_i, ev_i = perm[:3600], perm[3600:]
        sp_cal_p, sp_cal_t = probs_all[cal_i], truth_all[cal_i]
        sp_ev_p, sp_ev_t = probs_all[ev_i], truth_all[ev_i]

        for method in ("bonferroni", "global_max"):
            alpha = 0.10
            # official-style: threshold from train-pool cal, metrics on official test
            if method == "bonferroni":
                q = quantile_per_column(s_cal, alpha / K)
            else:
                q = compute_conformal_quantile(image_score_topm(s_cal, 0), alpha)
            m_off = concept_metrics(off_ev_p, off_ev_t, q)
            # same pool
            s_sp = torch.where(sp_cal_t > 0.5, 1.0 - sp_cal_p, sp_cal_p)
            if method == "bonferroni":
                q2 = quantile_per_column(s_sp, alpha / K)
            else:
                q2 = compute_conformal_quantile(image_score_topm(s_sp, 0), alpha)
            m_sp = concept_metrics(sp_ev_p, sp_ev_t, q2)
            res[f"K{K}_{method}"] = {"official": {k: v for k, v in m_off.items() if k != "p_at_most"},
                                     "same_pool": {k: v for k, v in m_sp.items() if k != "p_at_most"}}
            print(f"  K={K} {method:>12}: official cov {m_off['joint_cov']:.3f} ambig {m_off['ambiguous']:.1f}"
                  f"   |  same-pool cov {m_sp['joint_cov']:.3f} ambig {m_sp['ambiguous']:.1f}")
    if out is not None:
        out["table"] = res
    return res


# ─────────────────────────────── main ─────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["heads", "check1", "gate", "table", "all"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_draws_check1", type=int, default=100)
    ap.add_argument("--n_draws_gate", type=int, default=20)
    args = ap.parse_args()

    data = Data()
    out = {}
    if RESULTS_JSON.exists():
        out = json.loads(RESULTS_JSON.read_text())

    if args.stage in ("heads", "all") or not (CACHE / "cub_heads.pt").exists():
        out["heads"] = train_heads(data, seed=args.seed)
        RESULTS_JSON.write_text(json.dumps(out, indent=2))
    heads = load_heads(data)

    if args.stage in ("check1", "all"):
        stage_check1(data, heads, n_draws=args.n_draws_check1, out=out)
        RESULTS_JSON.write_text(json.dumps(out, indent=2))
    if args.stage in ("gate", "all"):
        stage_gate(data, heads, n_draws=args.n_draws_gate, out=out)
        RESULTS_JSON.write_text(json.dumps(out, indent=2))
    if args.stage in ("table", "all"):
        stage_table(data, heads, out=out)
        RESULTS_JSON.write_text(json.dumps(out, indent=2))

    print(f"\n[*] Results saved to {RESULTS_JSON}")


if __name__ == "__main__":
    main()
