"""
Evaluation metrics for Conformal Concept Models.
Coverage, set size efficiency, conditional coverage, concept accuracy,
intervention efficiency AUC, and Nearest-Neighbor leakage diagnostics.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from ccm.sets import PredictionSet, ConceptPredictionSet


def empirical_coverage(
    sets: Union[PredictionSet, ConceptPredictionSet],
    targets: torch.Tensor,
) -> float:
    """Computes empirical marginal coverage."""
    return float(sets.coverage(targets)) if hasattr(sets, "coverage") else float(sets.marginal_coverage(targets))


def average_set_size(
    sets: Union[PredictionSet, ConceptPredictionSet],
) -> float:
    """Computes average set cardinality."""
    return float(sets.avg_size) if hasattr(sets, "avg_size") else float(sets.sizes.float().mean().item())


def empty_set_fraction(sets: PredictionSet) -> float:
    """Fraction of empty prediction sets."""
    return float(sets.empty_mask.float().mean().item())


def singleton_set_fraction(sets: PredictionSet) -> float:
    """Fraction of singleton prediction sets (cardinality == 1)."""
    return float(sets.singleton_mask.float().mean().item())


def set_size_distribution(sets: PredictionSet) -> Dict[str, float]:
    """Computes distribution statistics of set cardinalities."""
    sizes = sets.sizes.float().cpu().numpy()
    return {
        "mean": float(np.mean(sizes)),
        "median": float(np.median(sizes)),
        "std": float(np.std(sizes)),
        "min": float(np.min(sizes)),
        "max": float(np.max(sizes)),
        "p90": float(np.percentile(sizes, 90)),
        "p95": float(np.percentile(sizes, 95)),
    }


def conditional_coverage_by_group(
    sets: PredictionSet,
    targets: torch.Tensor,
    groups: torch.Tensor,
) -> Dict[int, float]:
    """
    Computes empirical coverage stratified by subgroup (e.g., class or difficulty).
    Returns dict mapping group_id -> coverage.
    """
    contained = sets.contains(targets).cpu()
    groups_np = groups.cpu().numpy()
    
    unique_groups = np.unique(groups_np)
    group_cov = {}
    for g in unique_groups:
        g_mask = groups_np == g
        if np.sum(g_mask) > 0:
            group_cov[int(g)] = float(contained[g_mask].float().mean().item())
    return group_cov


def concept_accuracy(
    concept_preds: torch.Tensor,
    concept_targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluates concept bottleneck prediction quality.
    Supports binary concept probabilities or continuous predictions.
    """
    c_preds = concept_preds.detach().cpu()
    c_targets = concept_targets.detach().cpu()

    # Binary concepts
    if (c_targets >= 0).all() and (c_targets <= 1).all():
        bin_preds = (c_preds > threshold).float()
        bin_targets = (c_targets > threshold).float()
        
        correct = (bin_preds == bin_targets).float()
        per_concept = correct.mean(dim=0).tolist()
        micro_acc = float(correct.mean().item())
        macro_acc = float(np.mean(per_concept))
        
        return {
            "micro_acc": micro_acc,
            "macro_acc": macro_acc,
            "per_concept": per_concept,
        }
    else:
        # Continuous concepts (MSE / MAE)
        mae = float(torch.abs(c_preds - c_targets).mean().item())
        mse = float(torch.mean((c_preds - c_targets) ** 2).item())
        return {
            "mae": mae,
            "mse": mse,
            "micro_acc": max(0.0, 1.0 - mae),
        }


def intervention_efficiency_auc(accuracies: List[float]) -> float:
    """
    Computes normalized Area Under the Intervention Curve (AUC-IC).
    Normalized to [0, 1] across steps 0 to T.
    """
    if not accuracies:
        return 0.0
    if len(accuracies) == 1:
        return accuracies[0]
    
    steps = np.linspace(0, 1, len(accuracies))
    auc = np.trapz(accuracies, steps)
    return float(auc)


def nn_leakage_score(
    representations: torch.Tensor,
    concepts: torch.Tensor,
    targets: torch.Tensor,
    n_neighbors: int = 5,
    n_splits: int = 5,
    seed: int = 0,
    n_control_projections: int = 1,
) -> Dict[str, Any]:
    """
    Nearest-Neighbor Concept Leakage Diagnostic (v2; fixes the 2026-11-07 review).

    Compares label predictability from latent representations z vs concepts c.
    The gap is SIGNED: positive = z predicts y better than c (leakage direction),
    negative = the bottleneck is more predictive than the latent (v1 clipped this
    to 0, reporting "no leakage" when the concepts outperformed the latent).

    The gap is NOT judged against 0: z and c live in different dimensions and
    scales, so part of any gap is distance-geometry artifact. The control projects
    z randomly to dim(c) -- identical information content, c's dimensionality --
    giving a null gap attributable to geometry alone. Leakage evidence is
    excess_gap_vs_control = gap - control_gap, judged against the fold CI.

    Protocol fixes over v1:
      - stratified K-fold (class-balanced, shuffled) instead of an index-order
        70/30 split, which on class-sorted data tests kNN on unseen classes;
      - standardization fitted on the train fold only (v1 fitted on all rows,
        a mild transductive leak).
    """
    z_np = representations.detach().cpu().numpy()
    c_np = concepts.detach().cpu().numpy()
    y_np = targets.detach().cpu().numpy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rng = np.random.RandomState(seed)
    d_z, d_c = z_np.shape[1], c_np.shape[1]
    projections = [rng.randn(d_z, d_c) / np.sqrt(d_c) for _ in range(max(0, n_control_projections))]

    accs_z, accs_c, agree = [], [], []
    ctrl_gaps = []  # per (fold, projection): acc_z_fold - acc_ctrl, the geometry null
    for train_idx, test_idx in skf.split(z_np, y_np):
        # Standardization fitted on the train fold only.
        z_mu, z_sd = z_np[train_idx].mean(axis=0), z_np[train_idx].std(axis=0) + 1e-6
        c_mu, c_sd = c_np[train_idx].mean(axis=0), c_np[train_idx].std(axis=0) + 1e-6
        z_tr, z_te = (z_np[train_idx] - z_mu) / z_sd, (z_np[test_idx] - z_mu) / z_sd
        c_tr, c_te = (c_np[train_idx] - c_mu) / c_sd, (c_np[test_idx] - c_mu) / c_sd
        y_tr, y_te = y_np[train_idx], y_np[test_idx]

        preds_z = KNeighborsClassifier(n_neighbors=n_neighbors).fit(z_tr, y_tr).predict(z_te)
        preds_c = KNeighborsClassifier(n_neighbors=n_neighbors).fit(c_tr, y_tr).predict(c_te)
        acc_z_f = float((preds_z == y_te).mean())
        accs_z.append(acc_z_f)
        accs_c.append(float((preds_c == y_te).mean()))
        agree.append(float((preds_z == preds_c).mean()))
        for proj in projections:
            acc_p = float(KNeighborsClassifier(n_neighbors=n_neighbors)
                          .fit(z_tr @ proj, y_tr)
                          .score(z_te @ proj, y_te))
            ctrl_gaps.append(acc_z_f - acc_p)

    accs_z_arr, accs_c_arr = np.asarray(accs_z), np.asarray(accs_c)
    gaps = accs_z_arr - accs_c_arr
    gap_mean = float(gaps.mean())
    gap_sd = float(gaps.std(ddof=1)) if n_splits > 1 else 0.0
    ci_half = 1.96 * gap_sd / np.sqrt(n_splits)
    control_gap = float(np.mean(ctrl_gaps)) if ctrl_gaps else float("nan")

    return {
        "nn_acc_latent": float(accs_z_arr.mean()),
        "nn_acc_concept": float(accs_c_arr.mean()),
        "leakage_gap": gap_mean,  # SIGNED: negative = concepts beat latent
        "leakage_gap_sd": gap_sd,
        "leakage_gap_ci95": [gap_mean - ci_half, gap_mean + ci_half],
        "leakage_ratio": float(accs_z_arr.mean() / (accs_c_arr.mean() + 1e-6)),
        "nn_agreement_rate": float(np.mean(agree)),
        "control_gap": control_gap,
        "excess_gap_vs_control": gap_mean - control_gap,
    }


def format_metrics_table(metrics: Dict[str, Any]) -> str:
    """Formats a clean markdown table from a metrics dictionary."""
    lines = [
        "| Metric | Value |",
        "| :--- | :--- |",
    ]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.4f} |")
        else:
            lines.append(f"| {k} | {v} |")
    return "\n".join(lines)
