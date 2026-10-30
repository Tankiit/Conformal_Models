"""
Evaluation metrics for Conformal Concept Models.
Coverage, set size efficiency, conditional coverage, concept accuracy,
intervention efficiency AUC, and Nearest-Neighbor leakage diagnostics.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
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
) -> Dict[str, float]:
    """
    Nearest-Neighbor Concept Leakage Diagnostic.
    Compares label predictability from intermediate representations z vs concepts c:
    If z achieves substantially higher NN-accuracy than c, the representation leaks
    information about the task label that is not captured by the concept bottleneck.
    """
    z_np = representations.detach().cpu().numpy()
    c_np = concepts.detach().cpu().numpy()
    y_np = targets.detach().cpu().numpy()

    # Normalize features
    z_norm = (z_np - z_np.mean(axis=0)) / (z_np.std(axis=0) + 1e-6)
    c_norm = (c_np - c_np.mean(axis=0)) / (c_np.std(axis=0) + 1e-6)

    # Train k-NN on 70% split, test on 30%
    n_samples = len(y_np)
    split_idx = int(n_samples * 0.7)
    
    knn_z = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_c = KNeighborsClassifier(n_neighbors=n_neighbors)

    knn_z.fit(z_norm[:split_idx], y_np[:split_idx])
    knn_c.fit(c_norm[:split_idx], y_np[:split_idx])

    acc_z = float(knn_z.score(z_norm[split_idx:], y_np[split_idx:]))
    acc_c = float(knn_c.score(c_norm[split_idx:], y_np[split_idx:]))

    leakage_gap = max(0.0, acc_z - acc_c)
    leakage_ratio = acc_z / (acc_c + 1e-6)

    return {
        "nn_acc_latent": acc_z,
        "nn_acc_concept": acc_c,
        "leakage_gap": leakage_gap,
        "leakage_ratio": leakage_ratio,
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
