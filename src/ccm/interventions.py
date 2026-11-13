"""
Concept intervention policies and simulation for Conformal Concept Models.
Allows evaluating downstream accuracy and conformal set size compression under
budgeted test-time concept interventions.
"""

from __future__ import annotations
import abc
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ccm.metrics import intervention_efficiency_auc
from ccm.scores import APSScore, NonconformityScore


class InterventionPolicy(abc.ABC):
    """Abstract base class for selecting which concepts to intervene upon."""

    @abc.abstractmethod
    def select_concepts(
        self,
        predicted_concepts: torch.Tensor,
        true_concepts: Optional[torch.Tensor],
        budget: int,
        cbm_model: Optional[torch.nn.Module] = None,
        x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns a boolean mask of shape [B, num_concepts] where True denotes
        the concept should be intervened on (overwritten with ground truth).
        """
        pass


class RandomInterventionPolicy(InterventionPolicy):
    """Randomly selects 'budget' concepts per sample."""

    def select_concepts(
        self,
        predicted_concepts: torch.Tensor,
        true_concepts: Optional[torch.Tensor] = None,
        budget: int = 1,
        cbm_model: Optional[torch.nn.Module] = None,
        x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C = predicted_concepts.shape
        budget = min(budget, C)
        mask = torch.zeros(B, C, dtype=torch.bool, device=predicted_concepts.device)
        for i in range(B):
            idx = torch.randperm(C)[:budget]
            mask[i, idx] = True
        return mask


class UncertaintyInterventionPolicy(InterventionPolicy):
    """
    Selects concepts with maximum predictive entropy (uncertainty):
    H(p) = -p log(p) - (1-p) log(1-p).
    """

    def select_concepts(
        self,
        predicted_concepts: torch.Tensor,
        true_concepts: Optional[torch.Tensor] = None,
        budget: int = 1,
        cbm_model: Optional[torch.nn.Module] = None,
        x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        probs = torch.sigmoid(predicted_concepts) if (predicted_concepts < 0).any() or (predicted_concepts > 1).any() else predicted_concepts
        probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
        # Binary entropy: -p log p - (1-p) log (1-p)
        entropy = -(probs * torch.log(probs) + (1.0 - probs) * torch.log(1.0 - probs))

        B, C = predicted_concepts.shape
        budget = min(budget, C)
        _, top_idx = torch.topk(entropy, k=budget, dim=-1)

        mask = torch.zeros(B, C, dtype=torch.bool, device=predicted_concepts.device)
        mask.scatter_(1, top_idx, True)
        return mask


class OracleOptimalInterventionPolicy(InterventionPolicy):
    """
    Oracle policy: intervenes on concepts where the model's prediction error is largest.
    Provides upper-bound benchmark on intervention effectiveness.
    """

    def select_concepts(
        self,
        predicted_concepts: torch.Tensor,
        true_concepts: Optional[torch.Tensor] = None,
        budget: int = 1,
        cbm_model: Optional[torch.nn.Module] = None,
        x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if true_concepts is None:
            raise ValueError("Oracle policy requires ground truth concepts")

        probs = torch.sigmoid(predicted_concepts) if (predicted_concepts < 0).any() or (predicted_concepts > 1).any() else predicted_concepts
        errors = torch.abs(probs - true_concepts.float())

        B, C = predicted_concepts.shape
        budget = min(budget, C)
        _, top_idx = torch.topk(errors, k=budget, dim=-1)

        mask = torch.zeros(B, C, dtype=torch.bool, device=predicted_concepts.device)
        mask.scatter_(1, top_idx, True)
        return mask


class GreedyConformalInterventionPolicy(InterventionPolicy):
    """
    Conformal-guided intervention policy:
    Intervenes on the concept that results in the largest reduction in downstream task prediction entropy
    or maximizes downstream confidence.
    """

    def select_concepts(
        self,
        predicted_concepts: torch.Tensor,
        true_concepts: Optional[torch.Tensor] = None,
        budget: int = 1,
        cbm_model: Optional[torch.nn.Module] = None,
        x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cbm_model is None or not hasattr(cbm_model, "c_to_y"):
            # Fallback to uncertainty policy if model is not passed
            fallback = UncertaintyInterventionPolicy()
            return fallback.select_concepts(predicted_concepts, true_concepts, budget)

        B, C = predicted_concepts.shape
        budget = min(budget, C)
        device = predicted_concepts.device

        # If true concepts unavailable, use hard thresholded concepts
        ground_truth = true_concepts.float() if true_concepts is not None else (predicted_concepts > 0.5).float()

        selected_mask = torch.zeros(B, C, dtype=torch.bool, device=device)

        # Greedy forward selection
        for step in range(budget):
            best_gain = torch.full((B,), -float("inf"), device=device)
            best_concept = torch.zeros(B, dtype=torch.long, device=device)

            for j in range(C):
                # Candidate concept vector
                candidate_c = predicted_concepts.clone()
                # Apply current selections + candidate j
                step_mask = selected_mask.clone()
                step_mask[:, j] = True
                candidate_c = torch.where(step_mask, ground_truth, candidate_c)

                with torch.no_grad():
                    logits_y = cbm_model.c_to_y(candidate_c)
                    probs_y = F.softmax(logits_y, dim=-1)
                    # Confidence gain (max probability)
                    conf, _ = torch.max(probs_y, dim=-1)

                improvement = conf
                better = (improvement > best_gain) & (~selected_mask[:, j])
                best_gain = torch.where(better, improvement, best_gain)
                best_concept = torch.where(better, torch.tensor(j, device=device), best_concept)

            selected_mask.scatter_(1, best_concept.unsqueeze(1), True)

        return selected_mask


def apply_interventions(
    cbm_model: torch.nn.Module,
    x: torch.Tensor,
    true_concepts: torch.Tensor,
    policy: InterventionPolicy,
    budget: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Executes concept intervention on batch x.
    Returns:
      - task_logits: logits for downstream target after intervention
      - intervened_concepts: concept values after replacement
      - intervention_mask: boolean mask indicating which concepts were modified
    """
    with torch.no_grad():
        initial_concepts, _ = cbm_model.predict_concepts(x)

    mask = policy.select_concepts(
        predicted_concepts=initial_concepts,
        true_concepts=true_concepts,
        budget=budget,
        cbm_model=cbm_model,
        x=x,
    )

    intervened_c = torch.where(mask, true_concepts.float(), initial_concepts)
    with torch.no_grad():
        task_logits = cbm_model.c_to_y(intervened_c)

    return task_logits, intervened_c, mask


def evaluate_intervention_curve(
    cbm_model: torch.nn.Module,
    calibrator: Any,
    data_loader: Any,
    policy: InterventionPolicy,
    max_budget: int = 5,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Simulates intervention budgets from 0 to max_budget concepts.
    Measures task accuracy and conformal set size at each intervention step.
    """
    dev = device or next(cbm_model.parameters()).device
    cbm_model.eval()

    accuracies = []
    avg_set_sizes = []
    coverages = []

    for budget in range(max_budget + 1):
        total_correct = 0
        total_samples = 0
        total_covered = 0
        total_set_size = 0.0

        for batch in data_loader:
            x, c_true, y_true = batch[0].to(dev), batch[1].to(dev), batch[2].to(dev)
            
            if budget == 0:
                with torch.no_grad():
                    logits_y, _ = cbm_model(x)
            else:
                logits_y, _, _ = apply_interventions(cbm_model, x, c_true, policy, budget)

            preds = torch.argmax(logits_y, dim=-1)
            total_correct += int((preds == y_true).sum().item())
            total_samples += len(y_true)

            if calibrator is not None:
                # No-fill: this loop reports coverage, and filling empty sets with
                # top-1 would inflate it by P(empty) * P(top1 | empty).
                sets = calibrator.predict(logits_y, fill_empty=False)
                total_covered += int(sets.contains(y_true).sum().item())
                total_set_size += float(sets.sizes.sum().item())

        acc = total_correct / total_samples
        accuracies.append(acc)

        if calibrator is not None:
            coverages.append(total_covered / total_samples)
            avg_set_sizes.append(total_set_size / total_samples)

    auc = intervention_efficiency_auc(accuracies)

    return {
        "budgets": list(range(max_budget + 1)),
        "accuracies": accuracies,
        "coverages": coverages if calibrator else [],
        "avg_set_sizes": avg_set_sizes if calibrator else [],
        "auc_accuracy": auc,
    }


def paired_set_change(
    cbm_model: torch.nn.Module,
    calibrator: Any,
    pairs: Iterable[Tuple[torch.Tensor, ...]],
    use_gt_concepts: bool,
    seed: int = 0,
    alpha: Optional[float] = None,
    score_fn: Optional[NonconformityScore] = None,
    substitution_labels: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Paired nuisance-flip evaluation for the CLeaR leakage test (Shapes3D U-flip
    design; see the vault spec "Shapes3D Paired-Nuisance Generator Spec").

    For each pair (x, x_flip) -- identical declared concepts C, flipped hidden
    nuisance U -- measures whether the conformal label set changes. The APS
    randomization u is HELD FIXED within a pair (APSScore(uniforms=...)), so set
    changes cannot be randomization artifacts; identical inputs with shared u give
    identical sets, so the u-pairing null is exactly zero (verify with (x, x)
    pairs as a sanity check before the main grid).

    `pairs` batches may be (x, x_flip, c_true) or (x, x_flip, c_true, y_true).
    When labels are present the COST readouts are computed (2026-11-13 review):
    coverage under each wall, coverage under BOTH, directional loss/gain (how
    often breaking the shortcut REMOVES a previously covered true label), and --
    with `substitution_labels` (e.g. the same-shape/wrong-scale class Y^1 for
    the composite encoding) -- how often the shortcut substitute enters the set
    exactly where the true label was lost, plus the empty-set transition rate
    (no-fill APS makes empties possible).

    Variants (this IS the D5 choice -- it defines what the CLeaR claim means):
      - use_gt_concepts=True (Test 1): strict label head f(c) gives EXACTLY zero
        set-change by construction; any change reveals an x-path (residual).
      - use_gt_concepts=False (Test 2, deployed pipeline): zero set-change is
        NOT the leak-free signature; the null is the rho_train=0 measured rate.

    Returns dict with set_change_rate, mean_prob_shift_l1, per-pair tensors
    ("changed", "prob_shifts", and with labels also "contains_x",
    "contains_x_flip", "set_sizes_x", "set_sizes_x_flip"), n_pairs, the variant,
    and the cost readouts when labels are given.
    """
    # Default to the CALIBRATOR's own score: its threshold was calibrated under
    # that score, so silently substituting another one mismatches threshold and
    # scores (empty/garbage sets).
    score_fn = score_fn or getattr(calibrator, "score_fn", None) or APSScore(from_logits=True, randomized=True)
    alpha_val = alpha if alpha is not None else getattr(calibrator, "default_alpha", 0.10)
    if alpha_val in getattr(calibrator, "calibrated_alphas", {}):
        threshold = calibrator.calibrated_alphas[alpha_val]
    elif getattr(calibrator, "q_hat", None) is not None:
        threshold = calibrator.q_hat
    else:
        raise RuntimeError("Calibrator has not been calibrated yet.")

    device = next(cbm_model.parameters()).device
    cbm_model.eval()
    g = torch.Generator().manual_seed(seed)

    changed_all, shift_all = [], []
    contains_x_all, contains_f_all = [], []
    size_x_all, size_f_all = [], []
    sub_in_flip_all = []
    n_seen = 0
    with torch.no_grad():
        for batch in pairs:
            if len(batch) == 4:
                x, x_flip, c_true, y_true = batch
                y_true = y_true.to(device).long()
            else:
                x, x_flip, c_true = batch[:3]
                y_true = None
            x, x_flip, c_true = x.to(device), x_flip.to(device), c_true.to(device)
            B = x.shape[0]
            if use_gt_concepts:
                ones_mask = torch.ones_like(c_true, dtype=torch.bool)
                logits, _ = cbm_model(x, intervened_concepts=c_true, intervention_mask=ones_mask)
                logits_flip, _ = cbm_model(x_flip, intervened_concepts=c_true, intervention_mask=ones_mask)
            else:
                logits, _ = cbm_model(x)
                logits_flip, _ = cbm_model(x_flip)

            # Fixed u shared within each pair: both members use the same draw.
            u = torch.rand(B, 1, generator=g).to(device)
            both = torch.cat([logits, logits_flip], dim=0)
            if isinstance(score_fn, APSScore):
                scores = score_fn(both, None, uniforms=u.repeat(2, 1))
            else:
                scores = score_fn(both, None)
            mask = scores <= threshold
            mask_x, mask_flip = mask[:B], mask[B:]
            changed = (mask_x != mask_flip).any(dim=-1)

            p = F.softmax(logits, dim=-1)
            p_flip = F.softmax(logits_flip, dim=-1)
            shift = (p - p_flip).abs().sum(dim=-1)

            changed_all.append(changed.cpu())
            shift_all.append(shift.cpu())
            size_x_all.append(mask_x.sum(dim=-1).cpu())
            size_f_all.append(mask_flip.sum(dim=-1).cpu())
            if y_true is not None:
                batch_idx = torch.arange(B, device=device)
                contains_x_all.append(mask_x[batch_idx, y_true].cpu())
                contains_f_all.append(mask_flip[batch_idx, y_true].cpu())
                if substitution_labels is not None:
                    sub = substitution_labels[n_seen:n_seen + B].to(device).long()
                    sub_in_flip_all.append(mask_flip[batch_idx, sub].cpu())
            n_seen += B

    changed = torch.cat(changed_all)
    shifts = torch.cat(shift_all)
    sizes_x = torch.cat(size_x_all)
    sizes_flip = torch.cat(size_f_all)
    out: Dict[str, Any] = {
        "set_change_rate": float(changed.float().mean().item()),
        "mean_prob_shift_l1": float(shifts.mean().item()),
        "n_pairs": int(len(changed)),
        "changed": changed,
        "prob_shifts": shifts,
        "set_sizes_x": sizes_x,
        "set_sizes_x_flip": sizes_flip,
        "use_gt_concepts": use_gt_concepts,
    }
    if contains_x_all:
        cx = torch.cat(contains_x_all)
        cf = torch.cat(contains_f_all)
        lost = cx & ~cf  # covered under x, lost under the flipped nuisance
        gained = cf & ~cx
        out.update({
            "contains_x": cx,
            "contains_x_flip": cf,
            "coverage_x": float(cx.float().mean().item()),
            "coverage_x_flip": float(cf.float().mean().item()),
            "coverage_both": float((cx & cf).float().mean().item()),
            "directional_loss": float(lost.float().mean().item()),
            "directional_gain": float(gained.float().mean().item()),
        })
        if int(lost.sum()) > 0:
            out["empty_flip_given_lost"] = float((sizes_flip[lost] == 0).float().mean().item())
        if sub_in_flip_all:
            sub_mask = torch.cat(sub_in_flip_all)
            out["substitution_rate_given_lost"] = float(sub_mask[lost].float().mean().item()) if int(lost.sum()) > 0 else 0.0
    return out
