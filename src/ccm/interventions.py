"""
Concept intervention policies and simulation for Conformal Concept Models.
Allows evaluating downstream accuracy and conformal set size compression under
budgeted test-time concept interventions.
"""

from __future__ import annotations
import abc
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ccm.metrics import intervention_efficiency_auc


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
                sets = calibrator.predict(logits_y)
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
