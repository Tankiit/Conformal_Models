"""
Prediction set containers and set propagation operations for Conformal Concept Models.
"""

from __future__ import annotations
import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class PredictionSet:
    """
    Container for discrete multi-class prediction sets C(x) subset of {0, ..., K-1}.
    Represented internally as a boolean mask [B, K].
    """

    def __init__(
        self,
        mask: torch.Tensor,
        class_names: Optional[List[str]] = None,
    ):
        if mask.dtype != torch.bool:
            mask = mask.bool()
        self.mask = mask
        self.batch_size, self.num_classes = mask.shape
        self.class_names = class_names

    @property
    def sizes(self) -> torch.Tensor:
        """Returns tensor of set cardinalities [B]."""
        return self.mask.sum(dim=-1)

    @property
    def avg_size(self) -> float:
        """Mean set cardinality across batch."""
        return float(self.sizes.float().mean().item())

    @property
    def empty_mask(self) -> torch.Tensor:
        """Boolean mask indicating empty prediction sets [B]."""
        return self.sizes == 0

    @property
    def singleton_mask(self) -> torch.Tensor:
        """Boolean mask indicating singleton prediction sets [B]."""
        return self.sizes == 1

    def contains(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Checks whether true target labels are included in the prediction sets.
        Returns: boolean tensor of shape [B].
        """
        targets = targets.long().to(self.mask.device)
        batch_idx = torch.arange(self.batch_size, device=self.mask.device)
        return self.mask[batch_idx, targets]

    def coverage(self, targets: torch.Tensor) -> float:
        """Computes empirical marginal coverage fraction."""
        contained = self.contains(targets)
        return float(contained.float().mean().item())

    def to_label_list(self) -> List[List[int]]:
        """Converts mask tensor to Python list of label index lists."""
        result = []
        for i in range(self.batch_size):
            included = torch.where(self.mask[i])[0].tolist()
            result.append(included)
        return result

    def fill_empty_with_top1(self, probs_or_logits: torch.Tensor) -> PredictionSet:
        """
        Returns a new PredictionSet where any empty set is replaced by the argmax prediction.
        """
        new_mask = self.mask.clone()
        empty = self.empty_mask
        if empty.any():
            top1 = torch.argmax(probs_or_logits, dim=-1)
            new_mask[empty, top1[empty]] = True
        return PredictionSet(new_mask, class_names=self.class_names)

    def __len__(self) -> int:
        return self.batch_size

    def __getitem__(self, idx: Union[int, slice, torch.Tensor]) -> PredictionSet:
        return PredictionSet(self.mask[idx], class_names=self.class_names)


class ConceptPredictionSet:
    """
    Container for concept prediction sets C_C(x).
    Supports:
      - Binary multi-label concepts: mask [B, num_concepts, 2] where dim -1 holds {included in 0, included in 1}
      - Continuous concepts: intervals [lower, upper] with shape [B, num_concepts]
    """

    def __init__(
        self,
        binary_mask: Optional[torch.Tensor] = None,
        intervals: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        concept_names: Optional[List[str]] = None,
    ):
        self.binary_mask = binary_mask.bool() if binary_mask is not None else None
        self.intervals = intervals
        self.concept_names = concept_names

        if self.binary_mask is not None:
            self.batch_size, self.num_concepts, _ = self.binary_mask.shape
        elif self.intervals is not None:
            self.batch_size, self.num_concepts = self.intervals[0].shape
        else:
            raise ValueError("Must provide either binary_mask or intervals")

    @property
    def sizes(self) -> torch.Tensor:
        """
        Returns tensor of set cardinalities per concept [B, num_concepts] (0, 1, or 2 for binary).
        For continuous, returns interval widths (upper - lower).
        """
        if self.binary_mask is not None:
            return self.binary_mask.sum(dim=-1)
        else:
            return self.intervals[1] - self.intervals[0]

    def contains(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Checks if concept targets are contained within prediction sets/intervals.
        Returns: boolean tensor [B, num_concepts].
        """
        targets = targets.to(self.binary_mask.device if self.binary_mask is not None else self.intervals[0].device)
        if self.binary_mask is not None:
            targets_bin = (targets > 0.5).long()
            B, C = targets_bin.shape
            batch_idx = torch.arange(B, device=targets_bin.device).unsqueeze(1).expand(B, C)
            concept_idx = torch.arange(C, device=targets_bin.device).unsqueeze(0).expand(B, C)
            return self.binary_mask[batch_idx, concept_idx, targets_bin]
        else:
            lower, upper = self.intervals
            return (targets >= lower) & (targets <= upper)

    def marginal_coverage(self, targets: torch.Tensor) -> float:
        """Average coverage across all individual concepts and samples."""
        return float(self.contains(targets).float().mean().item())

    def joint_coverage(self, targets: torch.Tensor) -> float:
        """Fraction of samples where ALL concepts are simultaneously covered."""
        contained_all = self.contains(targets).all(dim=-1)
        return float(contained_all.float().mean().item())

    def get_uncertain_concepts(self, threshold_size: float = 1.0) -> torch.Tensor:
        """
        Returns boolean tensor [B, num_concepts] where prediction set size > threshold_size
        (e.g., size is 2 in binary case).
        """
        return self.sizes > threshold_size

    def __len__(self) -> int:
        return self.batch_size


def propagate_concept_sets_to_task(
    c_to_y_model: torch.nn.Module,
    concept_set: ConceptPredictionSet,
    max_combinations: int = 64,
) -> PredictionSet:
    """
    Propagates concept uncertainty through the concept-to-task predictor c -> y.
    For each sample, iterates over the Cartesian product of possible concept configurations
    in C_C(x) (up to max_combinations) and computes the union of predicted task classes:
        C_Y(x) = Union_{c in C_C(x)} { argmax f_Y(c) }
    """
    device = next(c_to_y_model.parameters()).device
    B = len(concept_set)

    with torch.no_grad():
        dummy_c = torch.zeros(1, concept_set.num_concepts, device=device)
        dummy_out = c_to_y_model(dummy_c)
        num_classes = dummy_out.shape[-1]

    task_mask = torch.zeros(B, num_classes, dtype=torch.bool, device=device)

    for i in range(B):
        # Generate possible concept configurations for sample i
        possible_values_per_concept = []
        for j in range(concept_set.num_concepts):
            if concept_set.binary_mask is not None:
                incl = torch.where(concept_set.binary_mask[i, j])[0].tolist()
                possible_values_per_concept.append(incl if len(incl) > 0 else [0.0])
            else:
                low = concept_set.intervals[0][i, j].item()
                high = concept_set.intervals[1][i, j].item()
                possible_values_per_concept.append([low, high, (low + high) / 2])

        # Cartesian product
        combos = list(itertools.product(*possible_values_per_concept))
        if len(combos) > max_combinations:
            combos = combos[:max_combinations]

        c_tensor = torch.tensor(combos, dtype=torch.float32, device=device)
        with torch.no_grad():
            task_logits = c_to_y_model(c_tensor)
            pred_classes = torch.argmax(task_logits, dim=-1)
            task_mask[i, pred_classes] = True

    return PredictionSet(task_mask)
