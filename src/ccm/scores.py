"""
Non-conformity score functions for Conformal Concept Models.
Implements Softmax, APS (Adaptive Prediction Sets), RAPS (Regularized APS),
Margin, Difficulty-Normalized, and Concept-specific residual scores.
"""

from __future__ import annotations
import abc
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class NonconformityScore(abc.ABC):
    """Abstract base class for non-conformity score functions."""

    @abc.abstractmethod
    def __call__(
        self,
        probs_or_logits: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes nonconformity scores.
        If targets is provided, returns scores of shape [batch_size].
        If targets is None, returns scores for all possible classes [batch_size, num_classes].
        """
        pass


class SoftmaxScore(NonconformityScore):
    """
    Standard Hinge / Softmax nonconformity score:
    s(x, y) = 1 - p(y | x)
    """

    def __init__(self, from_logits: bool = True):
        self.from_logits = from_logits

    def __call__(
        self,
        probs_or_logits: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        probs = F.softmax(probs_or_logits, dim=-1) if self.from_logits else probs_or_logits
        probs = torch.clamp(probs, min=1e-12, max=1.0)
        scores_all = 1.0 - probs
        if targets is None:
            return scores_all
        batch_idx = torch.arange(probs.size(0), device=probs.device)
        return scores_all[batch_idx, targets.long()]


HingeScore = SoftmaxScore


class APSScore(NonconformityScore):
    """
    Adaptive Prediction Sets (APS) score (Romano et al., 2020).
    Cumulative softmax probability of classes sorted descending until label y.
    Supports randomized tie-breaking with random uniform variable u in [0, 1].
    """

    def __init__(self, from_logits: bool = True, randomized: bool = True):
        self.from_logits = from_logits
        self.randomized = randomized

    def __call__(
        self,
        probs_or_logits: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        random_seed: Optional[int] = None,
    ) -> torch.Tensor:
        probs = F.softmax(probs_or_logits, dim=-1) if self.from_logits else probs_or_logits
        probs = torch.clamp(probs, min=1e-12, max=1.0)
        B, K = probs.shape

        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Rank of each class: rank_map[b, c] is the 0-indexed position of class c in sorted list
        rank_map = torch.empty_like(sorted_indices)
        rank_map.scatter_(1, sorted_indices, torch.arange(K, device=probs.device).repeat(B, 1))

        if self.randomized:
            if random_seed is not None:
                g = torch.Generator(device=probs.device).manual_seed(random_seed)
                u = torch.rand(B, 1, device=probs.device, generator=g)
            else:
                u = torch.rand(B, 1, device=probs.device)
        else:
            u = torch.ones(B, 1, device=probs.device)

        if targets is None:
            # Score for all classes
            # cum_probs_prev is cum_probs shifted right by 1 with 0 at the start
            cum_probs_prev = torch.cat([torch.zeros(B, 1, device=probs.device), cum_probs[:, :-1]], dim=-1)
            # Reorder back to original class indices
            orig_cum_prev = torch.empty_like(cum_probs_prev)
            orig_cum_prev.scatter_(1, sorted_indices, cum_probs_prev)
            return orig_cum_prev + u * probs
        else:
            targets = targets.long()
            batch_idx = torch.arange(B, device=probs.device)
            target_ranks = rank_map[batch_idx, targets]
            
            # cum_prob up to target
            target_cum = cum_probs[batch_idx, target_ranks]
            target_prob = probs[batch_idx, targets]
            return (target_cum - target_prob) + u.squeeze(1) * target_prob


class RAPSScore(NonconformityScore):
    """
    Regularized Adaptive Prediction Sets (RAPS) (Angelopoulos et al., 2021).
    Penalizes large set sizes via regularization parameter lambda_reg and k_reg:
    s(x, y) = cum_prob(y) + lambda_reg * max(0, rank(y) - k_reg) + u * p(y|x)
    """

    def __init__(
        self,
        from_logits: bool = True,
        lambda_reg: float = 0.01,
        k_reg: int = 5,
        randomized: bool = True,
    ):
        self.from_logits = from_logits
        self.lambda_reg = lambda_reg
        self.k_reg = k_reg
        self.randomized = randomized

    def __call__(
        self,
        probs_or_logits: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        random_seed: Optional[int] = None,
    ) -> torch.Tensor:
        probs = F.softmax(probs_or_logits, dim=-1) if self.from_logits else probs_or_logits
        probs = torch.clamp(probs, min=1e-12, max=1.0)
        B, K = probs.shape

        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        rank_map = torch.empty_like(sorted_indices)
        rank_map.scatter_(1, sorted_indices, torch.arange(K, device=probs.device).repeat(B, 1))

        if self.randomized:
            if random_seed is not None:
                g = torch.Generator(device=probs.device).manual_seed(random_seed)
                u = torch.rand(B, 1, device=probs.device, generator=g)
            else:
                u = torch.rand(B, 1, device=probs.device)
        else:
            u = torch.ones(B, 1, device=probs.device)

        # Penalty term for ranks
        ranks = torch.arange(K, device=probs.device).float()
        penalties = self.lambda_reg * torch.clamp(ranks - self.k_reg, min=0.0)

        if targets is None:
            cum_probs_prev = torch.cat([torch.zeros(B, 1, device=probs.device), cum_probs[:, :-1]], dim=-1)
            scores_sorted = cum_probs_prev + u * sorted_probs + penalties.unsqueeze(0)
            scores_orig = torch.empty_like(scores_sorted)
            scores_orig.scatter_(1, sorted_indices, scores_sorted)
            return scores_orig
        else:
            targets = targets.long()
            batch_idx = torch.arange(B, device=probs.device)
            target_ranks = rank_map[batch_idx, targets]
            target_cum = cum_probs[batch_idx, target_ranks]
            target_prob = probs[batch_idx, targets]
            target_pen = penalties[target_ranks]
            return (target_cum - target_prob) + u.squeeze(1) * target_prob + target_pen


class MarginScore(NonconformityScore):
    """
    Margin score:
    s(x, y) = max_{y' != y} p(y' | x) - p(y | x)
    """

    def __init__(self, from_logits: bool = True):
        self.from_logits = from_logits

    def __call__(
        self,
        probs_or_logits: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        probs = F.softmax(probs_or_logits, dim=-1) if self.from_logits else probs_or_logits
        B, K = probs.shape

        if targets is None:
            top2_vals, _ = torch.topk(probs, k=2, dim=-1)
            max_other = torch.where(
                probs == top2_vals[:, :1],
                top2_vals[:, 1:2],
                top2_vals[:, :1],
            )
            return max_other - probs
        else:
            targets = targets.long()
            batch_idx = torch.arange(B, device=probs.device)
            p_y = probs[batch_idx, targets]
            
            # Mask true label to find max of other classes
            masked_probs = probs.clone()
            masked_probs[batch_idx, targets] = -float("inf")
            max_other, _ = torch.max(masked_probs, dim=-1)
            return max_other - p_y


class NormalizedScore(NonconformityScore):
    """
    Normalized / Difficulty-weighted score:
    s(x, y) = base_score(x, y) / (difficulty(x) + eps)
    Adapts conformal threshold locally according to instance difficulty.
    """

    def __init__(
        self,
        base_score_fn: NonconformityScore,
        eps: float = 1e-4,
    ):
        self.base_score_fn = base_score_fn
        self.eps = eps

    def __call__(
        self,
        probs_or_logits: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        difficulty: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raw_scores = self.base_score_fn(probs_or_logits, targets)
        if difficulty is None:
            return raw_scores
        diff = difficulty.to(raw_scores.device)
        if diff.dim() == 1 and raw_scores.dim() == 2:
            diff = diff.unsqueeze(1)
        return raw_scores / (diff + self.eps)


class ConceptResidualScore(NonconformityScore):
    """
    Score function for concept predictions.
    Supports:
      - Continuous concepts: absolute residual |c - c_hat|
      - Binary / multi-label concepts: 1 - p(c_j | x)
    """

    def __init__(self, continuous: bool = False):
        self.continuous = continuous

    def __call__(
        self,
        preds_or_probs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.continuous:
            if targets is None:
                raise ValueError("Targets required for continuous concept residual score")
            return torch.abs(preds_or_probs - targets)
        else:
            # Binary probabilities in [0, 1]
            probs = torch.sigmoid(preds_or_probs) if preds_or_probs.min() < 0 or preds_or_probs.max() > 1 else preds_or_probs
            if targets is None:
                # Returns conformity for both 0 and 1: [B, num_concepts, 2]
                score_0 = probs  # s(c=0) = 1 - (1 - p) = p
                score_1 = 1.0 - probs  # s(c=1) = 1 - p
                return torch.stack([score_0, score_1], dim=-1)
            else:
                targets = targets.float()
                # If target is 1: 1 - probs. If target is 0: probs.
                return torch.where(targets > 0.5, 1.0 - probs, probs)


def get_score_function(name: str = "aps", **kwargs) -> NonconformityScore:
    """Factory helper to obtain a non-conformity score function."""
    name_clean = name.lower().strip()
    if name_clean in ("softmax", "hinge"):
        return SoftmaxScore(from_logits=kwargs.get("from_logits", True))
    elif name_clean == "aps":
        return APSScore(
            from_logits=kwargs.get("from_logits", True),
            randomized=kwargs.get("randomized", True),
        )
    elif name_clean == "raps":
        return RAPSScore(
            from_logits=kwargs.get("from_logits", True),
            lambda_reg=kwargs.get("lambda_reg", 0.01),
            k_reg=kwargs.get("k_reg", 5),
            randomized=kwargs.get("randomized", True),
        )
    elif name_clean == "margin":
        return MarginScore(from_logits=kwargs.get("from_logits", True))
    elif name_clean == "concept":
        return ConceptResidualScore(continuous=kwargs.get("continuous", False))
    else:
        raise ValueError(f"Unknown nonconformity score function: '{name}'")
