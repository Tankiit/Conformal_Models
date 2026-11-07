"""
Conformal prediction calibration engines for Conformal Concept Models.
Provides standard split conformal prediction, APS, RAPS, class/group-conditional (Mondrian),
and concept-bottleneck calibration with rigorous finite-sample coverage guarantees.
"""

from __future__ import annotations
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ccm.scores import NonconformityScore, SoftmaxScore, APSScore, RAPSScore, ConceptResidualScore, get_score_function
from ccm.sets import PredictionSet, ConceptPredictionSet


def compute_conformal_quantile(
    scores: torch.Tensor,
    alpha: float,
) -> float:
    """
    Computes finite-sample (1 - alpha) conformal quantile:
    q_hat = s_{(k)} where k = ceil((n + 1) * (1 - alpha))
    (Angelopoulos & Bates, 2021; Vovk et al., 2005)
    """
    scores_flat = scores.detach().flatten()
    n = scores_flat.numel()
    if n == 0:
        raise ValueError("Cannot compute quantile on empty tensor")
    
    # 1-based index of conformal quantile
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        # No order statistic of n points reaches the (n+1)(1-alpha) quantile;
        # the only valid choice is +infinity (yields a vacuous set, honestly).
        return float("inf")
    if k <= 1:
        return float(torch.min(scores_flat).item())
    
    sorted_scores, _ = torch.sort(scores_flat)
    # k is 1-based, so in 0-based indexing we take index k - 1
    q_val = float(sorted_scores[k - 1].item())
    return q_val


def conformal_quantile_per_column(scores: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Exact per-column (per-concept) conformal quantiles for an [n, K] score matrix:
    q_j = s_(k) with k = ceil((n + 1)(1 - alpha)); +inf if k > n.
    Passing alpha / K yields the Bonferroni joint threshold vector.
    """
    n = scores.shape[0]
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return torch.full((scores.shape[1],), float("inf"))
    sorted_scores, _ = torch.sort(scores, dim=0)
    return sorted_scores[k - 1, :]


def m_plus_one_score(per_concept_scores: torch.Tensor, m: int) -> torch.Tensor:
    """
    Row-wise (m+1)-th largest per-concept score: [B, K] -> [B].
    Calibrating this score at level alpha gives the exact finite-sample guarantee
        P(at most m wrong concepts) >= 1 - alpha
    under exchangeability alone (no independence across concepts needed).

    m = 0 reduces to the row max, i.e. the "all concepts right" joint claim
    (global-max threshold). m >= K clamps to the row min, which is conservative:
    at most K-1 concepts can be wrong, so the at-most-m claim then holds trivially.
    """
    sorted_desc, _ = torch.sort(per_concept_scores, dim=-1, descending=True)
    idx = min(m, sorted_desc.shape[1] - 1)
    return sorted_desc[:, idx]


def induced_concept_sets(
    label_set_mask: torch.Tensor,
    class_concept_table: torch.Tensor,
) -> ConceptPredictionSet:
    """
    Concept sets induced by conformal label sets when concept labels are
    class-denoised (constant within each class). Value v of concept j is
    possible iff some class in the label set has concept j = v.

    Why joint coverage equals label-set coverage here (the collapse): denoised
    labels make C = g(Y), so if the true class is in the label set, every true
    concept value appears in the induced set -- joint coverage >= label coverage
    always. The converse holds empirically (0.900 vs 0.898 on CUB-112) because
    g is injective on classes and label sets are small: a few wrong classes
    essentially never jointly cover all concept bits. Only defined for
    class-denoised tables; on raw per-image labels there is no such table.
    """
    any1 = (label_set_mask.float() @ class_concept_table) > 0
    any0 = (label_set_mask.float() @ (1.0 - class_concept_table)) > 0
    return ConceptPredictionSet(binary_mask=torch.stack([any0, any1], dim=-1))


class ConformalCalibrator:
    """
    Standard Split Conformal Prediction Calibrator for classification and concept bottlenecks.
    Supports arbitrary non-conformity scores (Softmax, APS, RAPS, Margin).
    """

    def __init__(
        self,
        score_fn: Optional[NonconformityScore] = None,
        alpha: float = 0.10,
    ):
        self.score_fn = score_fn or APSScore()
        self.default_alpha = alpha
        self.q_hat: Optional[float] = None
        self.calibrated_alphas: Dict[float, float] = {}

    def calibrate(
        self,
        logits_or_probs: torch.Tensor,
        targets: torch.Tensor,
        alpha: Optional[float] = None,
    ) -> float:
        """
        Calibrates on held-out calibration set (logits/probs, ground-truth targets).
        Returns the calibrated threshold q_hat.
        """
        alpha_val = alpha if alpha is not None else self.default_alpha
        calib_scores = self.score_fn(logits_or_probs, targets)
        q_hat = compute_conformal_quantile(calib_scores, alpha_val)
        
        self.calibrated_alphas[alpha_val] = q_hat
        if alpha is None or alpha == self.default_alpha:
            self.q_hat = q_hat
        return q_hat

    def predict(
        self,
        logits_or_probs: torch.Tensor,
        alpha: Optional[float] = None,
        fill_empty: bool = False,
    ) -> PredictionSet:
        """
        Constructs prediction sets for test data:
        C(x) = { y : s(x, y) <= q_hat }

        fill_empty=False (default) returns the raw conformal sets: empty sets are
        possible, and the measured quantity matches the calibrated statement
        P(y in C(x)) >= 1 - alpha. Any number reported as "coverage" must come
        from this mode (settled decision D3, 2026-11-06).
        fill_empty=True is a deployment convenience ("always hand the user a
        usable prediction") that replaces empty sets with the top-1 class; it
        inflates measured coverage by ~ P(empty) * P(top-1 correct | empty)
        (+5-6 pts at alpha=0.10 on CUB) and must never be used for coverage
        claims.
        """
        alpha_val = alpha if alpha is not None else self.default_alpha
        if alpha_val in self.calibrated_alphas:
            threshold = self.calibrated_alphas[alpha_val]
        elif self.q_hat is not None:
            threshold = self.q_hat
        else:
            raise RuntimeError("Calibrator has not been calibrated yet. Call calibrate() first.")

        # Compute scores for all candidate classes: [B, K]
        all_class_scores = self.score_fn(logits_or_probs, targets=None)
        
        # Inclusion condition: s(x, y) <= threshold
        mask = all_class_scores <= threshold
        pred_set = PredictionSet(mask)

        if fill_empty:
            pred_set = pred_set.fill_empty_with_top1(logits_or_probs)

        return pred_set

    def calibrate_concepts(
        self,
        concept_probs: torch.Tensor,
        concept_targets: torch.Tensor,
        alpha: Optional[float] = None,
        joint: bool = False,
        joint_m: Optional[int] = None,
    ) -> Union[torch.Tensor, float]:
        """
        Calibrates conformal sets for concept bottleneck predictions.
        If joint=True, computes a single threshold over all concepts simultaneously:
          - joint_m=None: max-score threshold -> P(all concepts right) >= 1 - alpha
          - joint_m=m:    (m+1)-th-largest threshold -> P(at most m wrong) >= 1 - alpha
        If joint=False, calibrates independent thresholds per concept (marginal).
        """
        alpha_val = alpha if alpha is not None else self.default_alpha
        concept_score_fn = ConceptResidualScore(continuous=False)

        # Shape: [B, num_concepts]
        per_concept_scores = concept_score_fn(concept_probs, concept_targets)
        B, num_concepts = per_concept_scores.shape

        if joint:
            image_scores = m_plus_one_score(per_concept_scores, joint_m if joint_m is not None else 0)
            q_joint = compute_conformal_quantile(image_scores, alpha_val)
            self.concept_q_joint = q_joint
            return q_joint
        else:
            # Per-concept thresholds
            thresholds = torch.zeros(num_concepts, device=concept_probs.device)
            for j in range(num_concepts):
                scores_j = per_concept_scores[:, j]
                thresholds[j] = compute_conformal_quantile(scores_j, alpha_val)
            self.concept_q_per_concept = thresholds
            return thresholds

    def predict_concepts(
        self,
        concept_probs: torch.Tensor,
        thresholds: Optional[Union[torch.Tensor, float]] = None,
    ) -> ConceptPredictionSet:
        """
        Constructs prediction sets for concept bottleneck predictions.
        """
        if thresholds is None:
            if hasattr(self, "concept_q_per_concept"):
                thresholds = self.concept_q_per_concept
            elif hasattr(self, "concept_q_joint"):
                thresholds = self.concept_q_joint
            else:
                raise RuntimeError("Concept thresholds not calibrated yet.")

        concept_score_fn = ConceptResidualScore(continuous=False)
        # Shape: [B, num_concepts, 2] (score for 0, score for 1)
        all_concept_scores = concept_score_fn(concept_probs, targets=None)

        if isinstance(thresholds, (float, int)):
            mask = all_concept_scores <= thresholds
        else:
            # thresholds is [num_concepts]
            thresh_tensor = thresholds.to(concept_probs.device).view(1, -1, 1)
            mask = all_concept_scores <= thresh_tensor

        return ConceptPredictionSet(binary_mask=mask)


class MondrianConformalCalibrator:
    """
    Mondrian / Group-Conditional Conformal Calibrator.
    Calibrates group-specific thresholds {q_g} to guarantee conditional coverage:
        P(Y in C(X) | Group = g) >= 1 - alpha for each group g.
    Groups can represent classes, concept difficulty strata, or demographic groups.
    """

    def __init__(
        self,
        score_fn: Optional[NonconformityScore] = None,
        alpha: float = 0.10,
        min_group_size: int = 10,
    ):
        self.score_fn = score_fn or APSScore()
        self.alpha = alpha
        self.min_group_size = min_group_size
        self.group_thresholds: Dict[int, float] = {}
        self.default_threshold: float = 1.0
        # Groups with < min_group_size calibration points: they fell back to the
        # global threshold, so only the MARGINAL guarantee holds for them.
        self.fallback_groups: Dict[int, int] = {}

    def calibrate(
        self,
        logits_or_probs: torch.Tensor,
        targets: torch.Tensor,
        groups: torch.Tensor,
        alpha: Optional[float] = None,
    ) -> Dict[int, float]:
        alpha_val = alpha or self.alpha
        calib_scores = self.score_fn(logits_or_probs, targets)

        # Global fallback threshold
        self.default_threshold = compute_conformal_quantile(calib_scores, alpha_val)
        self.fallback_groups = {}

        unique_groups = torch.unique(groups).tolist()
        for g in unique_groups:
            g_mask = groups == g
            g_scores = calib_scores[g_mask]
            if g_scores.numel() >= self.min_group_size:
                self.group_thresholds[int(g)] = compute_conformal_quantile(g_scores, alpha_val)
            else:
                # Too few calibration points for a group-conditional quantile.
                # These groups keep only the marginal guarantee; any paper claim
                # of conditional coverage must exclude them or merge them under
                # a rule fixed before calibration (open decision D6).
                self.group_thresholds[int(g)] = self.default_threshold
                self.fallback_groups[int(g)] = int(g_scores.numel())

        if self.fallback_groups:
            warnings.warn(
                "Mondrian calibrator: groups "
                f"{sorted(self.fallback_groups.keys())} have < {self.min_group_size} "
                "calibration points and fall back to the GLOBAL threshold; the "
                "group-conditional guarantee does not hold for them. See "
                "self.fallback_groups (group -> calibration count).",
                stacklevel=2,
            )
        return self.group_thresholds

    def predict(
        self,
        logits_or_probs: torch.Tensor,
        groups: torch.Tensor,
        fill_empty: bool = False,
    ) -> PredictionSet:
        """Same fill_empty semantics as ConformalCalibrator.predict (no-fill default)."""
        all_class_scores = self.score_fn(logits_or_probs, targets=None)
        B, K = all_class_scores.shape

        # Assemble per-sample threshold
        thresholds = torch.zeros(B, 1, device=logits_or_probs.device)
        for i in range(B):
            g = int(groups[i].item())
            thresholds[i] = self.group_thresholds.get(g, self.default_threshold)

        mask = all_class_scores <= thresholds
        pred_set = PredictionSet(mask)
        if fill_empty:
            pred_set = pred_set.fill_empty_with_top1(logits_or_probs)
        return pred_set
