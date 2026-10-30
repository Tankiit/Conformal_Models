"""
Instance difficulty estimation models and scoring for conditional and normalized conformal prediction.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SampleDifficultyEstimator(nn.Module):
    """
    Estimates instance-level difficulty sigma(x) used for:
      1. Normalized non-conformity scores: s(x, y) / sigma(x)
      2. Group-conditional / Mondrian conformal prediction (stratified into Easy, Medium, Hard).
    """

    def __init__(self, mode: str = "concept_entropy", num_strata: int = 3):
        super().__init__()
        self.mode = mode
        self.num_strata = num_strata
        self.strata_quantiles: Optional[torch.Tensor] = None

    def compute_difficulty(
        self,
        task_logits: Optional[torch.Tensor] = None,
        concept_probs: Optional[torch.Tensor] = None,
        latent_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes continuous difficulty scores [B]. Higher score means more difficult.
        """
        if self.mode == "concept_entropy" and concept_probs is not None:
            p = torch.clamp(concept_probs, 1e-6, 1.0 - 1e-6)
            ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
            return ent.mean(dim=-1)

        elif self.mode == "task_entropy" and task_logits is not None:
            p = F.softmax(task_logits, dim=-1)
            p = torch.clamp(p, 1e-6, 1.0)
            return -(p * torch.log(p)).sum(dim=-1)

        elif self.mode == "margin" and task_logits is not None:
            p = F.softmax(task_logits, dim=-1)
            top2_vals, _ = torch.topk(p, k=min(2, p.shape[-1]), dim=-1)
            if top2_vals.shape[-1] >= 2:
                margin = top2_vals[:, 0] - top2_vals[:, 1]
            else:
                margin = top2_vals[:, 0]
            return 1.0 - margin

        elif self.mode == "latent_norm" and latent_features is not None:
            return torch.norm(latent_features, dim=-1)

        else:
            # Fallback uniform difficulty
            B = task_logits.shape[0] if task_logits is not None else 1
            device = task_logits.device if task_logits is not None else torch.device("cpu")
            return torch.ones(B, device=device)

    def fit_strata(self, difficulties: torch.Tensor) -> torch.Tensor:
        """
        Computes quantiles across calibration data to discretize difficulty into strata.
        """
        quantiles = torch.linspace(0.0, 1.0, self.num_strata + 1)[1:-1]
        self.strata_quantiles = torch.quantile(difficulties.float().cpu(), quantiles)
        return self.strata_quantiles

    def assign_strata(self, difficulties: torch.Tensor) -> torch.Tensor:
        """
        Assigns each difficulty score to a discrete group index {0, 1, ..., num_strata - 1}.
        """
        if self.strata_quantiles is None:
            self.fit_strata(difficulties)

        diffs = difficulties.cpu()
        strata = torch.zeros(diffs.shape[0], dtype=torch.long)
        for q_val in self.strata_quantiles:
            strata += (diffs > q_val).long()
        return strata.to(difficulties.device)
