"""
Concept Bottleneck Model (CBM) architectures with support for sequential,
joint, and hybrid/residual training, test-time interventions, and latent inspection.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBackbone(nn.Module):
    """Convolutional feature extractor for image inputs."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MLPBackbone(nn.Module):
    """Simple MLP backbone for pre-extracted features or vector inputs."""

    def __init__(self, in_dim: int, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.net(x)


class ConceptBottleneckModel(nn.Module):
    """
    Standard and Hybrid Concept Bottleneck Model:
      x -> z (latent representation)
      z -> c (concept predictions)
      c [optional: + z_residual] -> y (task prediction)
    """

    def __init__(
        self,
        num_concepts: int,
        num_classes: int,
        backbone_type: str = "conv",
        in_channels: int = 3,
        latent_dim: int = 128,
        hybrid_residual: bool = False,
        residual_dim: int = 16,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.hybrid_residual = hybrid_residual

        # Backbone x -> z
        if backbone_type == "conv":
            self.backbone = ConvBackbone(in_channels=in_channels, latent_dim=latent_dim)
        else:
            self.backbone = MLPBackbone(in_dim=in_channels, latent_dim=latent_dim)

        # Concept head z -> c (logits)
        self.x_to_c = nn.Linear(latent_dim, num_concepts)

        # Optional residual leakage head for hybrid models
        if hybrid_residual:
            self.res_head = nn.Linear(latent_dim, residual_dim)
            task_in_dim = num_concepts + residual_dim
        else:
            self.res_head = None
            task_in_dim = num_concepts

        # Downstream task head c -> y
        self.c_to_y = nn.Sequential(
            nn.Linear(task_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def extract_representations(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts both latent embeddings z and concept logits c.
        Returns: (z, c_logits).
        """
        z = self.backbone(x)
        c_logits = self.x_to_c(z)
        return z, c_logits

    def predict_concepts(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns predicted concept probabilities / continuous values and concept logits.
        """
        z, c_logits = self.extract_representations(x)
        c_probs = torch.sigmoid(c_logits)
        return c_probs, c_logits

    def forward(
        self,
        x: torch.Tensor,
        intervened_concepts: Optional[torch.Tensor] = None,
        intervention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns: (task_logits, concept_logits).
        If intervention_mask is passed, replaces predicted concepts with intervened_concepts.
        """
        z, c_logits = self.extract_representations(x)
        c_acts = torch.sigmoid(c_logits)

        # Apply concept interventions if provided
        if intervention_mask is not None and intervened_concepts is not None:
            c_input = torch.where(intervention_mask, intervened_concepts.float(), c_acts)
        else:
            c_input = c_acts

        if self.hybrid_residual and self.res_head is not None:
            z_res = self.res_head(z)
            c_task_in = torch.cat([c_input, z_res], dim=-1)
        else:
            c_task_in = c_input

        task_logits = self.c_to_y(c_task_in)
        return task_logits, c_logits
