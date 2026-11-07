"""
Unit tests for the v2 Nearest-Neighbor leakage diagnostic (2026-11-07 review fixes):
signed gap, stratified K-fold with fold-fitted scaling, random-projection control.
"""

import pytest
import torch

from ccm.metrics import nn_leakage_score


def _block_classes(n_per_class=200, n_classes=3):
    """Class-SORTED labels: the layout that broke the v1 index-order 70/30 split."""
    return torch.repeat_interleave(torch.arange(n_classes), n_per_class)


def test_signed_gap_negative_when_concepts_beat_latent():
    torch.manual_seed(0)
    n = 600
    y = _block_classes()
    z = torch.randn(n, 32)                       # latent carries nothing
    y_onehot = torch.nn.functional.one_hot(y, 3).float()
    c = y_onehot * 2.0 + 0.1 * torch.randn(n, 3)  # concepts carry the label
    m = nn_leakage_score(z, c, y, n_neighbors=5, n_splits=3, seed=0)
    # v1 clipped this to 0.0 and reported "no leakage"
    assert m["leakage_gap"] < -0.5


def test_positive_excess_gap_when_latent_leaks():
    torch.manual_seed(1)
    n = 600
    y = _block_classes()
    y_onehot = torch.nn.functional.one_hot(y, 3).float()
    # Isotropic label signal: spread over all 32 dims via a random embedding, so a
    # random projection to 6 dims preserves it (control gap ~ 0; excess == gap).
    z = y_onehot @ torch.randn(3, 32) + 0.05 * torch.randn(n, 32)
    c = torch.randn(n, 6)                         # concepts carry nothing
    m = nn_leakage_score(z, c, y, n_neighbors=5, n_splits=3, seed=0)
    assert m["leakage_gap"] > 0.5
    assert m["excess_gap_vs_control"] > 0.3


def test_stratified_folds_survive_class_sorted_data():
    torch.manual_seed(2)
    n = 600
    y = _block_classes()
    y_onehot = torch.nn.functional.one_hot(y, 3).float()
    z = torch.randn(n, 16)
    z[:, :3] += y_onehot * 4.0
    c = torch.randn(n, 6)
    c[:, :3] += y_onehot * 4.0
    m = nn_leakage_score(z, c, y, n_neighbors=5, n_splits=3, seed=0)
    # v1's index-order split tested kNN on classes it never saw -> ~0 accuracy
    assert m["nn_acc_latent"] > 0.9
    assert m["nn_acc_concept"] > 0.9


def test_control_null_when_concepts_are_projection_of_latent():
    torch.manual_seed(3)
    n = 600
    y = _block_classes()
    y_onehot = torch.nn.functional.one_hot(y, 3).float()
    # Isotropic signal in z, then c = a fixed random projection of z to 6 dims:
    # c holds the same information as the control projections, so the excess gap
    # over the control band must be ~0 (no leakage beyond geometry, by symmetry).
    z = y_onehot @ torch.randn(3, 24) + 0.05 * torch.randn(n, 24)
    proj = torch.randn(24, 6) / (6 ** 0.5)
    c = z @ proj                                   # same information, c's dimension
    m = nn_leakage_score(z, c, y, n_neighbors=5, n_splits=3, seed=0,
                         n_control_projections=3)
    # Nothing leaks beyond geometry: excess gap over the control band is ~0
    assert abs(m["excess_gap_vs_control"]) < 0.1


def test_output_contract():
    torch.manual_seed(4)
    y = _block_classes(n_per_class=50)
    m = nn_leakage_score(torch.randn(150, 8), torch.randn(150, 4), y,
                         n_splits=3, seed=0, n_control_projections=1)
    for key in ("nn_acc_latent", "nn_acc_concept", "leakage_gap", "leakage_gap_sd",
                "leakage_ratio", "nn_agreement_rate", "control_gap",
                "excess_gap_vs_control"):
        assert key in m
    lo, hi = m["leakage_gap_ci95"]
    assert lo <= m["leakage_gap"] <= hi
