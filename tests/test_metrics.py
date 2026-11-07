"""
Unit tests for the v3 Nearest-Neighbor leakage diagnostic (2026-11-07 review rounds):
signed gap, stratified K-fold with fold-fitted scaling, and the embedding-null
calibration (v2's single-projection subtraction reduced algebraically to
acc_proj - acc_c and was not a null; see nn_leakage_score docstring).
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


def test_positive_gap_far_beyond_null_when_latent_leaks():
    torch.manual_seed(1)
    n = 600
    y = _block_classes()
    y_onehot = torch.nn.functional.one_hot(y, 3).float()
    # Isotropic label signal in z, spread over all dims via a random embedding.
    z = y_onehot @ torch.randn(3, 32) + 0.05 * torch.randn(n, 32)
    c = torch.randn(n, 6)                         # concepts carry nothing
    m = nn_leakage_score(z, c, y, n_neighbors=5, n_splits=3, seed=0,
                         n_control_projections=2, n_null_draws=4)
    assert m["leakage_gap"] > 0.5
    # null: z re-synthesized from c alone carries nothing -> null band ~ 0
    assert m["gap_vs_null_z"] > 5


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


def test_null_centered_when_concepts_match_latent():
    torch.manual_seed(3)
    n = 600
    y = _block_classes()
    y_onehot = torch.nn.functional.one_hot(y, 3).float()
    # Isotropic signal in z, then c = a fixed random projection of z to 6 dims:
    # c holds everything z knows, so the observed gap must sit inside the null
    # band (z re-synthesized from c carries the same information).
    z = y_onehot @ torch.randn(3, 24) + 0.05 * torch.randn(n, 24)
    proj = torch.randn(24, 6) / (6 ** 0.5)
    c = z @ proj                                   # same information, c's dimension
    m = nn_leakage_score(z, c, y, n_neighbors=5, n_splits=3, seed=0,
                         n_control_projections=2, n_null_draws=4)
    assert abs(m["leakage_gap"] - m["null_gap_mean"]) < 0.05
    assert abs(m["gap_vs_null_z"]) < 5


def test_output_contract():
    torch.manual_seed(4)
    y = _block_classes(n_per_class=50)
    m = nn_leakage_score(torch.randn(150, 8), torch.randn(150, 4), y,
                         n_splits=3, seed=0, n_control_projections=1, n_null_draws=2)
    for key in ("nn_acc_latent", "nn_acc_concept", "leakage_gap", "leakage_gap_sd",
                "leakage_ratio", "nn_agreement_rate", "projection_gap_mean",
                "projection_gap_sd", "null_gap_mean", "null_gap_sd", "gap_vs_null_z"):
        assert key in m
    lo, hi = m["leakage_gap_ci95"]
    assert lo <= m["leakage_gap"] <= hi
