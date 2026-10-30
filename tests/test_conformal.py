"""
Unit tests for Conformal Prediction engines and score functions.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from ccm.conformal import compute_conformal_quantile, ConformalCalibrator, MondrianConformalCalibrator
from ccm.scores import SoftmaxScore, APSScore, RAPSScore, MarginScore, NormalizedScore, ConceptResidualScore


def test_conformal_quantile_formula():
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    n = 10
    alpha = 0.10
    # ceil((n + 1) * (1 - alpha)) / n = ceil(11 * 0.9) / 10 = ceil(9.9) / 10 = 10 / 10 = 1.0
    q = compute_conformal_quantile(scores, alpha)
    assert q == pytest.approx(1.0, abs=1e-5)

    alpha = 0.20
    # ceil(11 * 0.8) / 10 = ceil(8.8) / 10 = 9 / 10 = 0.9
    q = compute_conformal_quantile(scores, alpha)
    assert q == pytest.approx(0.9, abs=1e-5)


def test_score_functions_shapes():
    B, K = 32, 5
    logits = torch.randn(B, K)
    targets = torch.randint(0, K, (B,))

    for ScoreClass in [SoftmaxScore, APSScore, RAPSScore, MarginScore]:
        fn = ScoreClass(from_logits=True)
        # All classes
        scores_all = fn(logits, targets=None)
        assert scores_all.shape == (B, K)
        # Target scores
        scores_targets = fn(logits, targets=targets)
        assert scores_targets.shape == (B,)


def test_marginal_coverage_guarantee():
    torch.manual_seed(42)
    num_calib = 1000
    num_test = 1000
    num_classes = 4

    # Generate synthetic softmax probabilities
    calib_probs = F.softmax(torch.randn(num_calib, num_classes) * 1.5, dim=-1)
    calib_targets = torch.multinomial(calib_probs, num_samples=1).squeeze(-1)

    test_probs = F.softmax(torch.randn(num_test, num_classes) * 1.5, dim=-1)
    test_targets = torch.multinomial(test_probs, num_samples=1).squeeze(-1)

    alpha = 0.10
    calibrator = ConformalCalibrator(score_fn=APSScore(from_logits=False, randomized=False), alpha=alpha)
    calibrator.calibrate(calib_probs, calib_targets, alpha=alpha)

    pred_sets = calibrator.predict(test_probs, alpha=alpha, fill_empty=True)
    cov = pred_sets.coverage(test_targets)

    # Empirical coverage should be >= 1 - alpha - small tolerance
    assert cov >= (1.0 - alpha - 0.035), f"Expected coverage >= 0.865, got {cov}"


def test_raps_regularization_reduces_size():
    torch.manual_seed(42)
    B, K = 500, 20
    logits = torch.randn(B, K)
    targets = torch.randint(0, K, (B,))

    calib_logits, test_logits = logits[:250], logits[250:]
    calib_y, test_y = targets[:250], targets[250:]

    aps_calib = ConformalCalibrator(score_fn=APSScore(from_logits=True, randomized=False), alpha=0.1)
    aps_calib.calibrate(calib_logits, calib_y)
    aps_sets = aps_calib.predict(test_logits)

    raps_calib = ConformalCalibrator(score_fn=RAPSScore(from_logits=True, lambda_reg=0.05, k_reg=2, randomized=False), alpha=0.1)
    raps_calib.calibrate(calib_logits, calib_y)
    raps_sets = raps_calib.predict(test_logits)

    assert raps_sets.avg_size <= aps_sets.avg_size + 0.5


def test_concept_conformal_calibration():
    torch.manual_seed(42)
    B, C = 200, 6
    probs = torch.sigmoid(torch.randn(B, C))
    targets = (probs > 0.4).float()

    calibrator = ConformalCalibrator()
    thresholds = calibrator.calibrate_concepts(probs[:100], targets[:100], alpha=0.15)
    c_sets = calibrator.predict_concepts(probs[100:], thresholds)

    cov = c_sets.marginal_coverage(targets[100:])
    assert cov >= 0.80


def test_mondrian_calibrator():
    torch.manual_seed(42)
    B, K = 300, 3
    logits = torch.randn(B, K)
    targets = torch.randint(0, K, (B,))
    groups = torch.randint(0, 2, (B,))  # 2 groups

    calibrator = MondrianConformalCalibrator(score_fn=SoftmaxScore(from_logits=True), alpha=0.10)
    calibrator.calibrate(logits[:150], targets[:150], groups[:150])
    pred_sets = calibrator.predict(logits[150:], groups[150:])

    cov = pred_sets.coverage(targets[150:])
    assert cov >= 0.80
