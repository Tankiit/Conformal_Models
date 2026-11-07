"""
Unit tests for Conformal Prediction engines and score functions.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from ccm.conformal import (
    compute_conformal_quantile,
    conformal_quantile_per_column,
    ConformalCalibrator,
    induced_concept_sets,
    MondrianConformalCalibrator,
    m_plus_one_score,
)
from ccm.models.difficulty import SampleDifficultyEstimator
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


def test_m_plus_one_score():
    scores = torch.tensor([[0.1, 0.9, 0.5, 0.3], [0.2, 0.4, 0.6, 0.8]])
    # m=0 is the row max ("all right" claim)
    assert torch.allclose(m_plus_one_score(scores, 0), torch.tensor([0.9, 0.8]))
    # m=1 is the second largest
    assert torch.allclose(m_plus_one_score(scores, 1), torch.tensor([0.5, 0.6]))
    # m >= K clamps to the row min (conservative)
    assert torch.allclose(m_plus_one_score(scores, 10), torch.tensor([0.1, 0.2]))


def test_conformal_quantile_per_column_matches_scalar():
    torch.manual_seed(0)
    scores = torch.rand(50, 4)
    q_col = conformal_quantile_per_column(scores, 0.10)
    for j in range(4):
        assert q_col[j].item() == pytest.approx(compute_conformal_quantile(scores[:, j], 0.10), abs=1e-6)


def test_induced_concept_sets_collapse():
    # Injective class -> concept table: 4 classes, 3 concepts, distinct codes.
    table = torch.tensor([[0., 0., 0.], [0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    # Label sets: row 0 = {true class 1}, row 1 = {2, 3} while truth is class 2,
    # row 2 = {0, 1, 3} while truth is class 3 (true class in set).
    mask = torch.zeros(3, 4, dtype=torch.bool)
    mask[0, 1] = True
    mask[1, 2] = True; mask[1, 3] = True
    mask[2, 0] = True; mask[2, 1] = True; mask[2, 3] = True
    truth = torch.stack([table[1], table[2], table[3]])

    cps = induced_concept_sets(mask, table)
    # True class in label set => every true concept value is induced => covered.
    assert cps.joint_coverage(truth) == 1.0
    # Singleton label set => no ambiguity at all.
    assert cps.get_uncertain_concepts()[0].sum() == 0
    # Codes 101 and 110 differ on concepts 1 and 2 => exactly two ambiguous concepts.
    assert cps.get_uncertain_concepts()[1].sum() == 2

    # True class NOT in set and codes differ everywhere => joint coverage 0.
    mask_wrong = torch.zeros(1, 4, dtype=torch.bool)
    mask_wrong[0, 0] = True  # set = {class 0}, truth = class 3 (code 110 vs 000)
    cps_wrong = induced_concept_sets(mask_wrong, table)
    assert cps_wrong.joint_coverage(table[3].unsqueeze(0)) == 0.0


def test_at_most_m_concept_calibration():
    torch.manual_seed(42)
    n_cal, n_ev, K = 2000, 1000, 20
    probs = torch.sigmoid(torch.randn(n_cal + n_ev, K) * 2.0)
    targets = (torch.rand(n_cal + n_ev, K) < probs).float()
    alpha, m = 0.10, 4

    calibrator = ConformalCalibrator(alpha=alpha)
    calibrator.calibrate_concepts(probs[:n_cal], targets[:n_cal], alpha=alpha, joint=True, joint_m=m)
    c_sets = calibrator.predict_concepts(probs[n_cal:])
    misses = (~c_sets.contains(targets[n_cal:])).sum(dim=-1)
    assert (misses <= m).float().mean().item() >= 1.0 - alpha - 0.03


def test_assign_strata_requires_fit():
    est = SampleDifficultyEstimator(mode="concept_entropy")
    with pytest.raises(RuntimeError, match="fit_strata"):
        est.assign_strata(torch.rand(10))


def test_mondrian_small_group_fallback_is_flagged():
    torch.manual_seed(42)
    B, K = 200, 3
    logits = torch.randn(B, K)
    targets = torch.randint(0, K, (B,))
    groups = torch.zeros(B, dtype=torch.long)
    groups[:5] = 1  # only 5 calibration points in group 1

    calibrator = MondrianConformalCalibrator(score_fn=SoftmaxScore(from_logits=True), alpha=0.10)
    with pytest.warns(UserWarning, match="GLOBAL threshold"):
        calibrator.calibrate(logits, targets, groups)
    assert calibrator.fallback_groups == {1: 5}
    assert calibrator.group_thresholds[1] == calibrator.default_threshold


def test_predict_no_fill_default():
    torch.manual_seed(0)
    logits = torch.randn(50, 4)
    targets = torch.randint(0, 4, (50,))
    calibrator = ConformalCalibrator(score_fn=SoftmaxScore(from_logits=True), alpha=0.10)
    calibrator.calibrate(logits[:25], targets[:25])
    # Force empty sets with a threshold below every score.
    calibrator.calibrated_alphas[0.10] = -1.0
    sets = calibrator.predict(logits[25:], alpha=0.10)
    assert (sets.sizes == 0).all()  # default leaves empty sets empty
    sets_filled = calibrator.predict(logits[25:], alpha=0.10, fill_empty=True)
    assert (sets_filled.sizes == 1).all()
