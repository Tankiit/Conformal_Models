"""
Unit tests for the paired U-flip set-change evaluation (CLeaR Shapes3D design):
Test 1 (ground-truth concepts fixed -> exactly zero set-change for a strict label
head), Test 2 (predicted concepts -> the concept head can move sets), and the
shared-u property (identical inputs + fixed u -> identical sets).
"""

import torch

from ccm.conformal import ConformalCalibrator
from ccm.interventions import paired_set_change
from ccm.scores import APSScore


class MockCBM(torch.nn.Module):
    """Concept j reads ONLY input dim j (weight 4, bias 0); linear label head."""

    def __init__(self, D=8, K=4, C=3, seed=0):
        super().__init__()
        self.x_to_c = torch.nn.Linear(D, K)
        with torch.no_grad():
            self.x_to_c.weight.zero_()
            self.x_to_c.weight[torch.arange(K), torch.arange(K)] = 4.0
            self.x_to_c.bias.zero_()
        self.c_to_y = torch.nn.Linear(K, C)
        g_state = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.c_to_y.weight.copy_(torch.randn(C, K, generator=g_state) * 2.0)
            self.c_to_y.bias.copy_(torch.randn(C, generator=g_state) * 0.5)

    def forward(self, x, intervened_concepts=None, intervention_mask=None):
        c_logits = self.x_to_c(x)
        c = torch.sigmoid(c_logits)
        if intervention_mask is not None and intervened_concepts is not None:
            c = torch.where(intervention_mask, intervened_concepts.float(), c)
        return self.c_to_y(c), c_logits


def _calibrated(n=200, D=8, C=3, seed=0):
    torch.manual_seed(seed)
    model = MockCBM()
    X = torch.randn(n, D)
    logits, _ = model(X)
    y = torch.randint(0, C, (n,))
    calibrator = ConformalCalibrator(
        score_fn=APSScore(from_logits=True, randomized=True), alpha=0.10)
    calibrator.calibrate(logits, y, alpha=0.10)
    return model, calibrator


def test_gt_concepts_zero_set_change_for_strict_label_head():
    model, cal = _calibrated()
    torch.manual_seed(1)
    x = torch.randn(128, 8)
    x_flip = x.clone()
    x_flip[:, :4] += 5.0                 # big nuisance-flip look-alike
    c_true = (torch.sigmoid(model.x_to_c(x)) > 0.5).float()
    res = paired_set_change(model, cal, [(x, x_flip, c_true)],
                            use_gt_concepts=True, seed=0)
    assert res["set_change_rate"] == 0.0  # label head is f(c): identical by construction
    assert res["mean_prob_shift_l1"] == 0.0


def test_predicted_concepts_can_change_sets():
    model, cal = _calibrated()
    torch.manual_seed(2)
    x = torch.randn(256, 8)
    x_flip = x.clone()
    x_flip[:, 0] += 6.0                  # moves concept 0 from mixed to ~1
    c_true = (torch.sigmoid(model.x_to_c(x)) > 0.5).float()
    res = paired_set_change(model, cal, [(x, x_flip, c_true)],
                            use_gt_concepts=False, seed=0)
    assert 0.0 < res["set_change_rate"] <= 1.0
    assert res["mean_prob_shift_l1"] > 0.0


def test_shared_u_identical_inputs_give_identical_sets():
    model, cal = _calibrated()
    torch.manual_seed(3)
    x = torch.randn(128, 8)
    c_true = (torch.sigmoid(model.x_to_c(x)) > 0.5).float()
    res = paired_set_change(model, cal, [(x, x, c_true)],
                            use_gt_concepts=False, seed=0)
    # u-pairing null is exactly zero: same logits + same u -> same randomized scores
    assert res["set_change_rate"] == 0.0
    assert res["mean_prob_shift_l1"] == 0.0


def test_batches_accumulate_and_report_n_pairs():
    model, cal = _calibrated()
    torch.manual_seed(4)
    x = torch.randn(64, 8)
    c_true = (torch.sigmoid(model.x_to_c(x)) > 0.5).float()
    res = paired_set_change(model, cal, [(x, x, c_true), (x, x, c_true)],
                            use_gt_concepts=False, seed=0)
    assert res["n_pairs"] == 128
    assert len(res["changed"]) == 128 and len(res["prob_shifts"]) == 128


class _LogitsAsInputs(torch.nn.Module):
    """Model whose task logits ARE the input batch (hand-checkable readouts)."""

    def __init__(self, num_classes=3):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.num_classes = num_classes

    def forward(self, x, intervened_concepts=None, intervention_mask=None):
        return x, None


def test_coverage_cost_readouts():
    from ccm.conformal import ConformalCalibrator
    from ccm.scores import SoftmaxScore

    torch.manual_seed(5)
    K = 3
    model = _LogitsAsInputs(K)
    # calibration: near-singleton sets at a lenient threshold
    cal_logits = torch.eye(K)[torch.randint(0, K, (300,))] * 6.0 + 0.1 * torch.randn(300, K)
    cal_y = cal_logits.argmax(dim=-1)
    cal = ConformalCalibrator(score_fn=SoftmaxScore(from_logits=True), alpha=0.10)
    cal.calibrate(cal_logits, cal_y)

    # hand-built pairs: y=0 in both; y=1 covered in x only, lost to class 2 in flip
    def logits_for(winner, K=3):
        z = torch.full((1, K), -6.0)
        z[0, winner] = 6.0
        return z

    x = torch.cat([logits_for(0), logits_for(1)])
    x_flip = torch.cat([logits_for(0), logits_for(2)])
    c = torch.zeros(2, 4)
    y = torch.tensor([0, 1])
    res = paired_set_change(model, cal, [(x, x_flip, c, y)],
                            use_gt_concepts=False, seed=0,
                            substitution_labels=torch.tensor([2, 2]))
    assert res["contains_x"].tolist() == [True, True]
    assert res["contains_x_flip"].tolist() == [True, False]
    assert res["coverage_x"] == 1.0 and res["coverage_x_flip"] == 0.5
    assert res["coverage_both"] == 0.5
    assert res["directional_loss"] == 0.5
    assert res["directional_gain"] == 0.0
    # the lost pair admits the substitute (class 2 = y^2) in the flipped set
    assert res["substitution_rate_given_lost"] == 1.0
