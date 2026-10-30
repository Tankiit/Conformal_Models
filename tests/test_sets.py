"""
Unit tests for PredictionSet, ConceptPredictionSet, and set propagation operations.
"""

import pytest
import torch
import torch.nn as nn

from ccm.sets import PredictionSet, ConceptPredictionSet, propagate_concept_sets_to_task


def test_prediction_set_properties():
    # 4 samples, 3 classes
    mask = torch.tensor([
        [True, False, False],   # size 1 (singleton)
        [True, True, False],    # size 2
        [False, False, False],  # size 0 (empty)
        [True, True, True],     # size 3
    ])
    ps = PredictionSet(mask)

    assert len(ps) == 4
    assert torch.equal(ps.sizes, torch.tensor([1, 2, 0, 3]))
    assert ps.avg_size == pytest.approx(1.5, abs=1e-5)
    assert torch.equal(ps.empty_mask, torch.tensor([False, False, True, False]))
    assert torch.equal(ps.singleton_mask, torch.tensor([True, False, False, False]))

    labels = ps.to_label_list()
    assert labels == [[0], [0, 1], [], [0, 1, 2]]

    targets = torch.tensor([0, 1, 2, 1])
    # sample 0: true in {0} -> True
    # sample 1: true in {0, 1} -> True
    # sample 2: 2 in {} -> False
    # sample 3: 1 in {0, 1, 2} -> True
    assert torch.equal(ps.contains(targets), torch.tensor([True, True, False, True]))
    assert ps.coverage(targets) == pytest.approx(0.75, abs=1e-5)


def test_prediction_set_fill_empty():
    mask = torch.tensor([
        [False, False, False],
        [True, False, False],
    ])
    logits = torch.tensor([
        [0.1, 0.8, 0.2],  # top1 is index 1
        [0.9, 0.1, 0.0],
    ])
    ps = PredictionSet(mask).fill_empty_with_top1(logits)
    assert not ps.empty_mask.any()
    assert ps.to_label_list()[0] == [1]
    assert ps.to_label_list()[1] == [0]


def test_concept_prediction_set_binary():
    # 2 samples, 3 concepts, 2 binary outcomes (0 or 1)
    mask = torch.tensor([
        [[True, False], [True, True], [False, True]],  # c0={0}, c1={0,1}, c2={1}
        [[False, True], [True, False], [True, False]], # c0={1}, c1={0}, c2={0}
    ])
    cps = ConceptPredictionSet(binary_mask=mask)

    assert len(cps) == 2
    assert cps.num_concepts == 3

    # targets: sample 0 has [0, 1, 1], sample 1 has [1, 0, 1]
    targets = torch.tensor([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],  # for sample 1, c2 is 1 but mask is {0} -> not contained
    ])

    contains = cps.contains(targets)
    assert contains[0].all()  # All covered for sample 0
    assert not contains[1, 2] # Sample 1 c2 not covered
    assert cps.joint_coverage(targets) == 0.5
    assert cps.marginal_coverage(targets) == pytest.approx(5 / 6, abs=1e-5)


def test_concept_prediction_set_continuous():
    lower = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    upper = torch.tensor([[1.0, 2.0], [4.0, 5.0]])
    cps = ConceptPredictionSet(intervals=(lower, upper))

    targets = torch.tensor([[0.5, 1.5], [2.5, 6.0]])
    contains = cps.contains(targets)
    assert contains[0, 0] and contains[0, 1]
    assert contains[1, 0] and not contains[1, 1]


def test_propagate_concept_sets():
    torch.manual_seed(42)
    # Simple linear model c -> y (3 concepts -> 2 classes)
    model = nn.Linear(3, 2)
    mask = torch.tensor([
        [[True, True], [True, False], [False, True]],
    ])
    cps = ConceptPredictionSet(binary_mask=mask)

    task_sets = propagate_concept_sets_to_task(model, cps)
    assert isinstance(task_sets, PredictionSet)
    assert len(task_sets) == 1
    assert task_sets.sizes[0] >= 1
