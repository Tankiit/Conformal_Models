"""
Unit tests for the Shapes3D paired-nuisance generator (CLeaR leakage design,
vault spec settled 2026-11-11): legacy mode unchanged, 4-tuple items with U out
of the concept vector, the rho knob (independent at 0, parity shortcut at 1),
the composite label rule, and pair rendering that differs exactly by the
nuisance palette.
"""

import pytest
import torch

from ccm.data.shapes3d import (
    PairedEvalDataset,
    Shapes3DDataset,
    get_paired_shapes3d_loaders,
)


def _paired(n=500, size=32, rho=0.0, seed=7, **kw):
    return Shapes3DDataset(
        num_samples=n, image_size=size, seed=seed,
        u_source="wall_hue", rho=rho, **kw
    )


def test_legacy_mode_unchanged():
    ds = Shapes3DDataset(num_samples=64, image_size=32, seed=3)
    assert ds.paired is False
    x, c, y = ds[0]
    assert x.shape == (3, 32, 32) and c.shape == (6,) and y.dim() == 0
    assert torch.equal(ds.concepts[:, 4] * 3, ds.labels.float())  # label = shape col
    assert ds.labels.min() >= 0 and ds.labels.max() <= 3


def test_paired_item_is_four_tuple_with_u_out_of_concepts():
    ds = _paired()
    x, c, y, u = ds[0]
    assert x.shape == (3, 32, 32)
    assert c.shape == (4,)  # U (wall_hue) and floor_hue are NOT concepts
    assert y.dim() == 0 and u.dim() == 0
    assert int(u) in (0, 1)
    assert ds.concepts.shape == (500, 4)
    assert set(ds.u.unique().tolist()) <= {0, 1}


def test_rho_zero_leaves_u_independent_of_y():
    ds = _paired(n=4000, rho=0.0, scale_buckets=2)
    for y in ds.labels.unique().tolist():
        rate = ds.u[ds.labels == y].float().mean().item()
        assert abs(rate - 0.5) < 0.08, f"P(U=1|Y={y}) = {rate}"


def test_rho_one_is_parity_shortcut():
    ds = _paired(n=2000, rho=1.0, scale_buckets=2)
    assert torch.equal(ds.u, ds.labels % 2)


def test_composite_label_rule():
    for B in (1, 2, 3):
        ds = _paired(n=1000, rho=0.5, scale_buckets=B)
        shapes = torch.round(ds.concepts[:, 2] * 3).long()
        buckets = torch.clamp((ds.concepts[:, 1] * B).long(), max=B - 1)
        assert torch.equal(ds.labels, shapes * B + buckets)
        assert ds.labels.min() >= 0 and ds.labels.max() <= 4 * B - 1


def test_flip_differs_exactly_by_nuisance_palette():
    ds = _paired(rho=0.5)
    for i in (0, 17, 499):
        x, _, _, u = ds[i]
        xf = ds.flip(i)
        half = ds.image_size // 2
        # u_source = wall_hue: only the top half (wall) may change; floor and
        # object region are bit-identical, including the noise.
        assert torch.equal(x[:, half:, :], xf[:, half:, :])
        assert not torch.equal(x[:, :half, :], xf[:, :half, :])
        # Wall-region difference per channel is the palette delta (up to float32
        # rounding) or exactly ~0 where the object straddles the wall/floor
        # boundary; anything else would mean the noise draws diverged. Channel 1
        # renders 0.5*(1 - w), so its delta is inverted relative to the hue.
        s = 1.0 if int(u) == 1 else -1.0
        diff = x[:, :half, :] - xf[:, :half, :]
        for ch, expected, ch_sign in ((0, 0.5, 1.0), (1, 0.25, -1.0), (2, 0.4, 1.0)):
            d = diff[ch] * (s * ch_sign)
            on_wall = (d - expected).abs() < 1e-6
            on_object = d.abs() < 1e-6
            assert on_wall.logical_or(on_object).all(), ch
            assert on_wall.any(), ch


def test_render_is_deterministic_across_accesses():
    ds = _paired(rho=0.3)
    assert torch.equal(ds[123][0], ds[123][0])


def test_floor_hue_source_flips_bottom_half():
    ds = Shapes3DDataset(
        num_samples=64, image_size=32, seed=11,
        u_source="floor_hue", rho=0.5,
    )
    x, xf = ds[5][0], ds.flip(5)
    half = 32 // 2
    assert torch.equal(x[:, :half, :], xf[:, :half, :])  # wall fixed
    assert not torch.equal(x[:, half:, :], xf[:, half:, :])


def test_render_orientation_smoke():
    ds = _paired(rho=0.5, render_orientation=True)
    x = ds[0][0]
    assert x.shape == (3, 32, 32)
    xf = ds.flip(0)
    half = 32 // 2
    assert torch.equal(x[:, half:, :], xf[:, half:, :])


def test_validation_is_loud():
    with pytest.raises(ValueError):
        Shapes3DDataset(u_source="wall_hue")  # rho missing
    with pytest.raises(ValueError):
        Shapes3DDataset(rho=0.5)  # u_source missing
    with pytest.raises(ValueError):
        Shapes3DDataset(u_source="object_hue", rho=0.5)
    with pytest.raises(ValueError):
        Shapes3DDataset(u_source="wall_hue", rho=1.5)
    with pytest.raises(ValueError):
        Shapes3DDataset(u_source="wall_hue", rho=0.5, scale_buckets=0)
    with pytest.raises(ValueError):
        Shapes3DDataset(u_source="wall_hue", rho=0.5, wall_palettes=(0.5, 0.5))
    with pytest.raises(ValueError):
        Shapes3DDataset(u_source="wall_hue", rho=0.5, label_rule="mystery")
    with pytest.raises(ValueError):
        Shapes3DDataset(u_source="wall_hue", rho=0.5, cache_file="x.npz")


def test_paired_eval_dataset_feeds_paired_set_change():
    ds = _paired(n=40, rho=0.5)
    ev = PairedEvalDataset(ds, indices=list(range(0, 40, 2)))
    assert len(ev) == 20
    x, xf, c = ev[0]
    assert x.shape == xf.shape == (3, 32, 32) and c.shape == (4,)
    loader = torch.utils.data.DataLoader(ev, batch_size=8)
    for x, xf, c in loader:  # exact iteration shape paired_set_change expects
        assert x.shape == xf.shape == (8, 3, 32, 32)
        assert c.shape == (8, 4)
        break
    with pytest.raises(ValueError):
        PairedEvalDataset(Shapes3DDataset(num_samples=8, image_size=16))


def test_get_paired_loaders_metadata_and_batches():
    cfg = {
        "seed": 0,
        "data": {"num_samples": 100, "image_size": 16, "rho": 1.0, "scale_buckets": 2},
        "training": {"batch_size": 16},
    }
    out = get_paired_shapes3d_loaders(cfg)
    assert out["num_concepts"] == 4 and out["num_classes"] == 8
    assert out["rho"] == 1.0 and out["u_source"] == "wall_hue"
    assert out["concept_names"] == ["object_hue", "scale", "shape", "orientation"]
    x, c, y, u = next(iter(out["train"]))
    assert x.shape == (16, 3, 16, 16) and c.shape == (16, 4)
    assert u.shape == (16,) and torch.equal(u, y % 2)  # rho=1 survives batching
    # pairs_test covers exactly the test split
    n_pairs = sum(b[0].shape[0] for b in out["pairs_test"])
    n_test = sum(b[0].shape[0] for b in out["test"])
    assert n_pairs == n_test == 20
