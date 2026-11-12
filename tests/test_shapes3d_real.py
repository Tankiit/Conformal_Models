"""
Unit tests for the REAL-3dshapes paired-nuisance dataset (user direction
2026-11-11: main CLeaR tables run on the official DeepMind images; the
procedural generator is the fixture). Tests run on a small synthetic GRID npz
(same schema: images + labels with the 6-factor column order) built with the
procedural renderer, so every level/pool/pairing property is checked without
the 480k HDF5.
"""

import numpy as np
import pytest
import torch

from ccm.data.shapes3d import _render_scene, PairedEvalDataset
from ccm.data.shapes3d_real import RealShapes3DPairedDataset, get_real_paired_shapes3d_loaders

# Tiny grid: floor 2 x wall 4 x obj 3 x scale 4 x shape 4 x orient 3
LEVELS = (2, 4, 3, 4, 4, 3)
N_GRID = int(np.prod(LEVELS))


@pytest.fixture(scope="module")
def grid_npz(tmp_path_factory):
    path = tmp_path_factory.mktemp("shapes3d_real") / "grid.npz"
    size = 16
    images = np.zeros((N_GRID, size, size, 3), dtype=np.uint8)
    labels = np.zeros((N_GRID, 6), dtype=np.float64)
    idx = 0
    for f in range(LEVELS[0]):
        for w in range(LEVELS[1]):
            for o in range(LEVELS[2]):
                for s in range(LEVELS[3]):
                    for sh in range(LEVELS[4]):
                        for a in range(LEVELS[5]):
                            img = _render_scene(
                                size,
                                floor_hue=f / (LEVELS[0] - 1),
                                wall_hue=w / (LEVELS[1] - 1),
                                obj_hue=o / (LEVELS[2] - 1),
                                scale=0.2 + 0.8 * s / (LEVELS[3] - 1),
                                shape=sh,
                                orientation=0.0,
                            )
                            images[idx] = np.round(
                                np.transpose(img, (1, 2, 0)) * 255
                            ).astype(np.uint8)
                            labels[idx] = [
                                f / (LEVELS[0] - 1),
                                w / (LEVELS[1] - 1),
                                o / (LEVELS[2] - 1),
                                0.2 + 0.8 * s / (LEVELS[3] - 1),
                                sh,
                                a / (LEVELS[5] - 1),
                            ]
                            idx += 1
    np.savez(path, images=images, labels=labels)
    return str(path)


def _real(npz, n=240, rho=0.0, seed=7, **kw):
    kw.setdefault("u_levels", (1, 3))  # fixture grid has 4 wall levels
    return RealShapes3DPairedDataset(npz, num_samples=n, rho=rho, seed=seed, **kw)


def test_items_are_four_tuples_with_u_out_of_concepts(grid_npz):
    ds = _real(grid_npz, rho=0.5)
    x, c, y, u = ds[0]
    assert x.shape == (3, 16, 16) and c.shape == (4,) and u.dim() == 0
    assert int(u) in (0, 1)
    assert ds.concepts.shape == (len(ds), 4)


def test_rho_counts_exact_by_construction(grid_npz):
    # fixture pool per (class, palette) = 3 obj x 3 orient x 2 scale-levels = 18
    POOL = 18
    for rho in (0.0, 0.5, 1.0):
        ds = _real(grid_npz, n=240, rho=rho)
        n_target = 240 // 8  # 30
        for yy in range(ds.num_classes):
            mask = ds.labels == yy
            n = int(mask.sum())
            n1 = int((mask & (ds.u == 1)).sum())
            p = (1 - rho) / 2 + rho * (yy % 2)
            n1_want, n0_want = round(n_target * p), n_target - round(n_target * p)
            assert n1 == min(n1_want, POOL), (rho, yy, n1)       # exact until capped
            assert n == min(n1_want, POOL) + min(n0_want, POOL)  # never topped up
        if rho == 1.0:
            assert torch.equal(ds.u, ds.labels % 2)  # survives capping


def test_composite_label_consistent_with_concepts(grid_npz):
    ds = _real(grid_npz, rho=0.5, scale_buckets=2)
    shape = torch.round(ds.concepts[:, 2] * 3).long()          # shape level
    scale_lvl = torch.round(ds.concepts[:, 1] * (LEVELS[3] - 1)).long()
    bucket = (scale_lvl * 2) // LEVELS[3]
    assert torch.equal(ds.labels, shape * 2 + bucket)


def test_flip_partner_same_c_different_wall(grid_npz):
    ds = _real(grid_npz, rho=0.5)
    half = 16 // 2
    for i in (0, 17, min(len(ds) - 1, 200)):
        x = ds[i][0]
        xf = ds.flip(i)
        # identical floor half + identical object (same C, same renderer);
        # only the wall (top) half carries the palette difference
        assert torch.equal(x[:, half:, :], xf[:, half:, :])
        assert not torch.equal(x[:, :half, :], xf[:, :half, :])
        assert not torch.equal(x, xf)
    assert len(set(ds._local.tolist())) == len(ds)  # no duplicate images


def test_pool_cap_warns_and_caps(grid_npz):
    # per (y, u) pool = 18 -> class cap 36 at rho=0.5
    ds = RealShapes3DPairedDataset(
        grid_npz, num_samples=10_000, rho=0.5, seed=0, u_levels=(1, 3)
    )
    assert len(ds) == 8 * 36
    assert len(ds) < 10_000


def test_loud_validation(grid_npz):
    with pytest.raises(FileNotFoundError, match="3dshapes.h5"):
        RealShapes3DPairedDataset("/nonexistent/3dshapes.h5")
    with pytest.raises(ValueError):
        _real(grid_npz, rho=1.5)
    with pytest.raises(ValueError):
        _real(grid_npz, rho=0.5, u_levels=(1, 1))
    with pytest.raises(ValueError):
        _real(grid_npz, rho=0.5, u_levels=(0, 99))
    with pytest.raises(ValueError):
        _real(grid_npz, rho=0.5, floor_level=5)
    with pytest.raises(ValueError):
        _real(grid_npz, rho=0.5, label_rule="mystery")


def test_loader_metadata_and_pairs(grid_npz):
    cfg = {
        "seed": 0,
        "data": {"h5_file": grid_npz, "num_samples": 200, "rho": 1.0,
                 "scale_buckets": 2, "floor_level": 0, "u_levels": [1, 3]},
        "training": {"batch_size": 32},
    }
    out = get_real_paired_shapes3d_loaders(cfg)
    assert out["num_concepts"] == 4 and out["num_classes"] == 8
    assert out["rho"] == 1.0 and out["u_source"] == "wall_hue"
    x, c, y, u = next(iter(out["train"]))
    assert x.shape[1:] == (3, 16, 16) and c.shape[1] == 4
    assert torch.equal(u, y % 2)
    n_pairs = sum(b[0].shape[0] for b in out["pairs_test"])
    n_test = sum(b[0].shape[0] for b in out["test"])
    assert n_pairs == n_test == len(out["test"].dataset)
    # PairedEvalDataset feeds paired_set_change directly (shape contract)
    for x, x_flip, c_true in out["pairs_test"]:
        assert x.shape == x_flip.shape and c_true.shape == (x.shape[0], 4)
        assert not torch.equal(x, x_flip)
        break
