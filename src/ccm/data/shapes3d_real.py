"""
REAL 3dshapes paired-nuisance dataset (CLeaR leakage test on the official
DeepMind 3dshapes images -- user direction 2026-11-11: main tables run on real
data; the procedural generator stays as the unit-test fixture).

Official grid: 480,000 images; label columns in order
[floor_hue(10), wall_hue(10), object_hue(10), scale(8), shape(4), orientation(15)]
-- every factor combination exists exactly once, which is what makes EXACT
U-flip pairs possible: for any declared-concept combo C there is a real image
for each wall hue.

The settled design (vault spec, D1/D2/D5, 2026-11-11) mapped onto the grid:
- floor_hue restricted to ONE level (default 0) so X varies only with (C, U);
- U = wall_hue at one of two palette levels (default levels 1 and 8), NOT a
  concept;
- C = {object_hue, scale, shape, orientation} as level-fraction concepts;
- Y = f(C) composite (shape, scale-bucket) -- same rule as the procedural
  generator;
- rho knob by SUBSAMPLING: per class y, draw round(n_y * p_y) real images from
  the U=1 pool and the rest from the U=0 pool, without replacement,
  p_y = (1-rho)/2 + rho*[y mod 2]. Marginally Y = f(C) always; the U-Y
  correlation exists only through this sampling.
- flip(i) returns the real partner image (same C, other wall palette); at
  rho=1 the partner is outside the sampled set by construction -- exactly the
  counterfactual the deployed model is evaluated on.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from ccm.data.shapes3d import PairedEvalDataset, Shapes3DDataset

DOWNLOAD_URL = "https://storage.googleapis.com/3d-shapes/3dshapes.h5"


class RealShapes3DPairedDataset(Dataset):
    """
    Same item interface as the procedural paired Shapes3DDataset:
    __getitem__ -> (x, c, y, u); flip(i) -> partner image; .paired = True.
    """

    PAIRED_CONCEPT_NAMES = Shapes3DDataset.PAIRED_CONCEPT_NAMES  # same 4 declared concepts

    def __init__(
        self,
        data_file: str,
        num_samples: int = 4800,
        rho: float = 0.0,
        seed: int = 42,
        label_rule: str = "composite",
        scale_buckets: int = 2,
        floor_level: int = 0,
        u_levels: Sequence[int] = (1, 8),
    ):
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"Real 3dshapes file not found: {data_file!r}. "
                f"Download the official HDF5 ({DOWNLOAD_URL}) first."
            )
        if not 0.0 <= float(rho) <= 1.0:
            raise ValueError(f"rho must be in [0, 1], got {rho!r}")
        if label_rule not in ("composite", "shape"):
            raise ValueError(f"label_rule must be 'composite' or 'shape', got {label_rule!r}")
        if scale_buckets < 1:
            raise ValueError(f"scale_buckets must be >= 1, got {scale_buckets}")
        if len(set(u_levels)) != 2:
            raise ValueError(f"u_levels must be two distinct wall levels, got {u_levels!r}")

        self.data_file = data_file
        self.num_samples = num_samples
        self.rho = float(rho)
        self.seed = seed
        self.label_rule = label_rule
        self.scale_buckets = scale_buckets
        self.floor_level = int(floor_level)
        self.u_levels = (int(u_levels[0]), int(u_levels[1]))
        self.paired = True
        self.u_source = "wall_hue"
        self.image_size = None  # set after loading

        images_handle, labels = self._load_labels(data_file)
        try:
            self._build(images_handle, labels)
        finally:
            if hasattr(images_handle, "close"):
                images_handle.close()

    # ------------------------------------------------------------------ io

    @staticmethod
    def _load_labels(data_file: str) -> Tuple[Any, np.ndarray]:
        """Returns (images source, labels array (N, 6)). Accepts the official
        HDF5 (keys 'images'/'labels') or an NPZ with the same keys (fixtures)."""
        if data_file.endswith(".npz"):
            npz = np.load(data_file)
            return npz, npz["labels"]
        import h5py

        f = h5py.File(data_file, "r")
        return f, f["labels"][:]

    def _read_images(self, images_source: Any, indices: np.ndarray) -> np.ndarray:
        """Read uint8 images at `indices` (HDF5: one sorted fancy read)."""
        order = sorted(indices.tolist())
        if isinstance(images_source, np.lib.npyio.NpzFile):
            imgs = images_source["images"][order]
        else:  # h5py File handle
            imgs = images_source["images"][order]
        # (n, H, W, 3) uint8 -> (n, 3, H, W)
        imgs = np.transpose(imgs, (0, 3, 1, 2))
        self.image_size = int(imgs.shape[2])
        return imgs

    # -------------------------------------------------------------- build

    def _build(self, images_source: Any, labels: np.ndarray) -> None:
        rng = np.random.RandomState(self.seed)

        # Factor levels derived from the data itself (works for the official
        # file and for small grid fixtures alike).
        n_levels = [len(np.unique(labels[:, k])) for k in range(6)]
        lvl = np.stack(
            [np.searchsorted(np.unique(labels[:, k]), labels[:, k]) for k in range(6)],
            axis=1,
        )  # (N, 6) integer level indices

        if not (0 <= self.floor_level < n_levels[0]):
            raise ValueError(f"floor_level must be in [0, {n_levels[0]}), got {self.floor_level}")
        for ul in self.u_levels:
            if not (0 <= ul < n_levels[1]):
                raise ValueError(f"u_levels entries must be in [0, {n_levels[1]}), got {self.u_levels}")

        # Restrict to one floor level; candidates = both wall palettes.
        floor_mask = lvl[:, 0] == self.floor_level
        wall_mask = np.isin(lvl[:, 1], self.u_levels)
        cand = np.where(floor_mask & wall_mask)[0]
        u_of_cand = (lvl[cand, 1] == self.u_levels[1]).astype(np.int64)

        # Declared-concept combo of each candidate: (obj, scale, shape, orient)
        combos = lvl[cand][:, [2, 3, 4, 5]]
        n_scale_lvls = n_levels[3]
        if self.label_rule == "composite":
            buckets = np.clip(
                (combos[:, 1] * self.scale_buckets) // n_scale_lvls, 0, self.scale_buckets - 1
            )
            y_of_cand = combos[:, 2] * self.scale_buckets + buckets
            self.num_classes = 4 * self.scale_buckets
        else:
            y_of_cand = combos[:, 2]
            self.num_classes = 4

        # Per-class sampling without replacement: exactly round(n_y * p_y)
        # images from the U=1 pool and the rest from the U=0 pool (the rho
        # knob). If a pool is too small the CLASS SHRINKS -- it is never
        # topped up from the other pool, which would break the parity
        # semantics (at rho=1 an odd class must contain U=1 images only).
        p_of_y = (1.0 - self.rho) / 2.0 + self.rho * (np.arange(self.num_classes) % 2)
        sel = []
        for y in range(self.num_classes):
            in_y = y_of_cand == y
            pool_u1 = cand[in_y & (u_of_cand == 1)]
            pool_u0 = cand[in_y & (u_of_cand == 0)]
            n_target = num_split(self.num_samples, self.num_classes, y)
            n1_want = int(round(n_target * p_of_y[y]))
            n0_want = n_target - n1_want
            n1 = min(n1_want, len(pool_u1))
            n0 = min(n0_want, len(pool_u0))
            if n1 + n0 < n_target:
                print(
                    f"[RealShapes3DPairedDataset] WARNING: class {y} capped at "
                    f"{n1 + n0} samples (wanted {n_target}; pools u0={len(pool_u0)}, "
                    f"u1={len(pool_u1)}); lower num_samples or widen floor/u levels."
                )
            take_u1 = rng.choice(pool_u1, size=n1, replace=False) if n1 else pool_u1[:0]
            take_u0 = rng.choice(pool_u0, size=n0, replace=False) if n0 else pool_u0[:0]
            sel.append(np.concatenate([take_u1, take_u0]))
        sel = rng.permutation(np.concatenate(sel))

        sel_h5 = sel
        sel_u = (lvl[sel_h5, 1] == self.u_levels[1]).astype(np.int64)
        sel_combo = lvl[sel_h5][:, [2, 3, 4, 5]]
        # Labels recomputed from the sampled combos: identical rule to the
        # pool construction, no index bookkeeping to get wrong.
        if self.label_rule == "composite":
            sel_y = sel_combo[:, 2] * self.scale_buckets + np.clip(
                (sel_combo[:, 1] * self.scale_buckets) // n_scale_lvls,
                0, self.scale_buckets - 1,
            )
        else:
            sel_y = sel_combo[:, 2]

        # Partner lookup: same combo, other palette -- exists for every combo
        # in the official full grid. Map combo tuple -> {u: h5 index}.
        partner_of = {}
        for h5_idx, combo, u in zip(cand, combos, u_of_cand):
            partner_of.setdefault(tuple(combo), {})[int(u)] = int(h5_idx)
        partner_h5 = np.array(
            [partner_of[tuple(c)][1 - int(uu)] for c, uu in zip(sel_combo, sel_u)], dtype=np.int64
        )

        # Preload selected + partner images (single sorted read from the HDF5).
        all_h5 = np.concatenate([sel_h5, partner_h5])
        uniq_h5, inv = np.unique(all_h5, return_inverse=True)
        n = len(sel_h5)
        images = self._read_images(images_source, uniq_h5)
        self._images = torch.from_numpy(np.ascontiguousarray(images))
        self._local = inv[:n]
        self._partner_local = inv[n:]

        self.concepts = torch.from_numpy(
            np.stack(
                [
                    sel_combo[:, 0] / max(n_levels[2] - 1, 1),
                    sel_combo[:, 1] / max(n_scale_lvls - 1, 1),
                    sel_combo[:, 2] / 3.0,
                    sel_combo[:, 3] / max(n_levels[5] - 1, 1),
                ],
                axis=1,
            ).astype(np.float32)
        )
        self.labels = torch.from_numpy(sel_y.astype(np.int64))
        self.u = torch.from_numpy(sel_u)

    # ------------------------------------------------------------- access

    def __len__(self) -> int:
        return len(self.labels)

    def _image(self, local_idx: np.ndarray) -> torch.Tensor:
        return self._images[local_idx].float() / 255.0

    def __getitem__(self, idx: int):
        return (
            self._image(self._local[idx]),
            self.concepts[idx],
            self.labels[idx],
            self.u[idx],
        )

    def flip(self, idx: int) -> torch.Tensor:
        """Real partner image: identical declared C, opposite wall palette."""
        return self._image(self._partner_local[idx])


def num_split(total: int, parts: int, i: int) -> int:
    """Split `total` into `parts` near-equal shares; share i."""
    return total // parts + (1 if i < total % parts else 0)


def get_real_paired_shapes3d_loaders(config: Dict[str, Any]) -> Dict[str, Any]:
    """Same-distribution train/cal/test at the SAME rho (D2: matched-rho main
    tables; build a second dataset at another rho for the stress eval), plus
    `pairs_test` feeding ccm.interventions.paired_set_change."""
    data_cfg = config.get("data", {})
    dataset = RealShapes3DPairedDataset(
        data_file=data_cfg.get("h5_file", "/Users/tanmoy/research/data/3dshapes.h5"),
        num_samples=data_cfg.get("num_samples", 4800),
        rho=data_cfg.get("rho", 0.0),
        seed=config.get("seed", 42),
        label_rule=data_cfg.get("label_rule", "composite"),
        scale_buckets=data_cfg.get("scale_buckets", 2),
        floor_level=data_cfg.get("floor_level", 0),
        u_levels=tuple(data_cfg.get("u_levels", (1, 8))),
    )

    n_total = len(dataset)
    n_train = int(0.6 * n_total)
    n_calib = int(0.2 * n_total)
    n_test = n_total - n_train - n_calib

    seed = config.get("seed", 42)
    batch_size = config.get("training", {}).get("batch_size", 64)
    generator = torch.Generator().manual_seed(seed)
    train_ds, calib_ds, test_ds = random_split(
        dataset, [n_train, n_calib, n_test], generator=generator
    )

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "calib": DataLoader(calib_ds, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        "pairs_test": DataLoader(
            PairedEvalDataset(dataset, indices=test_ds.indices),
            batch_size=batch_size,
            shuffle=False,
        ),
        "num_concepts": len(Shapes3DDataset.PAIRED_CONCEPT_NAMES),
        "num_classes": dataset.num_classes,
        "concept_names": Shapes3DDataset.PAIRED_CONCEPT_NAMES,
        "u_source": "wall_hue",
        "rho": dataset.rho,
        "label_rule": dataset.label_rule,
        "scale_buckets": dataset.scale_buckets,
        "data_file": dataset.data_file,
        "n_samples": n_total,
    }
