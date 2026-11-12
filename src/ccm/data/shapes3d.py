"""
Shapes3D Dataset loader and procedural generator for Conformal Concept Models.
Features 6 generative factors: floor color, wall color, object color, scale, shape, orientation.

Paired-nuisance mode (CLeaR leakage test; vault spec "Shapes3D Paired-Nuisance
Generator Spec", settled 2026-11-11): one factor -- default binarized wall_hue --
becomes a hidden nuisance U kept OUT of the concept vector; Y = f(C) through a
composite (shape, scale-bucket) rule; a sampling-time knob rho controls
P(U=1|Y=y) = (1-rho)/2 + rho*[y mod 2] (0 = independent, 1 = U is the parity of
Y, a perfect shortcut). Marginal Y remains a function of C alone. __getitem__
returns (x, c, y, u); .flip(i) re-renders the SAME C under the same per-sample
noise seed with U flipped, so pair members differ exactly by the palette.
"""

from __future__ import annotations
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split


def _render_scene(
    image_size: int,
    floor_hue: float,
    wall_hue: float,
    obj_hue: float,
    scale: float,
    shape: int,
    orientation: float = 0.0,
) -> np.ndarray:
    """
    Pure procedural renderer -> float32 image [3, image_size, image_size], no noise.

    orientation=0.0 (default) reproduces the legacy rendering bit-for-bit
    (cos(0)=1, sin(0)=0 make the rotated coordinates identical); keep the two in
    sync if either changes.
    """
    img = np.zeros((3, image_size, image_size), dtype=np.float32)
    y_coords, x_coords = np.ogrid[:image_size, :image_size]
    center = image_size / 2.0
    s = scale * (image_size * 0.35)

    # Wall (top half) and Floor (bottom half)
    img[0, : image_size // 2, :] = wall_hue
    img[1, : image_size // 2, :] = 0.5 * (1.0 - wall_hue)
    img[2, : image_size // 2, :] = 0.8 * wall_hue

    img[0, image_size // 2 :, :] = floor_hue
    img[1, image_size // 2 :, :] = 0.7 * floor_hue
    img[2, image_size // 2 :, :] = 1.0 - floor_hue

    # Object mask, computed in a frame rotated by `orientation`
    dx = x_coords - center
    dy = y_coords - center
    cos_t, sin_t = math.cos(orientation), math.sin(orientation)
    xr = cos_t * dx + sin_t * dy
    yr = -sin_t * dx + cos_t * dy
    dist_sq = dx ** 2 + dy ** 2
    if shape == 0:  # Cube (Square)
        mask = (np.abs(xr) <= s) & (np.abs(yr) <= s)
    elif shape == 1:  # Cylinder (Tall rectangle)
        mask = (np.abs(xr) <= s * 0.7) & (np.abs(yr) <= s * 1.2)
    elif shape == 2:  # Sphere (Circle) -- rotation-invariant
        mask = dist_sq <= (s ** 2)
    else:  # Pill (Oval)
        mask = (xr ** 2) / ((s * 0.7) ** 2) + (yr ** 2) / (s ** 2) <= 1.0

    img[0][mask] = obj_hue
    img[1][mask] = 1.0 - obj_hue
    img[2][mask] = 0.5 * obj_hue
    return img


class Shapes3DDataset(Dataset):
    """
    3D Shapes dataset with ground-truth generative concept attributes.
    Supports either pre-downloaded HDF5/NPZ files or procedural on-the-fly generation.
    """

    CONCEPT_NAMES = [
        "floor_hue",
        "wall_hue",
        "object_hue",
        "scale",
        "shape",
        "orientation",
    ]
    PAIRED_CONCEPT_NAMES = ["object_hue", "scale", "shape", "orientation"]
    SHAPE_NAMES = ["cube", "cylinder", "sphere", "pill"]
    _U_SOURCES = ("wall_hue", "floor_hue")

    def __init__(
        self,
        num_samples: int = 2000,
        image_size: int = 64,
        seed: int = 42,
        cache_file: Optional[str] = None,
        u_source: Optional[str] = None,
        rho: Optional[float] = None,
        label_rule: str = "composite",
        scale_buckets: int = 2,
        wall_palettes: Sequence[float] = (0.25, 0.75),
        floor_hue: float = 0.5,
        noise_std: float = 0.02,
        render_orientation: bool = False,
    ):
        """
        Legacy mode (u_source=None, rho=None): 6 concepts, label = shape, items
        are (x, c, y) -- unchanged behavior.

        Paired-nuisance mode (u_source and/or rho given): declared concepts are
        PAIRED_CONCEPT_NAMES (U excluded), the label follows `label_rule`
        ("composite" = (shape, scale-bucket) with `scale_buckets` buckets;
        "shape" = 4-way), U = binarized `u_source` with hues `wall_palettes`,
        the OTHER background hue is pinned to `floor_hue` so X = g(C, U) exactly,
        and items are (x, c, y, u). `render_orientation=True` additionally draws
        orientation into the image (off by default: legacy/spec-exact behavior
        is that orientation is sampled but not rendered -- a pilot knob).
        """
        super().__init__()
        if (u_source is None) != (rho is None):
            raise ValueError("paired mode needs both u_source and rho (or neither)")
        self.paired = u_source is not None
        if self.paired:
            if u_source not in self._U_SOURCES:
                raise ValueError(f"u_source must be one of {self._U_SOURCES}, got {u_source!r}")
            if not 0.0 <= float(rho) <= 1.0:
                raise ValueError(f"rho must be in [0, 1], got {rho!r}")
            if label_rule not in ("composite", "shape"):
                raise ValueError(f"label_rule must be 'composite' or 'shape', got {label_rule!r}")
            if scale_buckets < 1:
                raise ValueError(f"scale_buckets must be >= 1, got {scale_buckets}")
            if len(wall_palettes) != 2 or wall_palettes[0] == wall_palettes[1]:
                raise ValueError(f"wall_palettes must be two distinct hues, got {wall_palettes!r}")
            if cache_file is not None:
                raise ValueError("cache_file is not supported in paired-nuisance mode")

        self.num_samples = num_samples
        self.image_size = image_size
        self.seed = seed
        self.cache_file = cache_file
        self.u_source = u_source
        self.rho = float(rho) if rho is not None else None
        self.label_rule = label_rule
        self.scale_buckets = scale_buckets
        self.wall_palettes = (float(wall_palettes[0]), float(wall_palettes[1]))
        self.floor_hue = float(floor_hue)
        self.noise_std = float(noise_std)
        self.render_orientation = bool(render_orientation)

        if self.paired:
            self._generate_paired()
        else:
            self.images, self.concepts, self.labels = self._generate_or_load()

    # ---------------------------------------------------------------- paired

    def _generate_paired(self) -> None:
        """Sample declared factors, derive Y = f(C), draw U ~ Bernoulli(p_y)."""
        rng = np.random.RandomState(self.seed)
        n = self.num_samples

        obj_hues = rng.uniform(0.0, 1.0, size=n)
        scales = rng.uniform(0.2, 1.0, size=n)
        shapes = rng.choice(4, size=n).astype(np.int64)
        orientations = rng.uniform(-math.pi, math.pi, size=n)

        if self.label_rule == "composite":
            buckets = np.clip(
                ((scales - 0.2) / 0.8 * self.scale_buckets).astype(np.int64),
                0, self.scale_buckets - 1,
            )
            labels = shapes * self.scale_buckets + buckets
            self.num_classes = 4 * self.scale_buckets
        else:  # "shape"
            labels = shapes
            self.num_classes = 4

        # rho knob: P(U=1|Y=y) = (1-rho)/2 + rho*[y mod 2]; correlation between
        # U and Y exists ONLY through this sampling -- marginally Y = f(C) still.
        p_u = (1.0 - self.rho) / 2.0 + self.rho * (labels % 2)
        u = (rng.uniform(size=n) < p_u).astype(np.int64)
        noise_seeds = rng.randint(0, 2 ** 31 - 1, size=n)

        self._obj_hues, self._scales, self._shapes = obj_hues, scales, shapes
        self._orientations = orientations
        self.concepts = torch.from_numpy(np.stack([
            obj_hues,
            (scales - 0.2) / 0.8,
            shapes / 3.0,
            (orientations + math.pi) / (2.0 * math.pi),
        ], axis=1).astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.u = torch.from_numpy(u)
        self.noise_seeds = noise_seeds

    def _render_paired(self, idx: int, flip: bool = False) -> torch.Tensor:
        """Render sample idx; flip=True re-renders with U flipped, same noise seed."""
        u = int(self.u[idx]) ^ 1 if flip else int(self.u[idx])
        u_hue = self.wall_palettes[u]
        fixed = self.floor_hue
        if self.u_source == "wall_hue":
            wall_hue, floor_hue = u_hue, fixed
        else:
            wall_hue, floor_hue = fixed, u_hue
        img = _render_scene(
            self.image_size,
            floor_hue=floor_hue,
            wall_hue=wall_hue,
            obj_hue=self._obj_hues[idx],
            scale=self._scales[idx],
            shape=int(self._shapes[idx]),
            orientation=self._orientations[idx] if self.render_orientation else 0.0,
        )
        noise_rng = np.random.RandomState(int(self.noise_seeds[idx]))
        img = img + noise_rng.normal(0.0, self.noise_std, size=img.shape).astype(np.float32)
        return torch.from_numpy(np.clip(img, 0.0, 1.0))

    def flip(self, idx: int) -> torch.Tensor:
        """Image of the paired sample with U flipped: identical C, identical
        noise, only the nuisance palette differs. flip(flip(i)) == self[i][0]."""
        if not self.paired:
            raise RuntimeError("flip() is only available in paired-nuisance mode")
        return self._render_paired(idx, flip=True)

    def _generate_or_load(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cache_file and os.path.exists(self.cache_file):
            data = np.load(self.cache_file)
            # Keep uint8 image arrays in memory (4x smaller); __getitem__ casts to [0,1] floats.
            images = torch.from_numpy(data["images"])
            concepts = torch.from_numpy(data["concepts"]).float()
            labels = torch.from_numpy(data["labels"]).long()
            return images, concepts, labels

        rng = np.random.RandomState(self.seed)

        # Discrete factor values
        floor_hues = rng.uniform(0.0, 1.0, size=(self.num_samples, 1))
        wall_hues = rng.uniform(0.0, 1.0, size=(self.num_samples, 1))
        obj_hues = rng.uniform(0.0, 1.0, size=(self.num_samples, 1))
        scales = rng.uniform(0.2, 1.0, size=(self.num_samples, 1))
        shapes = rng.choice(4, size=(self.num_samples, 1))  # 4 shape classes
        orientations = rng.uniform(-math.pi, math.pi, size=(self.num_samples, 1))

        # Downstream task label: Shape class (0 to 3) or composite
        labels_np = shapes[:, 0].astype(np.int64)

        # Concept array [num_samples, 6]
        concepts_np = np.hstack([
            floor_hues,
            wall_hues,
            obj_hues,
            scales,
            shapes / 3.0,  # normalized
            (orientations + math.pi) / (2 * math.pi),
        ]).astype(np.float32)

        # Procedurally generate colored images: [num_samples, 3, image_size, image_size]
        images_np = np.zeros((self.num_samples, 3, self.image_size, self.image_size), dtype=np.float32)
        
        y_coords, x_coords = np.ogrid[:self.image_size, :self.image_size]
        center = self.image_size / 2.0

        for i in range(self.num_samples):
            f_h = floor_hues[i, 0]
            w_h = wall_hues[i, 0]
            o_h = obj_hues[i, 0]
            s = scales[i, 0] * (self.image_size * 0.35)
            sh = shapes[i, 0]

            # Wall (top half) and Floor (bottom half)
            images_np[i, 0, : self.image_size // 2, :] = w_h
            images_np[i, 1, : self.image_size // 2, :] = 0.5 * (1.0 - w_h)
            images_np[i, 2, : self.image_size // 2, :] = 0.8 * w_h

            images_np[i, 0, self.image_size // 2 :, :] = f_h
            images_np[i, 1, self.image_size // 2 :, :] = 0.7 * f_h
            images_np[i, 2, self.image_size // 2 :, :] = 1.0 - f_h

            # Shape mask
            dist_sq = (x_coords - center) ** 2 + (y_coords - center) ** 2
            if sh == 0:  # Cube (Square)
                mask = (np.abs(x_coords - center) <= s) & (np.abs(y_coords - center) <= s)
            elif sh == 1:  # Cylinder (Tall rectangle)
                mask = (np.abs(x_coords - center) <= s * 0.7) & (np.abs(y_coords - center) <= s * 1.2)
            elif sh == 2:  # Sphere (Circle)
                mask = dist_sq <= (s ** 2)
            else:  # Pill (Oval)
                mask = ((x_coords - center) ** 2) / ((s * 0.7) ** 2) + ((y_coords - center) ** 2) / (s ** 2) <= 1.0

            # Render object
            images_np[i, 0][mask] = o_h
            images_np[i, 1][mask] = 1.0 - o_h
            images_np[i, 2][mask] = 0.5 * o_h

        # Add light noise for realism
        images_np += rng.normal(0.0, 0.02, size=images_np.shape).astype(np.float32)
        images_np = np.clip(images_np, 0.0, 1.0)

        images = torch.from_numpy(images_np)
        concepts = torch.from_numpy(concepts_np)
        labels = torch.from_numpy(labels_np)

        return images, concepts, labels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        if not self.paired:
            img = self.images[idx]
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            return img, self.concepts[idx], self.labels[idx]
        return self._render_paired(idx), self.concepts[idx], self.labels[idx], self.u[idx]


class PairedEvalDataset(Dataset):
    """
    Wraps a paired-mode Shapes3DDataset and yields (x, x_flip, c_true) items --
    the exact batches ccm.interventions.paired_set_change consumes. Both members
    share the declared concepts AND the per-sample noise seed; they differ only
    in the nuisance palette.
    """

    def __init__(self, base: Shapes3DDataset, indices: Optional[Sequence[int]] = None):
        if not getattr(base, "paired", False):
            raise ValueError("PairedEvalDataset needs a paired-nuisance Shapes3DDataset")
        self.base = base
        self.indices = None if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.base) if self.indices is None else len(self.indices)

    def __getitem__(self, idx: int):
        j = idx if self.indices is None else self.indices[idx]
        x, c, _, _ = self.base[j]
        return x, self.base.flip(j), c


def get_paired_shapes3d_loaders(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train/cal/test loaders for the paired-nuisance Shapes3D experiment
    (same-distribution splits at the SAME rho, per D2: main tables at matched
    rho; build a second dataset with a different rho for the stress eval).
    Also returns `pairs_test`: a loader of (x, x_flip, c_true) over the test
    split for ccm.interventions.paired_set_change.
    """
    data_cfg = config.get("data", {})
    dataset = Shapes3DDataset(
        num_samples=data_cfg.get("num_samples", 2000),
        image_size=data_cfg.get("image_size", 64),
        seed=config.get("seed", 42),
        u_source=data_cfg.get("u_source", "wall_hue"),
        rho=data_cfg.get("rho", 0.0),
        label_rule=data_cfg.get("label_rule", "composite"),
        scale_buckets=data_cfg.get("scale_buckets", 2),
        wall_palettes=tuple(data_cfg.get("wall_palettes", (0.25, 0.75))),
        floor_hue=data_cfg.get("floor_hue", 0.5),
        noise_std=data_cfg.get("noise_std", 0.02),
        render_orientation=data_cfg.get("render_orientation", False),
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
        "u_source": dataset.u_source,
        "rho": dataset.rho,
        "label_rule": dataset.label_rule,
        "scale_buckets": dataset.scale_buckets,
    }


def get_shapes3d_loaders(
    config: Dict[str, Any],
) -> Dict[str, DataLoader]:
    """
    Constructs train, validation, calibration, and test dataloaders for Shapes3D.
    """
    data_cfg = config.get("data", {})
    num_samples = data_cfg.get("num_samples", 2000)
    image_size = data_cfg.get("image_size", 64)
    batch_size = config.get("training", {}).get("batch_size", 64)
    seed = config.get("seed", 42)

    dataset = Shapes3DDataset(
        num_samples=num_samples,
        image_size=image_size,
        seed=seed,
        cache_file=data_cfg.get("cache_file"),
    )

    n_total = len(dataset)
    n_train = int(0.6 * n_total)
    n_calib = int(0.2 * n_total)
    n_test = n_total - n_train - n_calib

    generator = torch.Generator().manual_seed(seed)
    train_ds, calib_ds, test_ds = random_split(
        dataset, [n_train, n_calib, n_test], generator=generator
    )

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "calib": DataLoader(calib_ds, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        "num_concepts": 6,
        "num_classes": 4,
        "concept_names": Shapes3DDataset.CONCEPT_NAMES,
    }
