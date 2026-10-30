"""
Shapes3D Dataset loader and procedural generator for Conformal Concept Models.
Features 6 generative factors: floor color, wall color, object color, scale, shape, orientation.
"""

from __future__ import annotations
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split


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
    SHAPE_NAMES = ["cube", "cylinder", "sphere", "pill"]

    def __init__(
        self,
        num_samples: int = 2000,
        image_size: int = 64,
        seed: int = 42,
        cache_file: Optional[str] = None,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.image_size = image_size
        self.seed = seed
        self.cache_file = cache_file

        self.images, self.concepts, self.labels = self._generate_or_load()

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img = self.images[idx]
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        return img, self.concepts[idx], self.labels[idx]


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
