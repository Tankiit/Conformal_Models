"""
Caltech-UCSD Birds-200-2011 (CUB) Dataset loader and benchmark simulator for Conformal Concept Models.
Handles binary bird attributes (wing color, bill shape, breast pattern, etc.) and bird species classes.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


class CUBDataset(Dataset):
    """
    CUB-200-2011 Dataset loader with attribute bottleneck annotations.
    Supports real CUB folder directory or synthetic proxy benchmark generation.
    """

    # Sample standard attribute names from CUB-200-2011
    CANONICAL_ATTRIBUTES = [
        "has_bill_shape_allpurpose", "has_bill_shape_cone", "has_bill_shape_curved",
        "has_bill_shape_dagger", "has_bill_shape_hooked", "has_wing_color_black",
        "has_wing_color_brown", "has_wing_color_white", "has_wing_color_yellow",
        "has_breast_color_black", "has_breast_color_brown", "has_breast_color_white",
        "has_breast_color_yellow", "has_throat_color_black", "has_throat_color_white",
        "has_eye_color_black", "has_eye_color_brown", "has_belly_color_white",
        "has_belly_color_yellow", "has_crown_color_black", "has_crown_color_red",
        "has_nape_color_black", "has_nape_color_blue", "has_forehead_color_black",
        "has_underparts_color_white", "has_size_small", "has_size_medium", "has_size_large"
    ]

    def __init__(
        self,
        root: Optional[str] = None,
        num_classes: int = 10,
        num_concepts: int = 28,
        num_samples: int = 1500,
        image_size: int = 64,
        seed: int = 42,
    ):
        super().__init__()
        self.root = root
        self.num_classes = num_classes
        self.num_concepts = num_concepts
        self.num_samples = num_samples
        self.image_size = image_size
        self.seed = seed

        self.images, self.concepts, self.labels = self._load_or_simulate()

    def _load_or_simulate(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # If root directory exists with CUB files
        if self.root and os.path.isdir(self.root) and os.path.exists(os.path.join(self.root, "images.txt")):
            # Real CUB dataset parsing logic
            return self._parse_real_cub()

        # Deterministic simulation of CUB structure
        rng = np.random.RandomState(self.seed)

        # Class prototypes in concept space: each class has a signature set of attributes
        class_concept_prototypes = rng.binomial(1, 0.35, size=(self.num_classes, self.num_concepts)).astype(np.float32)

        labels_list = []
        concepts_list = []
        images_list = []

        samples_per_class = self.num_samples // self.num_classes
        for class_id in range(self.num_classes):
            proto = class_concept_prototypes[class_id]
            for _ in range(samples_per_class):
                # Add observation noise: 8% attribute flip probability
                noise_flip = rng.binomial(1, 0.08, size=(self.num_concepts,))
                sample_concept = np.abs(proto - noise_flip).astype(np.float32)

                # Generate image embedding / synthetic feature pattern
                img = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)
                # Primary color channel determined by class
                primary_channel = class_id % 3
                img[primary_channel] += 0.4
                # Texture / spatial signal driven by active concepts
                active_idx = np.where(sample_concept > 0.5)[0]
                for idx in active_idx[:5]:
                    pos_x = (idx * 7) % (self.image_size - 10)
                    pos_y = (idx * 11) % (self.image_size - 10)
                    img[:, pos_y : pos_y + 10, pos_x : pos_x + 10] += 0.3

                # Normalization
                img += rng.normal(0.0, 0.05, size=img.shape).astype(np.float32)
                img = np.clip(img, 0.0, 1.0)

                labels_list.append(class_id)
                concepts_list.append(sample_concept)
                images_list.append(img)

        images = torch.from_numpy(np.array(images_list, dtype=np.float32))
        concepts = torch.from_numpy(np.array(concepts_list, dtype=np.float32))
        labels = torch.from_numpy(np.array(labels_list, dtype=np.int64))

        return images, concepts, labels

    def _parse_real_cub(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parse a real CUB-200-2011 directory: bird crops from bounding boxes,
        class labels, and binary attribute concepts for the canonical attributes."""
        from collections import defaultdict

        from PIL import Image

        # Attribute id -> normalized name ("has_bill_shape::curved_(up_or_down)" -> "has_bill_shape_curved")
        name_to_id: Dict[int, str] = {}
        with open(os.path.join(self.root, "attributes.txt"), encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    norm = (
                        parts[1].strip()
                        .lower()
                        .replace("::", "_")
                        .replace("-", "")
                        .split("_(")[0]
                    )
                    name_to_id[norm] = int(parts[0])

        try:
            attr_ids = [name_to_id[n] for n in self.CANONICAL_ATTRIBUTES[: self.num_concepts]]
        except KeyError as e:
            raise ValueError(f"CUB attribute {e} missing from attributes.txt") from e

        img_path: Dict[int, str] = {}
        with open(os.path.join(self.root, "images.txt"), encoding="utf-8") as f:
            for line in f:
                idx, path = line.strip().split()
                img_path[int(idx)] = path

        img_label: Dict[int, int] = {}
        with open(os.path.join(self.root, "image_class_labels.txt"), encoding="utf-8") as f:
            for line in f:
                idx, cls = line.strip().split()
                img_label[int(idx)] = int(cls)

        img_box: Dict[int, Tuple[float, float, float, float]] = {}
        with open(os.path.join(self.root, "bounding_boxes.txt"), encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                img_box[int(parts[0])] = tuple(float(v) for v in parts[1:5])

        by_class: Dict[int, List[int]] = defaultdict(list)
        for idx in sorted(img_path):
            by_class[img_label[idx]].append(idx)

        selected_classes = sorted(by_class)[: self.num_classes]
        per_class = max(1, self.num_samples // self.num_classes)
        rng = np.random.RandomState(self.seed)
        chosen: List[int] = []
        for cls in selected_classes:
            pool = by_class[cls]
            if len(pool) <= per_class:
                chosen.extend(pool)
            else:
                chosen.extend(rng.choice(pool, size=per_class, replace=False).tolist())

        row_of = {img_id: k for k, img_id in enumerate(chosen)}
        col_of = {a: k for k, a in enumerate(attr_ids)}

        concepts = np.zeros((len(chosen), len(attr_ids)), dtype=np.float32)
        with open(
            os.path.join(self.root, "attributes", "image_attribute_labels.txt"), encoding="utf-8"
        ) as f:
            for line in tqdm(f, desc="Parsing CUB attribute labels", miniters=250_000):
                parts = line.split()
                img_id, attr_id, present = int(parts[0]), int(parts[1]), int(parts[2])
                k = row_of.get(img_id)
                col = col_of.get(attr_id)
                if k is not None and col is not None:
                    concepts[k, col] = float(present)

        images = np.zeros((len(chosen), 3, self.image_size, self.image_size), dtype=np.float32)
        for k, img_id in enumerate(tqdm(chosen, desc="Loading CUB images")):
            x, y, w, h = img_box[img_id]
            with Image.open(os.path.join(self.root, "images", img_path[img_id])) as im:
                crop = im.convert("RGB").crop((x, y, x + w, y + h)).resize(
                    (self.image_size, self.image_size), Image.Resampling.BILINEAR
                )
            images[k] = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1) / 255.0

        class_to_idx = {cls: k for k, cls in enumerate(selected_classes)}
        labels = np.array([class_to_idx[img_label[i]] for i in chosen], dtype=np.int64)

        return torch.from_numpy(images), torch.from_numpy(concepts), torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.images[idx], self.concepts[idx], self.labels[idx]


def get_cub_loaders(
    config: Dict[str, Any],
) -> Dict[str, DataLoader]:
    """
    Constructs train, calibration, and test dataloaders for CUB-200-2011.
    """
    data_cfg = config.get("data", {})
    num_samples = data_cfg.get("num_samples", 1600)
    num_classes = data_cfg.get("num_classes", 10)
    num_concepts = data_cfg.get("num_concepts", 28)
    image_size = data_cfg.get("image_size", 64)
    batch_size = config.get("training", {}).get("batch_size", 32)
    seed = config.get("seed", 42)
    data_root = data_cfg.get("data_dir", None)

    dataset = CUBDataset(
        root=data_root,
        num_classes=num_classes,
        num_concepts=num_concepts,
        num_samples=num_samples,
        image_size=image_size,
        seed=seed,
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
        "num_concepts": num_concepts,
        "num_classes": num_classes,
        "concept_names": CUBDataset.CANONICAL_ATTRIBUTES[:num_concepts],
    }
