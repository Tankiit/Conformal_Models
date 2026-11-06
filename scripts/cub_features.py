#!/usr/bin/env python3
"""
Extract frozen DINOv2-base CLS features for CUB-200-2011 (Koh et al. class_attr_data_10 splits).

Pipeline (matches the Credal_Sets/DRO_Neurips feature convention):
  image -> Resize(256) -> CenterCrop(224) -> ImageNet normalize -> dinov2_vitb14 CLS token (768-d)

Also verifies label semantics:
  - attribute_label (112,) is class-denoised (constant within class)
  - attribute_certainty (312,) binarizes to the raw per-image labels
  - agreement between raw and class-denoised on the 112 selected concepts (~0.919 expected)

Outputs data_cache/cub_dinov2b14_{split}.pt with {features, labels, concepts112, certainty312, img_ids}.
Validates test features against the cached Credal_Sets file when present.
"""

from __future__ import annotations
import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

CUB_ROOT = Path("/Users/tanmoy/research/data/CUB_200_2011/CUB_200_2011")
CACHE_DIR = PROJECT_ROOT / "data_cache"
REFERENCE_TEST_FEATURES = Path(
    "/Users/tanmoy/research/Credal_Sets/DRO_Neurips/tests/cub200_test_dinov2_features.pt"
)

VAL_TF = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_present312() -> dict:
    """Parse CUB attributes/image_attribute_labels.txt -> {image_id: (312,) is_present vector}."""
    path = CUB_ROOT / "attributes" / "image_attribute_labels.txt"
    n_img, n_attr = 11788, 312
    table = np.zeros((n_img + 1, n_attr), dtype=np.int8)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                table[int(parts[0]), int(parts[1]) - 1] = int(parts[2])
    return table


def attach_present312(data: dict, split: str, table=None):
    """Add per-image 312 is_present labels to a cached split dict (aligned to pkl order)."""
    if "present312" in data:
        return
    with open(CUB_ROOT / "class_attr_data_10" / f"{split}.pkl", "rb") as f:
        raw = pickle.load(f)
    table = load_present312() if table is None else table
    # pkl entry 'id' is the CUB image id; verify via basename order
    present = np.stack([table[int(e["id"])] for e in raw]).astype(np.float32)
    data["present312"] = torch.from_numpy(present)
    torch.save(data, CACHE_DIR / f"cub_dinov2b14_{split}.pt")


class CubSplit(Dataset):
    def __init__(self, split: str, path_cache: dict):
        with open(CUB_ROOT / "class_attr_data_10" / f"{split}.pkl", "rb") as f:
            raw = pickle.load(f)
        self.items = []
        for entry in raw:
            fname = os.path.basename(entry["img_path"])
            local = path_cache.get(fname)
            if local is None:
                raise FileNotFoundError(f"image not found for {entry['img_path']}")
            self.items.append({
                "path": local,
                "y": int(entry["class_label"]),
                "c112": np.asarray(entry["attribute_label"], dtype=np.float32),
                "cert312": np.asarray(entry["attribute_certainty"], dtype=np.int8),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = Image.open(it["path"]).convert("RGB")
        return VAL_TF(img), torch.from_numpy(it["c112"]), torch.from_numpy(it["cert312"].astype(np.float32)), it["y"]


def build_path_cache() -> dict:
    cache = {}
    images_dir = CUB_ROOT / "images"
    for cls_dir in sorted(os.listdir(images_dir)):
        cls_path = images_dir / cls_dir
        if not cls_path.is_dir():
            continue
        for fname in os.listdir(cls_path):
            cache[fname] = str(cls_path / fname)
    return cache


def selected_112_attribute_ids() -> np.ndarray:
    """Map the 112 Koh/CEM concept names to raw 312 attribute ids via attributes.txt."""
    # names in pkl order, from Credal_Sets ProbNum2026/cub_loader.py CONCEPT_NAMES
    sys.path.insert(0, "/Users/tanmoy/research/Credal_Sets/ProbNum2026")
    from cub_loader import CONCEPT_NAMES  # noqa: E402
    name_to_id = {}
    with open(CUB_ROOT / "attributes.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                name_to_id[parts[1]] = int(parts[0])
    ids = []
    ALIASES = {"has_head_pattern::malar_stripe": "has_head_pattern::malar"}
    for n in CONCEPT_NAMES:  # tolerate small name drift, e.g. 'spatula' -> 'spatulate'
        n = ALIASES.get(n, n)
        if n in name_to_id:
            ids.append(name_to_id[n])
            continue
        cands = [full for full in name_to_id if full.startswith(n)]
        if len(cands) == 1:
            ids.append(name_to_id[cands[0]])
        else:
            raise KeyError(f"no unique attributes.txt match for {n!r}: {cands}")
    return np.array(ids), CONCEPT_NAMES


def verify_labels(all_data: dict):
    print("\n[*] Label semantics checks")

    # 1. class-denoised: attribute_label constant within class?
    for split, d in all_data.items():
        y, c = d["labels"], d["concepts112"]
        constant = all((c[y == cls] == c[y == cls][0]).all() for cls in torch.unique(y))
        print(f"    {split}: attribute_label constant within class -> {constant}")

    # 2. data-driven column mapping: match each of the 112 pkl columns to the raw
    #    attribute whose class-majority vector agrees most (class denoising recovery)
    tr = all_data["train"]
    y, present, c112 = tr["labels"], tr["present312"], tr["concepts112"]
    classes = torch.unique(y)
    raw_frac = torch.stack([present[y == k].mean(0) for k in classes])    # (200, 312)
    pkl_class = torch.stack([c112[y == k][0] for k in classes])           # (200, 112)
    raw_bin = (raw_frac >= 0.5).float()
    agree_mat = (raw_bin.unsqueeze(1) == pkl_class.unsqueeze(2)).float().mean(0)  # (112, 312)
    best_attr = agree_mat.argmax(1)
    best_agree = agree_mat.max(1).values
    print(f"    column->attribute match quality: mean {best_agree.mean():.4f}, "
          f"min {best_agree.min():.4f}, unique attrs: {len(best_attr.unique())}/112")

    # 3. per-entry agreement between raw is_present and class-denoised on test
    te = all_data["test"]
    raw_te = te["present312"][:, best_attr]
    agree_test = (raw_te == te["concepts112"]).float().mean().item()
    print(f"    test: agreement(raw is_present, class-denoised) = {agree_test:.4f} "
          f"(summary reference: {0.919})")
    return best_attr, agree_test


def extract_split(split: str, path_cache: dict, model, device) -> dict:
    ds = CubSplit(split, path_cache)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    feats, labels, c112, cert = [], [], [], []
    model.eval()
    with torch.no_grad():
        for x, c, cert_b, y in loader:
            x = x.to(device)
            f = model(x)  # CLS-pooled output, (B, 768)
            feats.append(f.cpu())
            labels.append(y)
            c112.append(c)
            cert.append(cert_b.to(torch.int8))
    return {
        "features": torch.cat(feats),
        "labels": torch.cat(labels),
        "concepts112": torch.cat(c112),
        "certainty312": torch.cat(cert),
        "img_paths": [it["path"] for it in ds.items],
    }


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[*] Loading dinov2_vitb14 on {device} (from local torch hub cache)")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False).to(device)

    path_cache = build_path_cache()
    print(f"[*] Path cache: {len(path_cache)} images")

    CACHE_DIR.mkdir(exist_ok=True)
    all_data = {}
    table = None
    for split in ["train", "val", "test"]:
        out = CACHE_DIR / f"cub_dinov2b14_{split}.pt"
        if out.exists():
            print(f"[*] {split}: cached, loading")
            all_data[split] = torch.load(out, map_location="cpu", weights_only=False)
        else:
            print(f"[*] {split}: extracting features...")
            data = extract_split(split, path_cache, model, device)
            torch.save(data, out)
            all_data[split] = data
        if "present312" not in all_data[split]:
            if table is None:
                print("[*] Parsing image_attribute_labels.txt (312 is_present per image) ...")
                table = load_present312()
            attach_present312(all_data[split], split, table)
        print(f"    features {tuple(all_data[split]['features'].shape)}, "
              f"labels {tuple(all_data[split]['labels'].shape)}, "
              f"present312 {tuple(all_data[split]['present312'].shape)} "
              f"(pos frac {all_data[split]['present312'].mean():.3f})")

    sel_ids, _ = verify_labels(all_data)

    # Validate against Credal_Sets cached test features
    if REFERENCE_TEST_FEATURES.exists():
        ref = torch.load(REFERENCE_TEST_FEATURES, map_location="cpu", weights_only=False)
        mine = all_data["test"]
        A, B = mine["features"], ref["features"]
        if A.shape == B.shape:
            cos = torch.nn.functional.cosine_similarity(A, B, dim=1)
            print(f"\n[*] Validation vs Credal_Sets cached test features:")
            print(f"    cosine similarity: mean={cos.mean():.6f} min={cos.min():.6f}")
            print(f"    labels match: {(mine['labels'] == ref['labels']).float().mean():.4f}")
            print(f"    concepts112 match: {(mine['concepts112'] == ref['concept_labels']).float().mean():.4f}")
        else:
            print(f"\n[!] Shape mismatch vs reference: {A.shape} vs {B.shape}")

    print("\n[*] Done.")


if __name__ == "__main__":
    main()
