#!/usr/bin/env python3
"""
Nearest Neighbor Agreement and Concept Leakage Audit Script.
Analyzes information leakage from intermediate representations z vs concept bottleneck c
via k-NN agreement, predictive fidelity, and residual task leakage diagnostics.
Logs findings to the Obsidian Vault.
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# Add src to pythonpath
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier

from ccm.data import get_shapes3d_loaders, get_cub_loaders
from ccm.models import ConceptBottleneckModel
from ccm.metrics import nn_leakage_score
from ccm.utils import set_seed, load_config, get_device, load_checkpoint, VaultLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Nearest Neighbor Agreement and Concept Leakage")
    parser.add_argument("--config", type=str, default="configs/clear_shapes3d.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint")
    parser.add_argument("--neighbors", type=int, default=5, help="Number of nearest neighbors (k)")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--vault_path", type=str, default=None, help="Obsidian vault directory")
    return parser.parse_args()


def run_nn_agreement(config_path: str, args: argparse.Namespace):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    device = get_device(args.device or cfg.get("training", {}).get("device"))
    print(f"[*] Running NN-Agreement Audit on device: {device} (k={args.neighbors})")

    ds_name = cfg.get("data", {}).get("dataset", "shapes3d").lower()
    loaders = get_shapes3d_loaders(cfg) if ds_name == "shapes3d" else get_cub_loaders(cfg)
    test_loader = loaders["test"]
    num_concepts = loaders["num_concepts"]
    num_classes = loaders["num_classes"]

    model = ConceptBottleneckModel(
        num_concepts=num_concepts,
        num_classes=num_classes,
        backbone_type=cfg.get("model", {}).get("backbone_type", "conv"),
        latent_dim=cfg.get("model", {}).get("latent_dim", 128),
    ).to(device)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        default_ckpt = Path(cfg.get("logging", {}).get("save_dir", "checkpoints")) / f"{cfg.get('experiment_name', 'model')}_final.pt"
        if default_ckpt.exists():
            ckpt_path = str(default_ckpt)

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"[*] Loading checkpoint: {ckpt_path}")
        state = load_checkpoint(ckpt_path, device=device)
        model.load_state_dict(state["model_state_dict"])
    else:
        print("[!] Checkpoint not found. Initializing and running quick feature extraction...")
        # Train for 2 epochs for demonstration
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(2):
            for x, c, y in loaders["train"]:
                x, c, y = x.to(device), c.to(device), y.to(device)
                optimizer.zero_grad()
                yl, cl = model(x)
                loss = F.cross_entropy(yl, y) + F.binary_cross_entropy_with_logits(cl, c.float())
                loss.backward()
                optimizer.step()

    model.eval()

    # Extract all test representations z, concepts c, and labels y
    all_z, all_c, all_y = [], [], []
    with torch.no_grad():
        for x, c_true, y_true in test_loader:
            x = x.to(device)
            z, c_logits = model.extract_representations(x)
            c_probs = torch.sigmoid(c_logits)
            all_z.append(z.cpu())
            all_c.append(c_probs.cpu())
            all_y.append(y_true.cpu())

    z_tensor = torch.cat(all_z, dim=0)
    c_tensor = torch.cat(all_c, dim=0)
    y_tensor = torch.cat(all_y, dim=0)

    # Compute NN leakage metrics
    k = args.neighbors
    metrics = nn_leakage_score(z_tensor, c_tensor, y_tensor, n_neighbors=k)

    # Compute direct prediction agreement between kNN(z) and kNN(c)
    z_np = z_tensor.numpy()
    c_np = c_tensor.numpy()
    y_np = y_tensor.numpy()

    # Standardize
    z_norm = (z_np - z_np.mean(axis=0)) / (z_np.std(axis=0) + 1e-6)
    c_norm = (c_np - c_np.mean(axis=0)) / (c_np.std(axis=0) + 1e-6)

    split = int(len(y_np) * 0.7)
    knn_z = KNeighborsClassifier(n_neighbors=k).fit(z_norm[:split], y_np[:split])
    knn_c = KNeighborsClassifier(n_neighbors=k).fit(c_norm[:split], y_np[:split])

    preds_z = knn_z.predict(z_norm[split:])
    preds_c = knn_c.predict(c_norm[split:])

    nn_agreement_rate = float(np.mean(preds_z == preds_c))
    metrics["nn_agreement_rate"] = nn_agreement_rate

    print("\n" + "=" * 45)
    print("      NEAREST-NEIGHBOR LEAKAGE AUDIT")
    print("=" * 45)
    print(f"  k-NN Accuracy (Latent z)     : {metrics['nn_acc_latent']:.4f}")
    print(f"  k-NN Accuracy (Concepts c)   : {metrics['nn_acc_concept']:.4f}")
    print(f"  Leakage Gap (z - c)          : {metrics['leakage_gap']:.4f}")
    print(f"  Leakage Ratio (z / c)        : {metrics['leakage_ratio']:.4f}")
    print(f"  Agreement Rate [kNN(z)==kNN(c)]: {metrics['nn_agreement_rate']:.4f}")
    print("=" * 45 + "\n")

    # Vault Logging
    vault_path = args.vault_path or cfg.get("logging", {}).get("vault_path", "vault")
    vault_dir = PROJECT_ROOT / vault_path
    vault_dir.mkdir(parents=True, exist_ok=True)
    report_note = vault_dir / "03 - Experiments" / f"NN_Agreement_Audit_{ds_name.title()}.md"

    leakage_status = "🟢 Minimal Leakage" if metrics["leakage_gap"] < 0.05 else ("🟡 Moderate Leakage" if metrics["leakage_gap"] < 0.15 else "🔴 Severe Concept Leakage")

    md_lines = [
        "---",
        f"title: \"Nearest Neighbor Agreement Audit - {ds_name.upper()}\"",
        f"dataset: {ds_name}",
        "type: leakage-diagnostic",
        "tags: [nn-agreement, concept-leakage, fidelity]",
        "---",
        f"# 🔍 Nearest-Neighbor Concept Leakage Diagnostic: {ds_name.upper()}",
        f"> **Audited on:** `{len(y_np)}` samples | **Neighborhood Size:** `k={k}` | **Assessment:** {leakage_status}\n",
        "## 1. Leakage & Agreement Scorecard",
        "| Metric | Observed Value | Theoretical Ideal | Status |",
        "| :--- | :--- | :--- | :--- |",
        f"| **k-NN Accuracy (Latent $z$)** | `{metrics['nn_acc_latent']:.4f}` | Equal to $c$ | Latent Fidelity |",
        f"| **k-NN Accuracy (Concepts $c$)** | `{metrics['nn_acc_concept']:.4f}` | High | Bottleneck Sufficiency |",
        f"| **Leakage Gap ($z - c$)** | `{metrics['leakage_gap']:+.4f}` | `0.0000` | {leakage_status} |",
        f"| **Leakage Ratio ($z / c$)** | `{metrics['leakage_ratio']:.4f}` | `1.0000` | Multiplicative Bias |",
        f"| **Latent vs Concept Agreement** | `{metrics['nn_agreement_rate']*100:.2f}%` | `100.0%` | Decision Consistency |",
        "\n## 2. Interpretation & Conformal Implications",
        "- **Information Bottleneck Integrity**: When the leakage gap is low, downstream conformal sets calibrated on concepts accurately reflect true prediction uncertainty without unmodeled latent side-channels.",
        "- **Intervention Reliability**: High NN agreement guarantees that intervening on human concepts will produce predictable, faithful shifts in model behavior rather than being overridden by leaked latent variables.",
        "\n## 🔗 Related Notes",
        "- [[00 - Dashboards/Audit Dashboard|Audit Dashboard]]",
        "- [[01 - Literature & Theory/Concept Bottleneck Models|CBM Theory]]",
        "- [[00 - Dashboards/Experiment Tracker|Experiment Tracker]]",
    ]

    with open(report_note, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # Log to logger
    logger = VaultLogger(vault_path=vault_dir)
    logger.log_run(
        experiment_type="nn-audit",
        run_name=f"nn_agreement_{ds_name}",
        config=cfg,
        metrics=metrics,
        notes=f"k-NN leakage audit evaluated with k={k}. Leakage gap: {metrics['leakage_gap']:.4f}.",
    )

    print(f"[*] NN Agreement audit logged to Obsidian Vault at: {report_note}")


def main():
    args = parse_args()
    config_file = PROJECT_ROOT / args.config if not os.path.isabs(args.config) else Path(args.config)
    run_nn_agreement(str(config_file), args)


if __name__ == "__main__":
    main()
