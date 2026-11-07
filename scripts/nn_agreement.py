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

import torch
import torch.nn.functional as F

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
        model_source = f"checkpoint:{ckpt_path}"
    else:
        print("[!] Checkpoint not found. Initializing and running quick feature extraction...")
        # Train for 2 epochs for demonstration -- NOT a converged model.
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
        model_source = "quick-train-2-epochs"

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

    # Compute NN leakage metrics (stratified K-fold, fold-fitted scaling, signed gap,
    # embedding-null calibration -- see ccm.metrics.nn_leakage_score docstring)
    k = args.neighbors
    metrics = nn_leakage_score(
        z_tensor, c_tensor, y_tensor,
        n_neighbors=k, n_splits=5, seed=cfg.get("seed", 42),
    )
    metrics["model_source"] = model_source
    metrics["leakage_gap_ci_lo"] = metrics["leakage_gap_ci95"][0]
    metrics["leakage_gap_ci_hi"] = metrics["leakage_gap_ci95"][1]

    print("\n" + "=" * 45)
    print("      NEAREST-NEIGHBOR LEAKAGE AUDIT")
    print("=" * 45)
    print(f"  k-NN Accuracy (Latent z)     : {metrics['nn_acc_latent']:.4f}")
    print(f"  k-NN Accuracy (Concepts c)   : {metrics['nn_acc_concept']:.4f}")
    print(f"  Leakage Gap (z - c), SIGNED  : {metrics['leakage_gap']:+.4f}  "
          f"(95% CI [{metrics['leakage_gap_ci_lo']:+.4f}, {metrics['leakage_gap_ci_hi']:+.4f}])")
    print(f"  Null band (z from c only)    : {metrics['null_gap_mean']:+.4f} ± {metrics['null_gap_sd']:.4f}")
    print(f"  Gap vs null, z-score         : {metrics['gap_vs_null_z']:+.2f}   [leakage evidence]")
    print(f"  Projection gap (descriptive) : {metrics['projection_gap_mean']:+.4f} ± {metrics['projection_gap_sd']:.4f}")
    print(f"  Leakage Ratio (z / c)        : {metrics['leakage_ratio']:.4f}")
    print(f"  Agreement Rate [kNN(z)==kNN(c)]: {metrics['nn_agreement_rate']:.4f}")
    print(f"  Model source                 : {model_source}")
    print("=" * 45 + "\n")

    # Vault Logging
    vault_path = args.vault_path or cfg.get("logging", {}).get("vault_path", "vault")
    vault_dir = PROJECT_ROOT / vault_path
    vault_dir.mkdir(parents=True, exist_ok=True)
    report_note = vault_dir / "03 - Experiments" / f"NN_Agreement_Audit_{ds_name.title()}.md"

    z_stat = metrics["gap_vs_null_z"]
    leakage_status = (
        "🔴 Severe (gap beyond null)" if z_stat > 4
        else "🟡 Moderate (gap beyond null)" if z_stat > 2
        else "🟢 Within the null band"
    )
    quick_train_warn = (
        "> [!FAILURE] INVALID for paper claims — model was quick-trained (2 epochs), not loaded from a checkpoint.\n"
        if model_source == "quick-train-2-epochs" else ""
    )

    md_lines = [
        "---",
        f"title: \"Nearest Neighbor Agreement Audit - {ds_name.upper()}\"",
        f"dataset: {ds_name}",
        "type: leakage-diagnostic",
        f"model_source: {model_source}",
        "tags: [nn-agreement, concept-leakage, fidelity]",
        "---",
        f"# 🔍 Nearest-Neighbor Concept Leakage Diagnostic: {ds_name.upper()}",
        quick_train_warn,
        f"> **Audited on:** `{len(y_tensor)}` samples | **Neighborhood Size:** `k={k}` | "
        f"**Model source:** `{model_source}` | **Assessment:** {leakage_status}\n",
        "## 1. Leakage & Agreement Scorecard",
        "| Metric | Observed Value | Reference | Status |",
        "| :--- | :--- | :--- | :--- |",
        f"| **k-NN Accuracy (Latent $z$)** | `{metrics['nn_acc_latent']:.4f}` | — | Latent Fidelity |",
        f"| **k-NN Accuracy (Concepts $c$)** | `{metrics['nn_acc_concept']:.4f}` | — | Bottleneck Sufficiency |",
        f"| **Leakage Gap ($z - c$), signed** | `{metrics['leakage_gap']:+.4f}` | null band, not 0 | {leakage_status} |",
        f"| **Gap 95% CI (folds)** | `[{metrics['leakage_gap_ci_lo']:+.4f}, {metrics['leakage_gap_ci_hi']:+.4f}]` | contains 0? | Estimation noise |",
        f"| **Null band ($z$ synthesized from $c$ only)** | `{metrics['null_gap_mean']:+.4f} ± {metrics['null_gap_sd']:.4f}` | observed gap inside? | Calibrated null |",
        f"| **Gap vs null (z-score)** | `{z_stat:+.2f}` | $|z| \\leq 2$ | Leakage evidence |",
        f"| **Projection gap (descriptive)** | `{metrics['projection_gap_mean']:+.4f} ± {metrics['projection_gap_sd']:.4f}` | — | $z$ vs random compression |",
        f"| **Latent vs Concept Agreement** | `{metrics['nn_agreement_rate']*100:.2f}%` | `100.0%` | Decision Consistency |",
        "\n## 2. Interpretation & Conformal Implications",
        "- **Signed gap**: negative values mean the concept bottleneck is MORE predictive than the latent (e.g., denoising); v1 clipped these to 0 and misreported them as \"no leakage\".",
        "- **Null band, not a subtraction**: the observed gap is judged against the distribution of gaps from $z$ re-synthesized from $c$ alone (same dimension, exactly the concepts' information). An earlier version subtracted a single random-projection gap -- algebraically that reduces to $\\mathrm{acc}_{proj} - \\mathrm{acc}_c$ (the latent's accuracy cancels) and is not calibrated.",
        "- **Intervention Reliability**: high NN agreement suggests intervening on human concepts produces predictable shifts, provided the observed gap stays within the null band.",
        "\n## 🔗 Related Notes",
        "- [[00 - Dashboards/Audit Dashboard|Audit Dashboard]]",
        "- [[01 - Literature & Theory/Concept Bottleneck Models|CBM Theory]]",
        "- [[00 - Dashboards/Experiment Tracker|Experiment Tracker]]",
    ]

    with open(report_note, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # Log to logger (drop the CI list; flat lo/hi keys keep the frontmatter scalar)
    log_metrics = {k: v for k, v in metrics.items() if k != "leakage_gap_ci95"}
    logger = VaultLogger(vault_path=vault_dir)
    logger.log_run(
        experiment_type="nn-audit",
        run_name=f"nn_agreement_{ds_name}",
        config=cfg,
        metrics=log_metrics,
        notes=(f"k-NN leakage audit, k={k}, stratified 5-fold, model_source={model_source}. "
               f"Signed gap {metrics['leakage_gap']:+.4f}; null band "
               f"{metrics['null_gap_mean']:+.4f} ± {metrics['null_gap_sd']:.4f}; "
               f"z {metrics['gap_vs_null_z']:+.2f}."),
    )

    print(f"[*] NN Agreement audit logged to Obsidian Vault at: {report_note}")


def main():
    args = parse_args()
    config_file = PROJECT_ROOT / args.config if not os.path.isabs(args.config) else Path(args.config)
    run_nn_agreement(str(config_file), args)


if __name__ == "__main__":
    main()
