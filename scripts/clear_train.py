#!/usr/bin/env python3
"""
CLEAR Training Script for Conformal Concept Models.
Trains joint/sequential CBM and calibrates conformal prediction sets.
Logs results and hyperparameter records directly into the Obsidian Vault.
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
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ccm.data import get_shapes3d_loaders, get_cub_loaders
from ccm.models import ConceptBottleneckModel
from ccm.conformal import ConformalCalibrator
from ccm.scores import get_score_function
from ccm.metrics import empirical_coverage, average_set_size, concept_accuracy
from ccm.utils import set_seed, load_config, get_device, save_checkpoint, VaultLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Train Conformal Concept Bottleneck Model")
    parser.add_argument("--config", type=str, default="configs/clear_shapes3d.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Target device (mps, cuda, cpu)")
    parser.add_argument("--vault_path", type=str, default=None, help="Obsidian vault directory")
    return parser.parse_args()


def train_cbm(config_path: str, args: argparse.Namespace):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    device = get_device(args.device or cfg.get("training", {}).get("device"))
    print(f"[*] Executing on device: {device}")

    # Data loaders
    ds_name = cfg.get("data", {}).get("dataset", "shapes3d").lower()
    if ds_name == "shapes3d":
        loaders = get_shapes3d_loaders(cfg)
    elif ds_name == "cub":
        loaders = get_cub_loaders(cfg)
    else:
        raise ValueError(f"Unsupported dataset: {ds_name}")

    train_loader = loaders["train"]
    calib_loader = loaders["calib"]
    test_loader = loaders["test"]
    num_concepts = loaders["num_concepts"]
    num_classes = loaders["num_classes"]

    # Model
    model_cfg = cfg.get("model", {})
    model = ConceptBottleneckModel(
        num_concepts=num_concepts,
        num_classes=num_classes,
        backbone_type=model_cfg.get("backbone_type", "conv"),
        latent_dim=model_cfg.get("latent_dim", 128),
        hybrid_residual=model_cfg.get("hybrid_residual", False),
        residual_dim=model_cfg.get("residual_dim", 16),
    ).to(device)

    train_cfg = cfg.get("training", {})
    epochs = args.epochs or train_cfg.get("epochs", 10)
    lr = args.lr or train_cfg.get("lr", 0.001)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    concept_weight = train_cfg.get("concept_loss_weight", 1.0)
    task_weight = train_cfg.get("task_loss_weight", 1.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"[*] Training CBM ({ds_name}) for {epochs} epochs...")
    model.train()
    for epoch in range(1, epochs + 1):
        running_task_loss = 0.0
        running_concept_loss = 0.0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for x, c_true, y_true in pbar:
            x, c_true, y_true = x.to(device), c_true.to(device), y_true.to(device)
            optimizer.zero_grad()

            y_logits, c_logits = model(x)

            loss_c = F.binary_cross_entropy_with_logits(c_logits, c_true.float())
            loss_y = F.cross_entropy(y_logits, y_true.long())
            loss = task_weight * loss_y + concept_weight * loss_c

            loss.backward()
            optimizer.step()

            running_task_loss += loss_y.item() * len(y_true)
            running_concept_loss += loss_c.item() * len(y_true)
            total_samples += len(y_true)

            pbar.set_postfix({
                "task_loss": f"{running_task_loss / total_samples:.4f}",
                "c_loss": f"{running_concept_loss / total_samples:.4f}",
            })

    # Save checkpoint
    save_dir = Path(cfg.get("logging", {}).get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"{cfg.get('experiment_name', 'model')}_final.pt"
    save_checkpoint(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "num_concepts": num_concepts,
            "num_classes": num_classes,
        },
        ckpt_path,
    )
    print(f"[*] Saved checkpoint to: {ckpt_path}")

    # Calibration phase
    print("[*] Calibrating Conformal Predictor on held-out calibration split...")
    conf_cfg = cfg.get("conformal", {})
    score_fn = get_score_function(
        conf_cfg.get("score_type", "aps"),
        lambda_reg=conf_cfg.get("lambda_reg", 0.01),
        k_reg=conf_cfg.get("k_reg", 3),
    )
    alpha = conf_cfg.get("alpha", 0.10)
    calibrator = ConformalCalibrator(score_fn=score_fn, alpha=alpha)

    model.eval()
    calib_y_logits, calib_y_trues = [], []
    calib_c_probs, calib_c_trues = [], []

    with torch.no_grad():
        for x, c_true, y_true in calib_loader:
            x, c_true, y_true = x.to(device), c_true.to(device), y_true.to(device)
            y_logits, c_logits = model(x)
            calib_y_logits.append(y_logits)
            calib_y_trues.append(y_true)
            calib_c_probs.append(torch.sigmoid(c_logits))
            calib_c_trues.append(c_true)

    calib_y_logits = torch.cat(calib_y_logits, dim=0)
    calib_y_trues = torch.cat(calib_y_trues, dim=0)
    calib_c_probs = torch.cat(calib_c_probs, dim=0)
    calib_c_trues = torch.cat(calib_c_trues, dim=0)

    q_hat = calibrator.calibrate(calib_y_logits, calib_y_trues, alpha=alpha)
    print(f"[*] Calibrated target threshold q_hat (1-alpha={1-alpha:.2f}): {q_hat:.4f}")

    # Calibrate concept thresholds
    c_thresholds = calibrator.calibrate_concepts(calib_c_probs, calib_c_trues, alpha=alpha, joint=False)

    # Evaluation on Test split
    print("[*] Evaluating on Test split...")
    test_y_logits, test_y_trues = [], []
    test_c_probs, test_c_trues = [], []

    with torch.no_grad():
        for x, c_true, y_true in test_loader:
            x, c_true, y_true = x.to(device), c_true.to(device), y_true.to(device)
            y_logits, c_logits = model(x)
            test_y_logits.append(y_logits)
            test_y_trues.append(y_true)
            test_c_probs.append(torch.sigmoid(c_logits))
            test_c_trues.append(c_true)

    test_y_logits = torch.cat(test_y_logits, dim=0)
    test_y_trues = torch.cat(test_y_trues, dim=0)
    test_c_probs = torch.cat(test_c_probs, dim=0)
    test_c_trues = torch.cat(test_c_trues, dim=0)

    # Standard metrics
    test_preds = torch.argmax(test_y_logits, dim=-1)
    task_acc = float((test_preds == test_y_trues).float().mean().item())
    c_metrics = concept_accuracy(test_c_probs, test_c_trues)
    concept_acc = c_metrics["micro_acc"]

    # Conformal metrics
    pred_sets = calibrator.predict(test_y_logits, alpha=alpha, fill_empty=conf_cfg.get("fill_empty", False))
    coverage = empirical_coverage(pred_sets, test_y_trues)
    avg_size = average_set_size(pred_sets)

    # Concept prediction sets
    concept_sets = calibrator.predict_concepts(test_c_probs, c_thresholds)
    concept_cov = concept_sets.marginal_coverage(test_c_trues)

    results = {
        "test_task_acc": task_acc,
        "test_concept_acc": concept_acc,
        "marginal_coverage": coverage,
        "target_coverage": 1.0 - alpha,
        "alpha": alpha,
        "avg_set_size": avg_size,
        "concept_coverage": concept_cov,
        "q_hat": q_hat,
    }

    print("\n" + "=" * 45)
    print("           TRAINING & AUDIT SUMMARY")
    print("=" * 45)
    for k, v in results.items():
        print(f"  {k:<22}: {v:.4f}")
    print("=" * 45 + "\n")

    # Vault Logging
    vault_path = args.vault_path or cfg.get("logging", {}).get("vault_path", "vault")
    vault_dir = PROJECT_ROOT / vault_path
    logger = VaultLogger(vault_path=vault_dir)

    log_file = logger.log_run(
        experiment_type=ds_name,
        run_name=f"{cfg.get('experiment_name', 'clear_train')}",
        config=cfg,
        metrics=results,
        notes=f"Trained {epochs} epochs on {ds_name} with {conf_cfg.get('score_type', 'aps').upper()} non-conformity score.",
    )
    print(f"[*] Run logged to Obsidian Vault at: {log_file}")

    return results


def main():
    args = parse_args()
    config_file = PROJECT_ROOT / args.config if not os.path.isabs(args.config) else Path(args.config)
    train_cbm(str(config_file), args)


if __name__ == "__main__":
    main()
