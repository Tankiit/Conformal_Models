#!/usr/bin/env python3
"""
CLeaR paired-nuisance pilot (Roadmap step 3; vault spec settled 2026-11-11).

Runs the rho x model grid on the paired Shapes3D generator. Per D2, each model
runs at rho = 0 FIRST -- that cell's Test-2 set-change rate IS the null the
rho = 1 cells are judged against (zero set-change is NOT the leak-free
signature in the deployed pipeline). Models: cbm, residual (cbm + residual
channel), blackbox (no bottleneck; Test 1 undefined).

Per cell: test accuracy, no-fill APS coverage / set size, prediction-flip
rate under U-flips, Test 2 (primary) and Test 1 (attribution control)
set-change rates with shared-u APS, the (x, x) u-pairing sanity null (must be
exactly 0), and the nn-leakage v3 audit on the same run.
"""

from __future__ import annotations
import argparse
import copy
import sys
from pathlib import Path

# Add src to pythonpath
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ccm.data import get_paired_shapes3d_loaders
from ccm.models import ConceptBottleneckModel
from ccm.models.cbm import ConvBackbone
from ccm.conformal import ConformalCalibrator
from ccm.scores import get_score_function
from ccm.interventions import paired_set_change
from ccm.metrics import nn_leakage_score
from ccm.utils import set_seed, load_config, get_device, VaultLogger


class BlackBoxClassifier(nn.Module):
    """No-bottleneck reference: conv backbone straight to the label head.

    forward accepts (and ignores) the intervention kwargs so it can flow
    through shared eval code; Test 1 is skipped for this model because it has
    no concept interface to fix.
    """

    def __init__(self, num_classes: int, latent_dim: int = 128):
        super().__init__()
        self.backbone = ConvBackbone(in_channels=3, latent_dim=latent_dim)
        self.head = nn.Linear(latent_dim, num_classes)

    def forward(self, x, intervened_concepts=None, intervention_mask=None):
        return self.head(self.backbone(x)), None


def parse_args():
    parser = argparse.ArgumentParser(description="CLeaR paired-nuisance pilot grid")
    parser.add_argument("--config", type=str, default="configs/clear_paired_shapes3d.yaml")
    parser.add_argument("--rhos", type=str, default=None, help="Comma-separated override, e.g. '0.0,1.0'")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated override: cbm,residual,blackbox")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--vault_path", type=str, default=None)
    return parser.parse_args()


def build_model(kind: str, num_concepts: int, num_classes: int, model_cfg: dict):
    if kind == "blackbox":
        return BlackBoxClassifier(num_classes, latent_dim=model_cfg.get("latent_dim", 128))
    return ConceptBottleneckModel(
        num_concepts=num_concepts,
        num_classes=num_classes,
        backbone_type=model_cfg.get("backbone_type", "conv"),
        latent_dim=model_cfg.get("latent_dim", 128),
        hybrid_residual=(kind == "residual"),
        residual_dim=model_cfg.get("residual_dim", 16),
    )


def train_model(model, train_loader, cfg, device, kind: str):
    t_cfg = cfg.get("training", {})
    epochs = t_cfg.get("epochs", 8)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=t_cfg.get("lr", 1e-3),
        weight_decay=t_cfg.get("weight_decay", 1e-4),
    )
    c_w, y_w = t_cfg.get("concept_loss_weight", 1.0), t_cfg.get("task_loss_weight", 1.0)

    model.train()
    for epoch in range(1, epochs + 1):
        running, total = 0.0, 0
        pbar = tqdm(train_loader, desc=f"  epoch {epoch}/{epochs}", leave=False)
        for x, c_true, y_true, _u in pbar:
            x, c_true, y_true = x.to(device), c_true.to(device), y_true.to(device)
            optimizer.zero_grad()
            y_logits, c_logits = model(x)
            loss = y_w * F.cross_entropy(y_logits, y_true.long())
            if kind != "blackbox":
                loss = loss + c_w * F.binary_cross_entropy_with_logits(c_logits, c_true.float())
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y_true)
            total += len(y_true)
            pbar.set_postfix(loss=f"{running / total:.4f}")


@torch.no_grad()
def collect(model, loader, device):
    """Returns logits, labels, true concepts, backbone latents z for a split."""
    logits, ys, cs, zs = [], [], [], []
    model.eval()
    for x, c_true, y_true, _u in loader:
        x = x.to(device)
        y_logits, _ = model(x)
        logits.append(y_logits.cpu())
        ys.append(y_true)
        cs.append(c_true)
        zs.append(model.backbone(x).cpu())
    return torch.cat(logits), torch.cat(ys), torch.cat(cs), torch.cat(zs)


@torch.no_grad()
def flip_rate(model, pairs, device):
    """Argmax flip rate under U-flips (the probability-sensitivity cousin)."""
    flips, total = 0, 0
    for x, x_flip, _c in pairs:
        x, x_flip = x.to(device), x_flip.to(device)
        pred = model(x)[0].argmax(dim=-1).cpu()
        pred_flip = model(x_flip)[0].argmax(dim=-1).cpu()
        flips += int((pred != pred_flip).sum().item())
        total += len(pred)
    return flips / total


def evaluate_cell(model, loaders, cfg, device, kind: str) -> dict:
    conf_cfg = cfg.get("conformal", {})
    alpha = conf_cfg.get("alpha", 0.10)
    calibrator = ConformalCalibrator(
        score_fn=get_score_function(conf_cfg.get("score_type", "aps")), alpha=alpha
    )
    cal_logits, cal_y, _, _ = collect(model, loaders["calib"], device)
    calibrator.calibrate(cal_logits, cal_y, alpha=alpha)

    te_logits, te_y, te_c, te_z = collect(model, loaders["test"], device)
    preds = te_logits.argmax(dim=-1)
    sets = calibrator.predict(te_logits, alpha=alpha, fill_empty=False)

    # Materialize pair batches once (memory: 2 x n_test images; fine at pilot
    # scale -- stream from the loader instead if n_test grows large).
    pairs = [(x, x_flip, c) for x, x_flip, c in loaders["pairs_test"]]
    sanity = [(x, x, c) for x, _x_flip, c in pairs]  # u-pairing null: exactly 0

    res = {
        "test_acc": float((preds == te_y).float().mean().item()),
        "coverage": float(sets.contains(te_y).float().mean().item()),
        "avg_set_size": float(sets.sizes.float().mean().item()),
        "sanity_null_set_change": paired_set_change(
            model, calibrator, sanity, use_gt_concepts=False, seed=0
        )["set_change_rate"],
        "flip_rate": flip_rate(model, pairs, device),
        "set_change_t2": paired_set_change(
            model, calibrator, pairs, use_gt_concepts=False, seed=0
        ),
    }
    if kind != "blackbox":
        res["set_change_t1"] = paired_set_change(
            model, calibrator, pairs, use_gt_concepts=True, seed=0
        )

    audit_cfg = cfg.get("nn_audit", {})
    res["nn_leakage"] = nn_leakage_score(
        te_z, te_c, te_y,
        n_neighbors=audit_cfg.get("n_neighbors", 5),
        n_splits=audit_cfg.get("n_splits", 5),
        n_null_draws=audit_cfg.get("n_null_draws", 8),
        n_control_projections=audit_cfg.get("n_control_projections", 5),
    )
    return res


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device(args.device or cfg.get("training", {}).get("device"))
    print(f"[*] Executing on device: {device}")

    grid = cfg.get("grid", {})
    rhos = [float(r) for r in (args.rhos.split(",") if args.rhos else grid.get("rhos", [0.0, 1.0]))]
    kinds = args.models.split(",") if args.models else grid.get("models", ["cbm"])
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs

    rows, flat_metrics = [], {}
    for kind in kinds:
        for rho in rhos:
            cell_cfg = copy.deepcopy(cfg)
            cell_cfg["data"]["rho"] = rho
            seed = int(cfg.get("seed", 42)) + int(rho) * 10 + {"cbm": 0, "residual": 1, "blackbox": 2}.get(kind, 3)
            set_seed(seed)

            if cell_cfg.get("data", {}).get("source", "procedural") == "real":
                from ccm.data.shapes3d_real import get_real_paired_shapes3d_loaders

                loaders = get_real_paired_shapes3d_loaders(cell_cfg)
            else:
                loaders = get_paired_shapes3d_loaders(cell_cfg)
            model = build_model(
                kind, loaders["num_concepts"], loaders["num_classes"], cfg.get("model", {})
            ).to(device)
            print(
                f"[*] {kind} @ rho={rho} ({cell_cfg.get('data', {}).get('source', 'procedural')}): "
                f"{loaders['num_classes']} classes, n={loaders.get('n_samples', '?')}, training..."
            )
            train_model(model, loaders["train"], cell_cfg, device, kind)

            res = evaluate_cell(model, loaders, cell_cfg, device, kind)
            tag = f"{kind}_rho{int(rho)}"
            row = {
                "model": kind, "rho": rho,
                "acc": res["test_acc"], "cov": res["coverage"], "size": res["avg_set_size"],
                "flip": res["flip_rate"],
                "sanity0": res["sanity_null_set_change"],
                "t2_change": res["set_change_t2"]["set_change_rate"],
                "t2_pshift": res["set_change_t2"]["mean_prob_shift_l1"],
                "t1_change": res["set_change_t1"]["set_change_rate"] if "set_change_t1" in res else None,
                "nn_gap_z": res["nn_leakage"]["gap_vs_null_z"],
            }
            rows.append(row)
            flat_metrics[f"{tag}_acc"] = row["acc"]
            flat_metrics[f"{tag}_coverage"] = row["cov"]
            flat_metrics[f"{tag}_set_size"] = row["size"]
            flat_metrics[f"{tag}_flip_rate"] = row["flip"]
            flat_metrics[f"{tag}_sanity_null"] = row["sanity0"]
            flat_metrics[f"{tag}_set_change_t2"] = row["t2_change"]
            flat_metrics[f"{tag}_prob_shift_l1"] = row["t2_pshift"]
            if row["t1_change"] is not None:
                flat_metrics[f"{tag}_set_change_t1"] = row["t1_change"]
            flat_metrics[f"{tag}_nn_gap_vs_null_z"] = row["nn_gap_z"]
            print(
                f"    acc={row['acc']:.3f} cov={row['cov']:.3f} size={row['size']:.2f} | "
                f"flip={row['flip']:.3f} T2-set-change={row['t2_change']:.3f} "
                f"(prob-shift {row['t2_pshift']:.3f}) "
                f"T1={row['t1_change'] if row['t1_change'] is None else round(row['t1_change'], 3)} "
                f"| sanity(x,x)={row['sanity0']:.3f} nn_gap_z={row['nn_gap_z']}"
            )

    header = "| model | rho | acc | cov | set size | flip rate | T2 set-change | T2 prob-shift | T1 set-change | sanity (x,x) | nn gap_vs_null_z |"
    sep = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    table = [header, sep] + [
        f"| {r['model']} | {r['rho']} | {r['acc']:.3f} | {r['cov']:.3f} | {r['size']:.2f} | "
        f"{r['flip']:.3f} | {r['t2_change']:.3f} | {r['t2_pshift']:.3f} | "
        f"{r['t1_change'] if r['t1_change'] is None else f'{r['t1_change']:.3f}'} | "
        f"{r['sanity0']:.3f} | {r['nn_gap_z']} |"
        for r in rows
    ]
    notes = (
        "Read against the rho=0 row of the SAME model: the Test-2 set-change "
        "rate at rho=0 is the null (deployed pipeline; concept head sees the "
        "whole image), per D5. sanity(x,x) must be exactly 0 (shared-u APS); "
        "any nonzero value invalidates the cell. rho=0 runs FIRST by design "
        "(D2).\n\n" + "\n".join(table)
    )
    print("\n" + "\n".join(table))

    if cfg.get("logging", {}).get("log_to_vault", True):
        vault_dir = PROJECT_ROOT / (args.vault_path or cfg.get("logging", {}).get("vault_path", "vault"))
        logger = VaultLogger(vault_path=vault_dir)
        log_file = logger.log_run(
            experiment_type="clear-paired-pilot",
            run_name=f"CLeaR Paired Pilot {Path(args.config).stem}",
            config={"config": args.config, "rhos": rhos, "models": kinds,
                    "epochs": cfg["training"].get("epochs"),
                    "num_samples": cfg.get("data", {}).get("num_samples"),
                    "scale_buckets": cfg.get("data", {}).get("scale_buckets", 2)},
            metrics=flat_metrics,
            notes=notes,
            tags=["clear", "leakage", "paired", "shapes3d"],
        )
        print(f"[*] Vault log written: {log_file}")


if __name__ == "__main__":
    main()
