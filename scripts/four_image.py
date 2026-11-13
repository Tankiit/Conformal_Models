#!/usr/bin/env python3
"""
Four-image (2x2) causal evaluation (2026-11-13 review): wall A/B x size on
opposite sides of the scale-bucket boundary, for one held-out scene config.

    |            | wall A | wall B |
    | small size |  y_s   |  y_s   |
    | large size |  y_s'  |  y_s'  |

All four renders are REAL grid images. Model, threshold, and the APS draw are
shared across each group's four renders, so differences are causal in the
render. Three questions:

  Q1 nuisance sensitivity  -- at fixed size, does flipping the wall REMOVE the
     correct label (directional coverage loss, both wall directions)?
  Q2 size sensitivity      -- at fixed wall, does the set keep its OWN label,
     exclude the other size, admit both (ambiguity), or confidently err?
  Q3 concept correction    -- on wall-induced losses, does replacing ONLY the
     predicted scale concept with ground truth restore the correct label?
     Controls: object-hue correction (irrelevant), full clamp (reference).

Usage:
  python scripts/four_image.py --config configs/clear_paired_shapes3d_real.yaml \
      --rhos 0.0,1.0 --models cbm,residual --n_groups 400
"""

from __future__ import annotations
import argparse
import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import torch

from clear_paired import build_model, train_model, collect, collect_paired_logits
from ccm.conformal import ConformalCalibrator
from ccm.data.shapes3d_real import get_real_paired_shapes3d_loaders
from ccm.interventions import paired_set_change  # noqa: F401 (shared-u score path)
from ccm.scores import APSScore, get_score_function
from ccm.utils import set_seed, load_config, get_device, VaultLogger


def parse_args():
    p = argparse.ArgumentParser(description="Four-image 2x2 causal evaluation")
    p.add_argument("--config", type=str, default="configs/clear_paired_shapes3d_real.yaml")
    p.add_argument("--rhos", type=str, default="0.0,1.0")
    p.add_argument("--models", type=str, default="cbm,residual")
    p.add_argument("--n_groups", type=int, default=400)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--vault_path", type=str, default=None)
    return p.parse_args()


@torch.no_grad()
def group_renders(dataset, combos):
    """combos: (G, 4) tensor of base scenes (obj, s, shape, orient).
    Returns renders (G, 4, 3, H, W) ordered [sA, sB, s2A, s2B], partner scene
    scale levels, and the two labels per group."""
    half = dataset._levels["n_scale"] // 2
    G = combos.shape[0]
    out = torch.empty(G, 4, 3, dataset.image_size, dataset.image_size)
    s2_of = torch.empty(G, dtype=torch.long)
    y1 = torch.empty(G, dtype=torch.long)
    y2 = torch.empty(G, dtype=torch.long)
    for g in range(G):
        obj, s, shape, orient = (int(v) for v in combos[g])
        s2 = dataset.scale_partner_level(s)
        s2_of[g] = s2
        b1 = 1 if s >= half else 0
        y1[g] = shape * 2 + b1
        y2[g] = shape * 2 + (1 - b1)
        for r, (ss, u) in enumerate([(s, 0), (s, 1), (s2, 0), (s2, 1)]):
            out[g, r] = dataset.render((obj, ss, shape, orient), u)
    return out, s2_of, y1, y2


@torch.no_grad()
def corrected_logits(model, x, combo_gt_concept, coord_idx, device):
    """Logits after replacing ONLY concept coordinate `coord_idx` of the
    PREDICTED concept vector with its ground-truth value (partial clamp)."""
    z, c_logits = model.extract_representations(x)
    c_hat = torch.sigmoid(c_logits)
    gt = combo_gt_concept.to(c_hat.device).unsqueeze(0)
    mask = torch.zeros(1, c_hat.shape[1], dtype=torch.bool, device=c_hat.device)
    mask[0, coord_idx] = True
    c_corr = torch.where(mask, gt.expand_as(c_hat), c_hat)
    logits, _ = model(x, intervened_concepts=c_corr, intervention_mask=mask.expand(x.shape[0], -1))
    return logits


def run_cell(kind, rho, args, cfg, device):
    cell_cfg = copy.deepcopy(cfg)
    cell_cfg["data"]["rho"] = rho
    seed = int(cfg.get("seed", 42)) + {"cbm": 0, "residual": 1, "blackbox": 2}.get(kind, 3)
    set_seed(seed)
    loaders = get_real_paired_shapes3d_loaders(cell_cfg)
    dataset = loaders["pairs_test"].dataset.base
    model = build_model(kind, loaders["num_concepts"], loaders["num_classes"], cfg.get("model", {})).to(device)
    print(f"[*] {kind} @ rho={rho:g}: training {cfg['training'].get('epochs', 8)} epochs ...")
    train_model(model, loaders["train"], cell_cfg, device, kind)

    alpha = cfg.get("conformal", {}).get("alpha", 0.10)
    score_fn = get_score_function(cfg.get("conformal", {}).get("score_type", "aps"))
    cal = ConformalCalibrator(score_fn=score_fn, alpha=alpha)
    cal_logits, cal_y, _, _ = collect(model, loaders["calib"], device)
    cal.calibrate(cal_logits, cal_y, alpha=alpha)
    q_ord = cal.calibrated_alphas[alpha]

    paired = ConformalCalibrator(score_fn=score_fn, alpha=alpha)
    pairs_cal = [(x, x_flip, c, y) for x, x_flip, c, y in loaders["pairs_calib"]]
    pa, pb, py = collect_paired_logits(model, pairs_cal, device)
    paired.calibrate_paired(pa, pb, py, alpha=alpha, seed=0)
    q_pair = paired.calibrated_alphas[alpha]

    # held-out groups from the TEST scene split
    test_scene_ids = dataset.scene_ids[torch.tensor(loaders["test"].dataset.indices)]
    uniq = torch.unique(test_scene_ids)
    rng = np.random.RandomState(0)
    sel = uniq[torch.as_tensor(rng.permutation(len(uniq))[: args.n_groups])]
    combos = dataset.scene_combos[sel]
    renders, s2_of, y1, y2 = group_renders(dataset, combos)
    G, K = len(combos), loaders["num_classes"]

    model.eval()
    aps = APSScore(from_logits=True, randomized=True)
    g_rng = torch.Generator().manual_seed(0)
    u = torch.rand(G, 1, generator=g_rng)

    results = {}
    for thr_name, thr in (("ord", q_ord), ("paired", q_pair)):
        logits_all = torch.empty(G, 4, K)
        for r in range(4):
            for i in range(0, G, 128):
                xb = renders[i:i + 128, r].to(device)
                logits_all[i:i + 128, r] = model(xb)[0].cpu()
        # shared-u APS scores per render (same u across the group's 4 renders)
        scores = torch.empty(G, 4, K)
        for r in range(4):
            s = aps(logits_all[:, r], None, uniforms=u)
            scores[:, r] = s
        masks = scores <= thr  # (G, 4, K)
        arange = torch.arange(G)

        own = torch.stack([masks[arange, 0, y1], masks[arange, 1, y1],
                           masks[arange, 2, y2], masks[arange, 3, y2]], dim=1)  # (G,4) own-label cover
        # Q1: nuisance loss at fixed size (wall flip), both directions, both sizes
        nu_A2B = (own[:, 0] & ~own[:, 1]) | (own[:, 2] & ~own[:, 3])
        nu_B2A = (own[:, 1] & ~own[:, 0]) | (own[:, 3] & ~own[:, 2])
        # Q2: size sensitivity at fixed wall, averaged over walls
        own_w = torch.cat([masks[arange, 0, y1], masks[arange, 1, y1]])       # own label at size s
        oth_w = torch.cat([masks[arange, 0, y2], masks[arange, 1, y2]])
        own_l = torch.cat([masks[arange, 2, y2], masks[arange, 3, y2]])       # own label at size s2
        oth_l = torch.cat([masks[arange, 2, y1], masks[arange, 3, y1]])
        own_all = torch.cat([own_w, own_l]); oth_all = torch.cat([oth_w, oth_l])
        res = {
            "own_cover": own.float().mean().item(),
            "nuis_loss_A2B": nu_A2B.float().mean().item(),
            "nuis_loss_B2A": nu_B2A.float().mean().item(),
            "cov_both_walls_s": (own[:, 0] & own[:, 1]).float().mean().item(),
            "cov_both_walls_s2": (own[:, 2] & own[:, 3]).float().mean().item(),
            "size_discriminates": (own_all & ~oth_all).float().mean().item(),
            "size_ambiguous": (own_all & oth_all).float().mean().item(),
            "size_confident_error": (~own_all & oth_all).float().mean().item(),
        }
        results[thr_name] = res
        if thr_name == "ord":
            # Q3: on wall-induced losses at the ORIGINAL size (cells 0->1 / 1->0),
            # does a single-concept correction restore y1? (ordinary threshold)
            fails_a = torch.where(own[:, 0] & ~own[:, 1])[0]  # lost when wall A -> B
            fails_b = torch.where(own[:, 1] & ~own[:, 0])[0]
            correction = {}
            for corr_name, coord, use_full in (
                ("scale", 1, False), ("hue", 0, False), ("clamp", None, True)
            ):
                restored = []
                for src_cell, dst_cell, idxs in ((0, 1, fails_a), (1, 0, fails_b)):
                    for i in idxs.tolist():
                        x = renders[i, dst_cell].unsqueeze(0).to(device)
                        combo = combos[i]
                        if use_full:
                            c_gt = dataset.concept_vector(combo).unsqueeze(0).to(device)
                            mask = torch.ones(1, 4, dtype=torch.bool, device=device)
                            logits, _ = model(x, intervened_concepts=c_gt, intervention_mask=mask)
                        else:
                            logits = corrected_logits(
                                model, x, dataset.concept_vector(combo)[coord], coord, device
                            )
                        s = aps(logits, None, uniforms=u[i].reshape(1, 1))
                        restored.append(bool((s[0] <= q_ord)[y1[i]]))
                correction[corr_name] = float(np.mean(restored)) if restored else float("nan")
            correction["n_failures"] = int(len(fails_a) + len(fails_b))
            results["correction"] = correction

    return results


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    device = get_device(args.device or cfg.get("training", {}).get("device"))
    print(f"[*] Four-image 2x2 evaluation on device: {device}")

    rhos = [float(r) for r in args.rhos.split(",")]
    kinds = args.models.split(",")

    flat, table = {}, ["| model | rho | thr | own-cov | nuis A→B | nuis B→A | cov-both (s / s') |"
                       " size-disc | ambiguous | conf-err | scale-fix | hue-fix | clamp-fix | n_fail |",
                       "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
                       " :---: | :---: | :---: | :---: |"]
    for kind in kinds:
        for rho in rhos:
            res = run_cell(kind, rho, args, cfg, device)
            tag = f"{kind}_rho{rho:g}"
            for thr in ("ord", "paired"):
                r = res[thr]
                for k, v in r.items():
                    if isinstance(v, float):
                        flat[f"{tag}_{thr}_{k}"] = v
            c = res["correction"]
            flat[f"{tag}_ord_scale_fix"] = c["scale"]
            flat[f"{tag}_ord_hue_fix"] = c["hue"]
            flat[f"{tag}_ord_clamp_fix"] = c["clamp"]
            flat[f"{tag}_n_failures"] = c["n_failures"]
            for thr in ("ord", "paired"):
                r = res[thr]
                table.append(
                    f"| {kind} | {rho:g} | {thr} | {r['own_cover']:.3f} | {r['nuis_loss_A2B']:.3f} | "
                    f"{r['nuis_loss_B2A']:.3f} | {r['cov_both_walls_s']:.3f} / {r['cov_both_walls_s2']:.3f} | "
                    f"{r['size_discriminates']:.3f} | {r['size_ambiguous']:.3f} | {r['size_confident_error']:.3f} | "
                    + ("—" if thr == "paired" else
                       f"{c['scale']:.3f} | {c['hue']:.3f} | {c['clamp']:.3f} | {c['n_failures']} |")
                )
            print("\n".join(table[-1:]))

    print("\n" + "\n".join(table))
    if cfg.get("logging", {}).get("log_to_vault", True):
        vault_dir = PROJECT_ROOT / (args.vault_path or cfg.get("logging", {}).get("vault_path", "vault"))
        logger = VaultLogger(vault_path=vault_dir)
        log_file = logger.log_run(
            experiment_type="four-image",
            run_name="Four-Image 2x2 (wall x size)",
            config={"config": args.config, "rhos": rhos, "models": kinds,
                    "n_groups": args.n_groups, "epochs": cfg["training"].get("epochs")},
            metrics=flat,
            notes=(
                "2x2 = wall {A,B} x size across the bucket boundary; renders are real grid "
                "images; model/threshold/APS draw shared across each group's four renders. "
                "nuis = directional coverage loss at fixed size; size-disc = own label in AND "
                "other size out at fixed wall; *-fix = P[correct label returns | wall-induced "
                "loss] when replacing only that concept (hue = irrelevant control, clamp = "
                "all-concepts reference).\n\n" + "\n".join(table)
            ),
            tags=["clear", "four-image", "causal", "shapes3d", "correction"],
        )
        print(f"[*] Vault log written: {log_file}")


if __name__ == "__main__":
    main()
