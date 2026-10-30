---
title: "nn_agreement_shapes3d"
experiment_type: nn-audit
date: 2026-10-29 22:06:17
tags: [conformal-cbm, experiment, nn-audit]
status: completed
nn_acc_latent: 0.525
nn_acc_concept: 0.55
leakage_gap: 0.0
leakage_ratio: 0.9545437190114199
nn_agreement_rate: 0.7
---

# Run Report: nn_agreement_shapes3d

> **Execution Date:** `2026-10-29 22:06:17` | **Type:** `nn-audit`

## 📊 Key Evaluation Metrics

| Metric | Value | Description |
| :--- | :--- | :--- |
| **nn_acc_latent** | `0.5250` | Observed Run Metric |
| **nn_acc_concept** | `0.5500` | Observed Run Metric |
| **leakage_gap** | `0.0000` | Observed Run Metric |
| **leakage_ratio** | `0.9545` | Observed Run Metric |
| **nn_agreement_rate** | `0.7000` | Observed Run Metric |

## ⚙️ Hyperparameters & Configuration

```yaml
conformal:
  alpha: 0.1
  alphas_audit:
  - 0.01
  - 0.05
  - 0.1
  - 0.2
  difficulty_mode: concept_entropy
  fill_empty: true
  score_type: aps
data:
  dataset: shapes3d
  image_size: 64
  num_classes: 4
  num_concepts: 6
  num_samples: 2000
experiment_name: clear_shapes3d_benchmark
interventions:
  max_budget: 5
  policies:
  - random
  - uncertainty
  - greedy
  - oracle
logging:
  log_to_vault: true
  save_dir: checkpoints/shapes3d
  vault_path: vault
model:
  backbone_type: conv
  hybrid_residual: false
  latent_dim: 128
  residual_dim: 16
seed: 42
training:
  batch_size: 64
  concept_loss_weight: 1.0
  device: null
  epochs: 10
  lr: 0.001
  task_loss_weight: 1.0
  weight_decay: 0.0001
```

## 📝 Researcher Notes

k-NN leakage audit evaluated with k=5. Leakage gap: 0.0000.


## 🔗 Related Notes
- [[00 - Dashboards/Experiment Tracker|Experiment Tracker]]
- [[00 - Dashboards/Audit Dashboard|Audit Dashboard]]
- [[03 - Experiments/Nn-Audit Benchmarks|Nn-Audit Benchmarks]]

