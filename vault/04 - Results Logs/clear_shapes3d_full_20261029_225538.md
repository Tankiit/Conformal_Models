---
title: "clear_shapes3d_full"
experiment_type: shapes3d
date: 2026-10-29 22:55:38
tags: [conformal-cbm, experiment, shapes3d]
status: completed
test_task_acc: 1.0
test_concept_acc: 0.9737534523010254
marginal_coverage: 1.0
target_coverage: 0.9
alpha: 0.1
avg_set_size: 1.0
concept_coverage: 0.8993802070617676
q_hat: 0.8992217779159546
---

# Run Report: clear_shapes3d_full

> **Execution Date:** `2026-10-29 22:55:38` | **Type:** `shapes3d`

## 📊 Key Evaluation Metrics

| Metric | Value | Description |
| :--- | :--- | :--- |
| **test_task_acc** | `1.0000` | Downstream Task Accuracy (Top-1) |
| **test_concept_acc** | `0.9738` | Concept Bottleneck Average Accuracy |
| **marginal_coverage** | `1.0000` | Empirical Target Coverage (1 - alpha) |
| **target_coverage** | `0.9000` | Observed Run Metric |
| **alpha** | `0.1000` | Significance Level (Error Rate) |
| **avg_set_size** | `1.0000` | Average Prediction Set Cardinality |
| **concept_coverage** | `0.8994` | Average Marginal Coverage across Concepts |
| **q_hat** | `0.8992` | Observed Run Metric |

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
  cache_file: /Users/cril/tanmoy/research/data/3dshapes_full_uint8.npz
  dataset: shapes3d
  image_size: 64
  num_classes: 4
  num_concepts: 6
  num_samples: 480000
experiment_name: clear_shapes3d_full
interventions:
  max_budget: 5
  policies:
  - random
  - uncertainty
  - greedy
  - oracle
logging:
  log_to_vault: true
  save_dir: checkpoints/shapes3d_full
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
  epochs: 5
  lr: 0.001
  task_loss_weight: 1.0
  weight_decay: 0.0001
```

## 📝 Researcher Notes

Trained 5 epochs on shapes3d with APS non-conformity score.


## 🔗 Related Notes
- [[00 - Dashboards/Experiment Tracker|Experiment Tracker]]
- [[00 - Dashboards/Audit Dashboard|Audit Dashboard]]
- [[03 - Experiments/Shapes3D Benchmarks|Shapes3D Benchmarks]]

