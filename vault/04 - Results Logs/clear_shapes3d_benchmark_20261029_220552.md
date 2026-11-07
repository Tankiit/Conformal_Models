---
title: "clear_shapes3d_benchmark"
experiment_type: shapes3d
date: 2026-10-29 22:05:52
tags: [conformal-cbm, experiment, shapes3d]
status: superseded
test_task_acc: 0.4424999952316284
test_concept_acc: 0.768750011920929
marginal_coverage: 0.8799999952316284
target_coverage: 0.9
alpha: 0.1
avg_set_size: 2.3450000286102295
concept_coverage: 0.8975000381469727
q_hat: 0.750403881072998
---

# Run Report: clear_shapes3d_benchmark

> [!WARNING] Synthetic pilot — superseded (tagged 2026-11-07)
> Small procedural-generator run, not the real 3D Shapes data; superseded by
> [[04 - Results Logs/clear_shapes3d_full_20261029_225538|clear_shapes3d_full]]
> (and by the saturation verdict in [[06 - Papers & Planning/Results Snapshot 2026-11-06|Results Snapshot]]).


> **Execution Date:** `2026-10-29 22:05:52` | **Type:** `shapes3d`

## 📊 Key Evaluation Metrics

| Metric | Value | Description |
| :--- | :--- | :--- |
| **test_task_acc** | `0.4425` | Downstream Task Accuracy (Top-1) |
| **test_concept_acc** | `0.7688` | Concept Bottleneck Average Accuracy |
| **marginal_coverage** | `0.8800` | Empirical Target Coverage (1 - alpha) |
| **target_coverage** | `0.9000` | Observed Run Metric |
| **alpha** | `0.1000` | Significance Level (Error Rate) |
| **avg_set_size** | `2.3450` | Average Prediction Set Cardinality |
| **concept_coverage** | `0.8975` | Average Marginal Coverage across Concepts |
| **q_hat** | `0.7504` | Observed Run Metric |

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

Trained 10 epochs on shapes3d with APS non-conformity score.


## 🔗 Related Notes
- [[00 - Dashboards/Experiment Tracker|Experiment Tracker]]
- [[00 - Dashboards/Audit Dashboard|Audit Dashboard]]
- [[03 - Experiments/Shapes3D Benchmarks|Shapes3D Benchmarks]]

