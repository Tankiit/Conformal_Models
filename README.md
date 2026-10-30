# Conformal Concept Models (CCM)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()
[![Obsidian](https://img.shields.io/badge/Obsidian-Vault%20Integrated-purple.svg)]()

A comprehensive research framework for **distribution-free uncertainty quantification, conformal prediction, and reliability auditing** in **Concept Bottleneck Models (CBMs)**.

---

## 🌟 Key Features

- **Distribution-Free Finite-Sample Guarantees**:
  Guarantees target coverage $P(Y \in C(X)) \ge 1 - \alpha$ for downstream tasks and joint hyper-box bounds on concept bottlenecks.
- **Multiple Nonconformity Scores**:
  Adaptive Prediction Sets (**APS**), Regularized Adaptive Prediction Sets (**RAPS**), Softmax/Hinge, Margin, and Difficulty-Normalized nonconformity scores.
- **CLEAR Reliability & Leakage Audit Engine**:
  Audits multi-alpha calibration, difficulty-stratified subgroups (Mondrian calibration), concept fidelity, and latent information leakage.
- **Budgeted Concept Interventions**:
  Supports test-time human/oracle interventions with multiple policies:
  - **Random Policy**
  - **Uncertainty Entropy Policy**
  - **Greedy Conformal-Guided Policy**
  - **Oracle Optimal Policy**
- **Integrated Obsidian / Wiki Research Vault**:
  Automated two-way tracking: every training, audit, and pilot script execution writes rich Markdown reports, updates master tracker tables, and populates dashboards in `vault/`.

---

## 📁 Repository Structure

```
conformal-concept-models/
├── README.md                          # Main project documentation
├── pyproject.toml                     # Package installation metadata
├── pytest.ini                         # Pytest configuration
├── requirements.txt                   # Dependencies
├── configs/
│   ├── clear_shapes3d.yaml            # Shapes3D audit benchmark config
│   └── cvpr_cub.yaml                  # CUB-200-2011 intervention pilot config
├── src/ccm/
│   ├── __init__.py                    # Public API exports
│   ├── conformal.py                   # ConformalCalibrator & MondrianCalibrator
│   ├── scores.py                      # Nonconformity score functions (APS, RAPS, Softmax)
│   ├── sets.py                        # PredictionSet & ConceptPredictionSet containers
│   ├── metrics.py                     # Coverage, set size, AUC, NN-leakage metrics
│   ├── interventions.py               # Intervention policies & evaluation curves
│   ├── utils.py                       # Checkpointing, reproducibility & VaultLogger
│   ├── data/
│   │   ├── __init__.py
│   │   ├── shapes3d.py                # 3D Shapes loader and procedural generator
│   │   └── cub.py                     # CUB-200-2011 bird attributes benchmark loader
│   └── models/
│       ├── __init__.py
│       ├── cbm.py                     # Concept Bottleneck Models (Joint & Sequential)
│       └── difficulty.py              # Sample difficulty estimator & Mondrian strata
├── scripts/
│   ├── clear_train.py                 # Train CBM and calibrate conformal sets
│   ├── clear_audit.py                 # CLEAR multi-alpha & subgroup audit
│   ├── cvpr_pilot.py                  # Budgeted concept intervention pilot
│   └── nn_agreement.py                # Nearest-Neighbor concept leakage diagnostic
├── tests/
│   ├── test_conformal.py              # Unit tests for conformal algorithms
│   └── test_sets.py                   # Unit tests for prediction set operations
└── vault/                             # Obsidian Research & Results Vault
    ├── .obsidian/                     # Pre-configured Obsidian workspace settings
    ├── Home.md                        # Master Wiki & Dashboard Map of Content
    ├── 00 - Dashboards/               # Master Tracker & Audit Dashboards
    ├── 01 - Literature & Theory/      # Mathematical proofs & CBM theory
    ├── 02 - Architecture & System/    # System design & score derivations
    ├── 03 - Experiments/              # Benchmark definitions and audit scorecards
    ├── 04 - Results Logs/             # Automated script execution run logs
    └── 05 - Templates/                # Markdown templates for new experiments
```

---

## ⚡ Quick Start

### 1. Environment Setup

Activate the conda environment:
```bash
conda activate torch-multimodal
cd /Users/cril/tanmoy/research/Concept_Bottleneck_Models/Conformal_Models/conformal-concept-models
pip install -e .
```

### 2. Run the Unit Test Suite

```bash
pytest tests/ -v
```

All 11 unit tests verify mathematical bounds, RAPS size compression, finite-sample quantiles, and set containment.

---

## 🏃 Running Experiments & Auto-Updating Vault

### A. Training & Conformal Calibration (CLEAR Benchmark)
Trains the CBM on Shapes3D, saves the checkpoint, calibrates split conformal sets at 90% confidence ($\alpha=0.10$), and logs results to `vault/04 - Results Logs/`:
```bash
python scripts/clear_train.py --config configs/clear_shapes3d.yaml --epochs 5
```

### B. CLEAR Reliability & Subgroup Audit
Performs a multi-alpha audit ($\alpha \in \{0.01, 0.05, 0.10, 0.20\}$) and evaluates coverage stratified across Easy, Medium, and Hard difficulty tiers:
```bash
python scripts/clear_audit.py --config configs/clear_shapes3d.yaml
```
*Generated Report:* `vault/03 - Experiments/CLEAR_Audit_Shapes3D.md`

### C. CVPR Budgeted Intervention Pilot
Benchmarks concept intervention policies (Random, Uncertainty, Greedy Conformal, Oracle) on fine-grained attributes, evaluating accuracy gains and set compression up to budget $k=6$:
```bash
python scripts/cvpr_pilot.py --config configs/cvpr_cub.yaml --budget 6 --epochs 4
```
*Generated Report:* `vault/03 - Experiments/CVPR_Pilot_Cub.md`

### D. Nearest-Neighbor Concept Leakage Diagnostic
Audits whether latent representations $z$ leak task-relevant shortcut signals not captured by concepts $c$:
```bash
python scripts/nn_agreement.py --config configs/clear_shapes3d.yaml
```
*Generated Report:* `vault/03 - Experiments/NN_Agreement_Audit_Shapes3d.md`

---

## 📓 Using the Obsidian / Wiki Vault

1. Open the [Obsidian app](https://obsidian.md/).
2. Select **"Open folder as vault"** and point to:
   ```
   /Users/cril/tanmoy/research/Concept_Bottleneck_Models/Conformal_Models/conformal-concept-models/vault
   ```
3. The vault comes pre-configured with:
   - Dark theme & accent color (`.obsidian/appearance.json`).
   - Interactive `Home.md` central wiki.
   - Live updating table in `00 - Dashboards/Experiment Tracker.md`.
   - Graph view linking literature notes, architectures, and experiment runs.

---

## 📜 Citation & References

- Koh, P. W., Nguyen, T., Tang, Y. S., et al. (2020). *Concept Bottleneck Models*. ICML.
- Angelopoulos, A. N., & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*.
- Romano, Y., Sesia, M., & Candès, E. (2020). *Classification with Valid and Adaptive Prediction Sets (APS)*. NeurIPS.
- Angelopoulos, A. N., Bates, S., Jordan, M. I., & Malik, J. (2021). *Uncertainty Sets for Image Classifiers using Regularized Adaptive Prediction Sets (RAPS)*. ICLR.
