"""
Utility functions and Obsidian Vault logging for Conformal Concept Models.
"""

from __future__ import annotations
import os
import random
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Returns the best available torch device or the preferred one."""
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def save_checkpoint(
    state: Dict[str, Any],
    filepath: Union[str, Path],
) -> None:
    """Saves model checkpoint."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    filepath: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Loads model checkpoint."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")
    return torch.load(path, map_location=device or "cpu")


class VaultLogger:
    """
    Obsidian Vault Logger that automatically formats and syncs experiment results,
    metrics, conformal coverage tables, and markdown notes directly to an Obsidian vault.
    """

    def __init__(self, vault_path: Optional[Union[str, Path]] = None):
        if vault_path is None:
            # Default vault directory inside project root or relative
            vault_path = Path(__file__).resolve().parents[2] / "vault"
        self.vault_path = Path(vault_path)
        self._ensure_vault_structure()

    def _ensure_vault_structure(self) -> None:
        """Ensures the basic directories exist in the vault."""
        subdirs = [
            ".obsidian",
            "00 - Dashboards",
            "01 - Literature & Theory",
            "02 - Architecture & System",
            "03 - Experiments",
            "04 - Results Logs",
            "05 - Templates",
        ]
        for sd in subdirs:
            (self.vault_path / sd).mkdir(parents=True, exist_ok=True)

    def log_run(
        self,
        experiment_type: str,
        run_name: str,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        notes: str = "",
        tags: Optional[List[str]] = None,
    ) -> Path:
        """
        Creates a timestamped Markdown run log note with YAML frontmatter in 04 - Results Logs.
        Also updates the dashboard experiment tracker table.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date_slug = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = f"{run_name}_{date_slug}".replace(" ", "_")
        log_file = self.vault_path / "04 - Results Logs" / f"{safe_name}.md"

        tag_list = tags or ["conformal-cbm", "experiment"]
        if experiment_type not in tag_list:
            tag_list.append(experiment_type)

        frontmatter = [
            "---",
            f"title: \"{run_name}\"",
            f"experiment_type: {experiment_type}",
            f"date: {timestamp}",
            f"tags: [{', '.join(tag_list)}]",
            "status: completed",
        ]
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool)):
                frontmatter.append(f"{k}: {v}")
        frontmatter.append("---\n")

        # Build Markdown content
        content = [
            "\n".join(frontmatter),
            f"# Run Report: {run_name}",
            f"\n> **Execution Date:** `{timestamp}` | **Type:** `{experiment_type}`\n",
            "## 📊 Key Evaluation Metrics\n",
            "| Metric | Value | Description |",
            "| :--- | :--- | :--- |",
        ]

        metric_descriptions = {
            "test_task_acc": "Downstream Task Accuracy (Top-1)",
            "test_concept_acc": "Concept Bottleneck Average Accuracy",
            "marginal_coverage": "Empirical Target Coverage (1 - alpha)",
            "alpha": "Significance Level (Error Rate)",
            "avg_set_size": "Average Prediction Set Cardinality",
            "empty_set_rate": "Fraction of Empty Prediction Sets",
            "concept_coverage": "Average Marginal Coverage across Concepts",
            "nn_agreement": "Nearest-Neighbor Concept vs Latent Agreement",
            "intervention_auc": "Accuracy Area Under Curve across Intervention Steps",
            "leakage_score": "Residual Latent Concept Leakage Ratio",
        }

        for k, v in metrics.items():
            desc = metric_descriptions.get(k, "Observed Run Metric")
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            content.append(f"| **{k}** | `{val_str}` | {desc} |")

        content.extend([
            "\n## ⚙️ Hyperparameters & Configuration\n",
            "```yaml",
            yaml.dump(config, default_flow_style=False).strip(),
            "```\n",
        ])

        if notes:
            content.extend([
                "## 📝 Researcher Notes\n",
                notes.strip(),
                "\n",
            ])

        content.extend([
            "## 🔗 Related Notes",
            "- [[00 - Dashboards/Experiment Tracker|Experiment Tracker]]",
            "- [[00 - Dashboards/Audit Dashboard|Audit Dashboard]]",
            f"- [[03 - Experiments/{experiment_type.title()} Benchmarks|{experiment_type.title()} Benchmarks]]",
            "\n",
        ])

        with open(log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(content))

        # Update Master Tracker
        self._append_to_tracker(
            run_name=run_name,
            file_link=f"[[04 - Results Logs/{safe_name}|{run_name}]]",
            experiment_type=experiment_type,
            timestamp=timestamp,
            metrics=metrics,
        )

        return log_file

    def _append_to_tracker(
        self,
        run_name: str,
        file_link: str,
        experiment_type: str,
        timestamp: str,
        metrics: Dict[str, Any],
    ) -> None:
        """Appends a row to the 00 - Dashboards/Experiment Tracker.md."""
        tracker_path = self.vault_path / "00 - Dashboards" / "Experiment Tracker.md"
        if not tracker_path.exists():
            return

        with open(tracker_path, "r", encoding="utf-8") as f:
            tracker_content = f.read()

        cov = metrics.get("marginal_coverage", metrics.get("coverage", "-"))
        cov_str = f"{cov:.3f}" if isinstance(cov, float) else str(cov)
        size = metrics.get("avg_set_size", metrics.get("set_size", "-"))
        size_str = f"{size:.2f}" if isinstance(size, float) else str(size)
        task_acc = metrics.get("test_task_acc", metrics.get("task_acc", "-"))
        task_acc_str = f"{task_acc:.3f}" if isinstance(task_acc, float) else str(task_acc)
        c_acc = metrics.get("test_concept_acc", metrics.get("concept_acc", "-"))
        c_acc_str = f"{c_acc:.3f}" if isinstance(c_acc, float) else str(c_acc)

        new_row = f"| {timestamp[:10]} | {file_link} | `{experiment_type}` | {task_acc_str} | {c_acc_str} | {cov_str} | {size_str} |\n"

        # Look for table marker
        marker = "<!-- RUNS_TABLE_BODY -->"
        if marker in tracker_content:
            updated = tracker_content.replace(marker, marker + "\n" + new_row)
            with open(tracker_path, "w", encoding="utf-8") as f:
                f.write(updated)
        else:
            with open(tracker_path, "a", encoding="utf-8") as f:
                f.write(new_row)
