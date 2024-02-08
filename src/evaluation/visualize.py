"""Visualization utilities for wafer maps and anomaly heatmaps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_wafer_map(
    wafer_map: np.ndarray,
    title: str = "",
    save_path: str | None = None,
) -> None:
    """Plot a single wafer map with color-coded dies."""
    cmap = plt.cm.colors.ListedColormap(["white", "#4ade80", "#ef4444"])
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(wafer_map, cmap=cmap, vmin=0, vmax=2)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_anomaly_heatmap(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    score: float,
    title: str = "",
    save_path: str | None = None,
) -> None:
    """Overlay anomaly heatmap on wafer image."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input Wafer Map")
    axes[0].axis("off")

    im = axes[1].imshow(anomaly_map, cmap="hot", interpolation="bilinear")
    axes[1].set_title(f"Anomaly Map (score: {score:.3f})")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(anomaly_map, cmap="hot", alpha=0.5, interpolation="bilinear")
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: str | None = None,
) -> None:
    """Plot confusion matrix with seaborn heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
