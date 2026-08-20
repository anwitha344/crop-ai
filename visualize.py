"""
Plotting utilities for exploratory data analysis, training curves, feature
importance, and the final spatial risk map.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_index_time_series(dates, mean_ndvi, mean_ndmi):
    """Plot mean NDVI and NDMI across dates for a field (basic EDA)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, mean_ndvi, marker="o", label="Mean NDVI")
    ax.plot(dates, mean_ndmi, marker="s", label="Mean NDMI")
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Index value")
    plt.title("Mean NDVI / NDMI over time")
    plt.legend()
    plt.tight_layout()
    return fig


def plot_patch_ndvi_over_time(X_satellite, patch_idx, ndvi_channel_idx, dates):
    """Visualize one patch's NDVI channel across all observation dates."""
    t_steps = X_satellite.shape[1]
    fig, axes = plt.subplots(1, t_steps, figsize=(3 * t_steps, 3), squeeze=False)
    axes = axes.flatten()
    for t in range(t_steps):
        ax = axes[t]
        ax.imshow(X_satellite[patch_idx, t, :, :, ndvi_channel_idx], cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_title(dates[t])
        ax.axis("off")
    plt.suptitle(f"NDVI over time - Patch {patch_idx}")
    plt.tight_layout()
    return fig


def plot_training_curves(history):
    """Plot loss, accuracy, and AUC curves from a Keras training history object."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    if "auc" in history.history:
        axes[2].plot(history.history["auc"], label="train")
        axes[2].plot(history.history["val_auc"], label="val")
        axes[2].set_title("AUC")
        axes[2].legend()

    plt.tight_layout()
    return fig


def plot_feature_importance(importances, top_n=15):
    """Plot a horizontal bar chart of Random Forest feature importances."""
    fig = plt.figure(figsize=(10, 6))
    importances.head(top_n).plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title("Random Forest Feature Importances")
    plt.tight_layout()
    return fig


def plot_risk_map(risk_grid, title="Crop Stress Risk Map"):
    """Plot the spatial risk grid as a color-coded heatmap with probability labels."""
    fig = plt.figure(figsize=(8, 8))
    im = plt.imshow(risk_grid, cmap="RdYlGn_r", vmin=0, vmax=1)
    plt.colorbar(im, label="Stress Probability")
    plt.title(title)

    for r in range(risk_grid.shape[0]):
        for c in range(risk_grid.shape[1]):
            if not np.isnan(risk_grid[r, c]):
                plt.text(c, r, f"{risk_grid[r, c]:.2f}", ha="center", va="center", color="black")

    plt.tight_layout()
    return fig
