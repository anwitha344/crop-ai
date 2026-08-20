"""
Builds a spatial crop stress risk map from per-patch model predictions.
"""

import numpy as np
import pandas as pd

from . import config


def build_risk_dataframe(probabilities, patch_coords,
                          bins=config.RISK_BINS, labels=config.RISK_LABELS):
    """Assemble a DataFrame of per-patch stress probabilities and risk categories."""
    df = pd.DataFrame({
        "patch_idx": np.arange(len(probabilities)),
        "row_start": [c[0] for c in patch_coords],
        "col_start": [c[1] for c in patch_coords],
        "stress_probability": probabilities,
    })
    df["risk_level"] = pd.cut(
        df["stress_probability"], bins=bins, labels=labels, include_lowest=True
    )
    return df


def build_risk_grid(risk_df, patch_size=config.PATCH_SIZE):
    """
    Reassemble per-patch predictions into a 2D grid matching the field's
    spatial layout, for visualization as a heatmap.
    """
    max_row = risk_df["row_start"].max() + patch_size
    max_col = risk_df["col_start"].max() + patch_size
    n_rows = max_row // patch_size
    n_cols = max_col // patch_size

    grid = np.full((n_rows, n_cols), np.nan)
    for _, row in risk_df.iterrows():
        r = row["row_start"] // patch_size
        c = row["col_start"] // patch_size
        grid[r, c] = row["stress_probability"]

    return grid
