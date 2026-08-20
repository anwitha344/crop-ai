"""
Dataset preparation utilities: train/test splitting and tabular feature
engineering for the Random Forest baseline.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def split_patches(n_samples, test_size=config.TEST_SIZE_FRACTION, seed=config.RANDOM_SEED):
    """
    Split patch indices into train/test sets.

    NOTE: with very few patches from a single field, this split does not
    guarantee spatial independence between train and test data. For
    production use, split by field or by time range instead (see README).
    """
    indices = np.arange(n_samples)
    test_count = max(1, int(test_size * n_samples))
    train_idx, test_idx = train_test_split(indices, test_size=test_count, random_state=seed)
    return train_idx, test_idx


def build_tabular_features(X_sat, X_sen, all_channels,
                            sensor_feature_names=config.SENSOR_FEATURES):
    """
    Build a flat tabular feature table from spatiotemporal tensors, for use
    with classical ML models (e.g. Random Forest). For each channel and
    sensor variable, computes mean, std, and change-over-time per patch.
    """
    n_samples = X_sat.shape[0]
    rows = []

    for n in range(n_samples):
        row = {}
        for c_idx, ch_name in enumerate(all_channels):
            series = X_sat[n, :, :, :, c_idx].mean(axis=(1, 2))
            row[f"{ch_name}_mean"] = series.mean()
            row[f"{ch_name}_std"] = series.std()
            row[f"{ch_name}_change"] = series[-1] - series[0]

        for s_idx, s_name in enumerate(sensor_feature_names):
            series = X_sen[n, :, s_idx]
            row[f"sensor_{s_name}_mean"] = series.mean()
            row[f"sensor_{s_name}_change"] = series[-1] - series[0]

        rows.append(row)

    return pd.DataFrame(rows)


def flatten_time_for_cnn(X_sat, y):
    """
    Flatten the time dimension so each (date, patch) pair becomes an
    independent training sample for the spatial-only CNN baseline. Each
    date inherits the patch's overall label (a simplification -- early
    dates in a stressed sequence may not yet show visible stress).
    """
    n, t, h, w, c = X_sat.shape
    X_flat = X_sat.reshape(n * t, h, w, c)
    y_flat = np.repeat(y, t)
    return X_flat, y_flat
