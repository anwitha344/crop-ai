"""
Handles environmental sensor data: prototype/simulated data generation,
synchronization with satellite observation dates, and building the
temporal sensor tensor fed into the model's sensor branch.

NOTE: The prototype generator in this module produces SIMULATED data for
development purposes. Replace with real sensor readings (IoT devices,
weather station APIs, etc.) before using this system in production.
"""

import numpy as np
import pandas as pd

from . import config


def generate_prototype_sensor_data(dates, seed=config.RANDOM_SEED):
    """
    Generate a clearly-labeled simulated sensor dataset aligned to the given
    satellite observation dates. Do not present this as real sensor data.
    """
    rng = np.random.default_rng(seed)
    n = len(dates)

    df = pd.DataFrame({
        "date": dates,
        "soil_moisture": np.round(np.linspace(32, 13, n) + rng.normal(0, 1.5, n), 1),
        "temp": np.round(np.linspace(28, 36, n) + rng.normal(0, 0.8, n), 1),
        "humidity": np.round(np.linspace(61, 55, n) + rng.normal(0, 2, n), 1),
        "leaf_wetness": np.round(
            np.clip(np.linspace(1.2, 0.4, n) + rng.normal(0, 0.3, n), 0, None), 1
        ),
    })
    df["data_source"] = "SIMULATED_PROTOTYPE"
    return df


def synchronize_with_satellite_dates(sensor_df, satellite_dates, window_hours=72):
    """
    Align sensor readings to each satellite observation date. If sensor_df
    has an hourly 'timestamp' column, aggregates over a trailing window.
    Otherwise assumes one row per satellite date already (prototype case).
    """
    rows = []
    for date in satellite_dates:
        target_date = pd.to_datetime(date)

        if "timestamp" in sensor_df.columns:
            window_start = target_date - pd.Timedelta(hours=window_hours)
            window_data = sensor_df[
                (pd.to_datetime(sensor_df["timestamp"]) > window_start)
                & (pd.to_datetime(sensor_df["timestamp"]) <= target_date)
            ]
            row = {
                "date": date,
                "mean_soil_moisture": window_data["soil_moisture"].mean(),
                "mean_temp": window_data["temp"].mean(),
                "max_temp": window_data["temp"].max(),
                "mean_humidity": window_data["humidity"].mean(),
                "total_leaf_wetness": window_data["leaf_wetness"].sum(),
            }
        else:
            match = sensor_df[sensor_df["date"] == date]
            if match.empty:
                raise ValueError(f"No sensor data found for date {date}")
            r = match.iloc[0]
            row = {
                "date": date,
                "mean_soil_moisture": r["soil_moisture"],
                "mean_temp": r["temp"],
                "max_temp": r["temp"],
                "mean_humidity": r["humidity"],
                "total_leaf_wetness": r["leaf_wetness"],
            }
        rows.append(row)

    return pd.DataFrame(rows)


def build_sensor_tensor(synced_sensor_df, n_patches,
                         feature_cols=("mean_soil_moisture", "mean_temp",
                                       "mean_humidity", "total_leaf_wetness")):
    """
    Build the (N_patches, T_dates, n_features) sensor tensor. Sensor readings
    are field-level, so the same sequence is broadcast to every patch.
    """
    feature_cols = list(feature_cols)
    sequence = synced_sensor_df[feature_cols].values  # (T, F)
    tensor = np.tile(sequence[np.newaxis, :, :], (n_patches, 1, 1))
    return tensor, sequence


def compute_sensor_statistics(sequence, feature_names=config.SENSOR_FEATURES):
    """Compute mean/std per sensor feature for normalization."""
    mean = sequence.mean(axis=0)
    std = sequence.std(axis=0) + 1e-8
    return {"mean": mean.tolist(), "std": std.tolist(), "feature_order": feature_names}


def normalize_sensor_tensor(tensor, stats):
    """Apply saved mean/std normalization to a sensor tensor."""
    mean = np.array(stats["mean"])
    std = np.array(stats["std"])
    return (tensor - mean) / std
