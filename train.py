"""
End-to-end training script for the crop stress prediction pipeline.

Run this after setting GEE_PROJECT_ID in src/config.py and authenticating
with `earthengine authenticate` (or via ee.Authenticate() interactively,
e.g. in a notebook, on first run).

Usage:
    python train.py
"""

import os

import ee
import numpy as np

from src import config
from src.data_acquisition import (
    initialize_earth_engine, define_aoi, get_sentinel2_collection,
    select_evenly_spaced_dates,
)
from src.preprocessing import preprocess_collection, clip_to_field
from src.features import (
    build_temporal_satellite_tensor, compute_band_statistics, save_json,
)
from src.sensor_data import (
    generate_prototype_sensor_data, synchronize_with_satellite_dates,
    build_sensor_tensor, compute_sensor_statistics, normalize_sensor_tensor,
)
from src.labeling import compute_proxy_labels
from src.dataset import split_patches, build_tabular_features, flatten_time_for_cnn
from src.models import build_cnn, build_full_model, compile_model
from src.evaluate import compute_classification_metrics, print_metrics
from src.risk_map import build_risk_dataframe, build_risk_grid
from src.visualize import plot_training_curves, plot_risk_map

from sklearn.ensemble import RandomForestClassifier


def main():
    os.makedirs(config.RAW_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    os.makedirs(config.LABELS_DIR, exist_ok=True)

    print("Initializing Earth Engine...")
    initialize_earth_engine()

    print("Defining field of interest and querying Sentinel-2...")
    aoi = define_aoi(**config.DEFAULT_AOI_BOUNDS)
    collection = get_sentinel2_collection(aoi, config.DEFAULT_START_DATE, config.DEFAULT_END_DATE)
    count = collection.size().getInfo()
    print(f"Found {count} candidate images")

    collection, dates_used = select_evenly_spaced_dates(collection, count, config.N_DATES)
    count = len(dates_used)
    print(f"Using dates: {dates_used}")

    print("Preprocessing (cloud mask, resample, scale, indices)...")
    collection = preprocess_collection(collection)
    collection = collection.map(lambda img: clip_to_field(img, aoi))

    print("Computing band normalization statistics...")
    band_stats = compute_band_statistics(collection, count, aoi)
    save_json(band_stats, config.BAND_STATS_PATH)

    print("Building satellite temporal tensor...")
    X_satellite, patch_coords, dates_used = build_temporal_satellite_tensor(
        collection, count, aoi
    )
    print(f"X_satellite shape: {X_satellite.shape}")
    n_patches = X_satellite.shape[0]

    print("Generating sensor data and synchronizing with satellite dates...")
    sensor_df = generate_prototype_sensor_data(dates_used)
    sensor_df.to_csv(f"{config.RAW_DIR}/sensor_data.csv", index=False)

    synced_sensor_df = synchronize_with_satellite_dates(sensor_df, dates_used)
    X_sensor, sensor_sequence = build_sensor_tensor(synced_sensor_df, n_patches)
    sensor_stats = compute_sensor_statistics(sensor_sequence)
    save_json(sensor_stats, config.SENSOR_STATS_PATH)
    X_sensor_norm = normalize_sensor_tensor(X_sensor, sensor_stats)

    print("Generating proxy labels...")
    labels_df, y_labels = compute_proxy_labels(
        X_satellite, sensor_sequence, config.ALL_CHANNELS, patch_coords
    )
    labels_df.to_csv(f"{config.LABELS_DIR}/proxy_labels.csv", index=False)
    print(f"Label distribution: {np.bincount(y_labels, minlength=2)}")

    print("Splitting train/test...")
    train_idx, test_idx = split_patches(n_patches)
    X_sat_train, X_sat_test = X_satellite[train_idx], X_satellite[test_idx]
    X_sen_train, X_sen_test = X_sensor_norm[train_idx], X_sensor_norm[test_idx]
    y_train, y_test = y_labels[train_idx], y_labels[test_idx]

    print("\n--- Random Forest baseline ---")
    X_tab_train = build_tabular_features(X_sat_train, X_sen_train, config.ALL_CHANNELS)
    X_tab_test = build_tabular_features(X_sat_test, X_sen_test, config.ALL_CHANNELS)

    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=config.RANDOM_SEED, class_weight="balanced"
    )
    rf_model.fit(X_tab_train, y_train)
    y_pred_rf = rf_model.predict(X_tab_test)
    y_proba_rf = (
        rf_model.predict_proba(X_tab_test)[:, 1] if len(np.unique(y_test)) > 1 else None
    )
    rf_metrics = compute_classification_metrics(y_test, y_pred_rf, y_proba_rf)
    print_metrics(rf_metrics, title="Random Forest Baseline")

    print("\n--- Spatial CNN baseline ---")
    X_cnn_train, y_cnn_train = flatten_time_for_cnn(X_sat_train, y_train)
    X_cnn_test, y_cnn_test = flatten_time_for_cnn(X_sat_test, y_test)

    cnn_model = build_cnn()
    compile_model(cnn_model)
    cnn_model.fit(
        X_cnn_train, y_cnn_train,
        validation_data=(X_cnn_test, y_cnn_test),
        epochs=config.CNN_EPOCHS,
        batch_size=config.CNN_BATCH_SIZE,
        verbose=1,
    )

    print("\n--- Full CNN-LSTM model ---")
    full_model = build_full_model(
        sensor_shape=(X_sen_train.shape[1], X_sen_train.shape[2]),
        n_timesteps=X_sat_train.shape[1],
    )
    compile_model(full_model)

    history = full_model.fit(
        x={"satellite_input": X_sat_train, "sensor_input": X_sen_train},
        y=y_train,
        validation_data=(
            {"satellite_input": X_sat_test, "sensor_input": X_sen_test},
            y_test,
        ),
        epochs=config.FULL_MODEL_EPOCHS,
        batch_size=config.FULL_MODEL_BATCH_SIZE,
        verbose=1,
    )

    fig = plot_training_curves(history)
    fig.savefig(f"{config.PROCESSED_DIR}/training_curves.png")

    print("\n--- Final evaluation ---")
    y_pred_proba = full_model.predict(
        {"satellite_input": X_sat_test, "sensor_input": X_sen_test}, verbose=0
    ).flatten()
    y_pred_class = (y_pred_proba > 0.5).astype(int)
    full_metrics = compute_classification_metrics(
        y_test, y_pred_class,
        y_pred_proba if len(np.unique(y_test)) > 1 else None,
    )
    print_metrics(full_metrics, title="CNN-LSTM Model - Test Set")

    full_model.save(config.MODEL_PATH)
    print(f"\nModel saved to {config.MODEL_PATH}")

    print("\n--- Generating field-wide risk map ---")
    y_all_proba = full_model.predict(
        {"satellite_input": X_satellite, "sensor_input": X_sensor_norm}, verbose=0
    ).flatten()
    risk_df = build_risk_dataframe(y_all_proba, patch_coords)
    risk_grid = build_risk_grid(risk_df)

    risk_df.to_csv(f"{config.PROCESSED_DIR}/risk_map.csv", index=False)
    np.save(f"{config.PROCESSED_DIR}/risk_grid.npy", risk_grid)

    fig = plot_risk_map(risk_grid)
    fig.savefig(f"{config.PROCESSED_DIR}/risk_map.png")

    print("\nDone. Outputs saved to:", config.PROCESSED_DIR)


if __name__ == "__main__":
    main()
