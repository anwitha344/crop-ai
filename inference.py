"""
Reusable inference pipeline: given any field's coordinates and a date
range, fetches Sentinel-2 imagery, preprocesses it, runs the trained model,
and returns per-patch stress predictions plus a spatial risk grid.

This allows the trained model to be applied to fields it has never seen,
not only the field it was trained on.
"""

import ee
import numpy as np

from . import config
from .data_acquisition import define_aoi, get_sentinel2_collection
from .preprocessing import mask_clouds, resample_to_target_resolution, add_vegetation_indices
from .features import get_field_array, extract_patches
from .risk_map import build_risk_dataframe, build_risk_grid


def predict_crop_stress(
    lon_min, lat_min, lon_max, lat_max,
    start_date, end_date,
    model,
    sensor_stats=None,
    n_dates=config.N_DATES,
    patch_size=config.PATCH_SIZE,
    channels=config.ALL_CHANNELS,
    bands=config.BANDS_OF_INTEREST,
    max_cloud_pct=40,
):
    """
    Run the full pipeline on a new field and return predictions.

    Returns:
        risk_df: DataFrame of per-patch stress probability and risk level
        risk_grid: 2D numpy array for visualization
        dates_used: list of date strings actually used
    """
    field_aoi = define_aoi(lon_min, lat_min, lon_max, lat_max)

    collection = get_sentinel2_collection(field_aoi, start_date, end_date, max_cloud_pct)
    total_found = collection.size().getInfo()
    if total_found == 0:
        raise ValueError(
            "No Sentinel-2 images found for this field/date range. "
            "Try widening the date range or loosening the cloud filter."
        )

    img_list = collection.toList(total_found)
    seen_dates = set()
    unique_pairs = []
    import datetime
    for i in range(total_found):
        img = ee.Image(img_list.get(i))
        date_ms = img.get("system:time_start").getInfo()
        date_str = datetime.datetime.utcfromtimestamp(date_ms / 1000).strftime("%Y-%m-%d")
        if date_str not in seen_dates:
            unique_pairs.append((date_str, img))
            seen_dates.add(date_str)

    if len(unique_pairs) < n_dates:
        raise ValueError(
            f"Only found {len(unique_pairs)} unique dates, need at least {n_dates}. "
            "Widen the date range."
        )

    step = len(unique_pairs) / n_dates
    sampled = [unique_pairs[int(i * step)] for i in range(n_dates)]
    dates_used = [d for d, _ in sampled]

    processed_arrays = []
    for _, img in sampled:
        masked = mask_clouds(img)
        resampled = resample_to_target_resolution(masked, bands=bands)
        scaled = resampled.divide(config.REFLECTANCE_SCALE_FACTOR)
        scaled = ee.Image(scaled).copyProperties(img, ["system:time_start"])
        with_indices = add_vegetation_indices(ee.Image(scaled))
        clipped = with_indices.clip(field_aoi)

        arr = get_field_array(clipped, channels, field_aoi)
        processed_arrays.append(arr)

    all_dates_patches = []
    patch_coords = None
    for arr in processed_arrays:
        patches, coords = extract_patches(arr, patch_size)
        all_dates_patches.append(patches)
        patch_coords = coords

    if len(all_dates_patches[0]) == 0:
        raise ValueError("Field too small to extract any patches. Use a larger bounding box.")

    stacked = np.stack(all_dates_patches, axis=0)
    X_sat_new = np.transpose(stacked, (1, 0, 2, 3, 4))
    X_sat_new = np.nan_to_num(X_sat_new, nan=0.0)

    n_patches = X_sat_new.shape[0]

    # Sensor data: placeholder values for demo purposes when live sensors are unavailable.
    # Replace with a real sensor feed / weather API call in production.
    sim_sensor = np.tile(
        np.array([[28.0, 30.0, 60.0, 1.0]] * n_dates)[np.newaxis, :, :],
        (n_patches, 1, 1),
    )
    if sensor_stats is not None:
        mean = np.array(sensor_stats["mean"])
        std = np.array(sensor_stats["std"])
        sim_sensor_norm = (sim_sensor - mean) / std
    else:
        sim_sensor_norm = sim_sensor

    proba = model.predict(
        {"satellite_input": X_sat_new, "sensor_input": sim_sensor_norm}, verbose=0
    ).flatten()

    risk_df = build_risk_dataframe(proba, patch_coords)
    risk_grid = build_risk_grid(risk_df, patch_size)

    return risk_df, risk_grid, dates_used
