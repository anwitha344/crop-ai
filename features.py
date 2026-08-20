"""
Converts Earth Engine images into numpy arrays, splits fields into spatial
patches, and computes/applies normalization statistics.
"""

import json

import ee
import geemap
import numpy as np

from . import config


def get_band_array(image, band, region, scale=config.TARGET_RESOLUTION_M):
    """Fetch a single band from an Earth Engine image as a numpy array."""
    band_img = image.select(band)
    return geemap.ee_to_numpy(band_img, region=region, bands=[band], scale=scale)


def get_field_array(image, channels, region, scale=config.TARGET_RESOLUTION_M):
    """Fetch all specified channels for one image, stacked as (H, W, C)."""
    arrays = []
    for channel in channels:
        arr = get_band_array(image, channel, region, scale=scale)
        if arr is None:
            raise ValueError(f"Could not fetch channel: {channel}")
        arrays.append(arr[:, :, 0])
    return np.stack(arrays, axis=-1)


def extract_patches(field_array, patch_size=config.PATCH_SIZE):
    """
    Split a (H, W, C) field array into non-overlapping (patch_size, patch_size, C)
    patches. Returns the patches array and a list of (row_start, col_start)
    coordinates for mapping predictions back to geography later.
    """
    h, w, _ = field_array.shape
    patches = []
    coords = []

    for row in range(0, h - patch_size + 1, patch_size):
        for col in range(0, w - patch_size + 1, patch_size):
            patch = field_array[row:row + patch_size, col:col + patch_size, :]
            patches.append(patch)
            coords.append((row, col))

    return np.array(patches), coords


def build_temporal_satellite_tensor(collection, count, field_polygon,
                                     channels=config.ALL_CHANNELS,
                                     patch_size=config.PATCH_SIZE):
    """
    Fetch every date in the collection, patch it, and stack into a temporal
    tensor of shape (N_patches, T_dates, patch_size, patch_size, C_channels).
    Also returns the patch coordinates and the list of date strings used.
    """
    import datetime

    image_list = collection.toList(count)
    all_dates_patches = []
    date_strings = []
    coords = None

    for i in range(count):
        img = ee.Image(image_list.get(i))
        date_ms = img.get("system:time_start").getInfo()
        date_str = datetime.datetime.utcfromtimestamp(date_ms / 1000).strftime("%Y-%m-%d")

        field_arr = get_field_array(img, channels, field_polygon)
        patches, coords = extract_patches(field_arr, patch_size)

        all_dates_patches.append(patches)
        date_strings.append(date_str)

    stacked = np.stack(all_dates_patches, axis=0)         # (T, N, H, W, C)
    tensor = np.transpose(stacked, (1, 0, 2, 3, 4))        # (N, T, H, W, C)
    tensor = np.nan_to_num(tensor, nan=0.0)                # cloud-masked pixels -> 0

    return tensor, coords, date_strings


def compute_band_statistics(collection, count, region, bands=config.BANDS_OF_INTEREST):
    """
    Compute per-band mean/std across all images in the collection.
    IMPORTANT: only ever compute these from training data, never validation/test data.
    """
    stats = {}
    image_list = collection.toList(count)

    for band in bands:
        all_values = []
        for i in range(count):
            img = ee.Image(image_list.get(i))
            arr = get_band_array(img, band, region)
            if arr is not None:
                all_values.append(arr.flatten())
        if all_values:
            combined = np.concatenate(all_values)
            combined = combined[~np.isnan(combined)]
            stats[band] = {"mean": float(np.mean(combined)), "std": float(np.std(combined))}

    return stats


def standardize_array(arr, band_name, stats):
    """Apply (x - mean) / std standardization using precomputed statistics."""
    mean = stats[band_name]["mean"]
    std = stats[band_name]["std"]
    return (arr - mean) / (std + 1e-8)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)
