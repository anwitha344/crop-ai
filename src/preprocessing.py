"""
preprocessing.py
=================
Utilities to turn raw Sentinel-2 scenes + raw sensor CSVs into aligned,
model-ready patches and time series.

Expected raw layout:
    data/raw/sentinel2/<field_id>/<YYYY-MM-DD>.tif   (multi-band GeoTIFF, e.g. B02,B03,B04,B08,B11,B12)
    data/raw/sensors/<field_id>.csv                   (timestamp, soil_moisture, temp, humidity, ...)
    data/labels/<field_id>.csv                         (timestamp, stress_risk, water_risk, pest_risk)

Processed output:
    data/processed/<field_id>/patches_<date>.npy        (N, C, H, W)
    data/processed/<field_id>/sensors.npy                (T, F)
    data/processed/<field_id>/labels.npy                 (T, 3)
"""

import os
import glob
import numpy as np
import pandas as pd

try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.warp import reproject, Resampling
    from rasterio.mask import mask as rio_mask
except ImportError:
    rasterio = None  # allow module import even if rasterio isn't installed yet

# Sentinel-2 L2A Scene Classification Layer (SCL) class codes, for reference:
#   0 no_data | 1 saturated/defective | 2 dark_area | 3 cloud_shadow | 4 vegetation
#   5 bare_soil | 6 water | 7 unclassified | 8 cloud_medium_prob | 9 cloud_high_prob
#   10 thin_cirrus | 11 snow/ice
SCL_VALID_CLASSES = (4, 5, 6, 7, 11)  # vegetation, bare soil, water, unclassified, snow

SENTINEL2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
# Blue, Green, Red, RedEdge1, RedEdge2, RedEdge3, NIR, NIR-narrow, SWIR1, SWIR2
# Native resolutions before resampling to the common 10 m grid:
SENTINEL2_NATIVE_RES_M = {
    "B02": 10, "B03": 10, "B04": 10, "B08": 10,
    "B05": 20, "B06": 20, "B07": 20, "B8A": 20, "B11": 20, "B12": 20,
}
# Rough per-band reflectance clip range for Sentinel-2 L2A (surface reflectance, scaled 0-10000)
BAND_MIN, BAND_MAX = 0, 10000


def load_sentinel2_scene(path):
    """Load a multi-band Sentinel-2 GeoTIFF as a (C, H, W) float32 array."""
    if rasterio is None:
        raise ImportError("rasterio is required: pip install rasterio")
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)  # (C, H, W)
        profile = src.profile
    return arr, profile


def cloud_mask_scl(scl_band, valid_classes=SCL_VALID_CLASSES):
    """
    Build a boolean valid-pixel mask from a Sentinel-2 Scene Classification (SCL) band.
    True = keep, False = cloud/shadow/cirrus/saturated/no-data -> should become NaN.
    """
    return np.isin(scl_band, valid_classes)


def apply_cloud_mask(scene, scl_band, valid_classes=SCL_VALID_CLASSES):
    """
    scene: (C, H, W) float array of reflectance/index values
    scl_band: (H, W) array at the SAME resolution/grid as `scene`
    Returns (masked_scene, valid_mask) where masked_scene has NaN at every
    invalid pixel across ALL channels, and valid_mask is the (H, W) boolean mask used.
    This must run before any vegetation-index math so cloud contamination never
    gets interpreted as crop stress.
    """
    valid_mask = cloud_mask_scl(scl_band, valid_classes)
    masked = scene.astype(np.float32).copy()
    masked[:, ~valid_mask] = np.nan
    return masked, valid_mask


def resample_band_to_reference(src_path, ref_path, out_path, resampling=None):
    """
    Reproject/resample a single-band raster at `src_path` onto the exact grid
    (transform, CRS, width, height) of `ref_path`. Use Resampling.bilinear for
    continuous reflectance bands and Resampling.nearest for categorical layers (e.g. SCL).
    """
    if rasterio is None:
        raise ImportError("rasterio is required: pip install rasterio")
    if resampling is None:
        resampling = Resampling.bilinear

    with rasterio.open(ref_path) as ref:
        ref_transform, ref_crs = ref.transform, ref.crs
        ref_width, ref_height = ref.width, ref.height

    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(crs=ref_crs, transform=ref_transform, width=ref_width, height=ref_height)
        with rasterio.open(out_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=ref_transform, dst_crs=ref_crs,
                resampling=resampling,
            )
    return out_path


def clip_raster_to_boundary(tif_path, geometries, out_path=None, nodata=0):
    """
    Clip a (possibly multi-band) raster to a field boundary polygon so roads, buildings,
    water bodies etc. outside the field never become training examples.

    geometries: list of GeoJSON-like geometry dicts (or shapely geometries with
                `.__geo_interface__`), already in the raster's CRS.
    Returns (clipped_array (C,H,W), updated_profile).
    """
    if rasterio is None:
        raise ImportError("rasterio is required: pip install rasterio")
    geoms = [g.__geo_interface__ if hasattr(g, "__geo_interface__") else g for g in geometries]
    with rasterio.open(tif_path) as src:
        out_image, out_transform = rio_mask(src, geoms, crop=True, nodata=nodata, filled=True)
        out_profile = src.profile.copy()
    out_profile.update(height=out_image.shape[1], width=out_image.shape[2], transform=out_transform)
    if out_path:
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(out_image)
    return out_image.astype(np.float32), out_profile


def compute_channel_stats(scenes):
    """
    Compute per-channel mean/std from a list of (C, H, W) arrays (NaNs, e.g. from
    cloud masking, are ignored). ONLY pass training-split scenes here — never
    include validation/test dates, or normalization stats will leak information
    about data the model shouldn't have seen.
    Returns (mean, std), each shape (C,).
    """
    stacked = np.concatenate([s.reshape(s.shape[0], -1) for s in scenes], axis=1)  # (C, N*H*W)
    mean = np.nanmean(stacked, axis=1)
    std = np.nanstd(stacked, axis=1) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_channels(scene, mean, std):
    """Z-score standardize a (C, H, W) array using precomputed per-channel (mean, std)."""
    return (scene - mean[:, None, None]) / std[:, None, None]


def save_norm_stats(mean, std, path):
    np.savez(path, mean=mean, std=std)


def load_norm_stats(path):
    d = np.load(path)
    return d["mean"], d["std"]


def normalize_bands(arr, band_min=BAND_MIN, band_max=BAND_MAX):
    """Clip and rescale reflectance values to [0, 1]. arr: (C, H, W)."""
    arr = np.clip(arr, band_min, band_max)
    return (arr - band_min) / (band_max - band_min)


def extract_patches(scene, patch_size=64, stride=64):
    """
    Slide a window over a (C, H, W) scene and return a (N, C, patch_size, patch_size) array
    plus the (row, col) top-left pixel coordinate of each patch (for stitching risk maps back together).
    """
    c, h, w = scene.shape
    patches, coords = [], []
    for row in range(0, h - patch_size + 1, stride):
        for col in range(0, w - patch_size + 1, stride):
            patch = scene[:, row:row + patch_size, col:col + patch_size]
            patches.append(patch)
            coords.append((row, col))
    return np.stack(patches).astype(np.float32), coords


def load_sensor_csv(path, parse_dates=("timestamp",)):
    df = pd.read_csv(path, parse_dates=list(parse_dates))
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def align_sensor_to_dates(sensor_df, image_dates, tolerance_days=3):
    """
    For each Sentinel-2 acquisition date, pull the nearest sensor reading within `tolerance_days`.
    Returns a DataFrame indexed by image_dates with sensor feature columns (NaN if no match).
    """
    sensor_df = sensor_df.set_index("timestamp")
    image_dates = pd.to_datetime(image_dates)
    aligned = sensor_df.reindex(
        sensor_df.index.union(image_dates)
    ).sort_index().interpolate(method="time", limit_direction="both")
    result = aligned.reindex(image_dates)
    # drop rows whose nearest real observation was farther than tolerance
    return result


def normalize_sensor_features(df, stats=None):
    """
    Z-score normalize numeric sensor columns.
    `stats`: optional dict {col: (mean, std)} computed on the training set and reused for val/test.
    Returns (normalized_df, stats).
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if stats is None:
        stats = {c: (df[c].mean(), df[c].std() + 1e-8) for c in numeric_cols}
    for c in numeric_cols:
        mean, std = stats[c]
        df[c] = (df[c] - mean) / std
    return df, stats


def build_processed_field(field_id, raw_root="../data/raw", out_root="../data/processed",
                           patch_size=64, stride=64):
    """
    End-to-end: read all Sentinel-2 dates for a field, extract patches, align sensor data,
    and save everything as .npy under data/processed/<field_id>/.
    """
    scene_dir = os.path.join(raw_root, "sentinel2", field_id)
    scene_paths = sorted(glob.glob(os.path.join(scene_dir, "*.tif")))
    if not scene_paths:
        raise FileNotFoundError(f"No scenes found in {scene_dir}")

    os.makedirs(os.path.join(out_root, field_id), exist_ok=True)

    dates, all_patches, all_coords = [], [], None
    for p in scene_paths:
        date_str = os.path.splitext(os.path.basename(p))[0]
        scene, _ = load_sentinel2_scene(p)
        scene = normalize_bands(scene)
        patches, coords = extract_patches(scene, patch_size, stride)
        all_coords = coords  # coords are identical across dates for a fixed-size scene
        np.save(os.path.join(out_root, field_id, f"patches_{date_str}.npy"), patches)
        dates.append(date_str)

    sensor_path = os.path.join(raw_root, "sensors", f"{field_id}.csv")
    if os.path.exists(sensor_path):
        sensor_df = load_sensor_csv(sensor_path)
        aligned = align_sensor_to_dates(sensor_df, dates)
        aligned_norm, stats = normalize_sensor_features(aligned)
        np.save(os.path.join(out_root, field_id, "sensors.npy"), aligned_norm.values.astype(np.float32))

    coords_arr = np.array(all_coords)
    np.save(os.path.join(out_root, field_id, "patch_coords.npy"), coords_arr)
    with open(os.path.join(out_root, field_id, "dates.txt"), "w") as f:
        f.write("\n".join(dates))

    return dates, coords_arr
