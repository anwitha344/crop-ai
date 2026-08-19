"""
features.py
============
Spectral-index and temporal feature engineering on top of raw Sentinel-2 bands.
Assumes band order [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
matching src.preprocessing.SENTINEL2_BANDS.
"""

import numpy as np

BLUE, GREEN, RED, RE1, RE2, RE3, NIR, NIR2, SWIR1, SWIR2 = range(10)
EPS = 1e-8


def ndvi(scene):
    """Normalized Difference Vegetation Index. scene: (C, H, W) or (N, C, H, W)."""
    nir, red = scene[..., NIR, :, :], scene[..., RED, :, :]
    return (nir - red) / (nir + red + EPS)


def ndwi(scene):
    """Normalized Difference Water Index (McFeeters) — open water / surface moisture, uses Green & NIR."""
    green, nir = scene[..., GREEN, :, :], scene[..., NIR, :, :]
    return (green - nir) / (green + nir + EPS)


def evi(scene, G=2.5, C1=6.0, C2=7.5, L=1.0):
    """Enhanced Vegetation Index — more sensitive in high-biomass areas, corrects for atmosphere/soil."""
    nir, red, blue = scene[..., NIR, :, :], scene[..., RED, :, :], scene[..., BLUE, :, :]
    return G * (nir - red) / (nir + C1 * red - C2 * blue + L + EPS)


def savi(scene, L=0.5):
    """Soil-Adjusted Vegetation Index — reduces soil brightness influence for sparse canopy."""
    nir, red = scene[..., NIR, :, :], scene[..., RED, :, :]
    return ((nir - red) / (nir + red + L + EPS)) * (1 + L)


def ndmi(scene):
    """Normalized Difference Moisture Index (Gao) — canopy water content, uses NIR & SWIR1. Water-stress signal."""
    nir, swir1 = scene[..., NIR, :, :], scene[..., SWIR1, :, :]
    return (nir - swir1) / (nir + swir1 + EPS)


def ndre(scene):
    """Normalized Difference Red-Edge — chlorophyll/early-stage stress, more sensitive than NDVI at high biomass."""
    nir, re1 = scene[..., NIR, :, :], scene[..., RE1, :, :]
    return (nir - re1) / (nir + re1 + EPS)


INDEX_NAMES_15CH = ("NDVI", "NDMI", "NDRE", "NDWI", "SAVI")
INDEX_FNS_15CH = {"NDVI": ndvi, "NDMI": ndmi, "NDRE": ndre, "NDWI": ndwi, "SAVI": savi}


def stack_15_channels(scene):
    """
    scene: (10, H, W) raw Sentinel-2 bands in SENTINEL2_BANDS order.
    Returns (15, H, W): the 10 raw bands + [NDVI, NDMI, NDRE, NDWI, SAVI].
    NaNs from cloud masking propagate naturally through the index math.
    """
    idx_stack = np.stack([INDEX_FNS_15CH[name](scene) for name in INDEX_NAMES_15CH], axis=0)
    return np.concatenate([scene, idx_stack], axis=0)


def stack_indices(scene):
    """Backwards-compatible alias: raw bands + [NDVI, NDMI, NDRE, NDWI, SAVI]."""
    return stack_15_channels(scene)


def patch_summary_stats(patch):
    """
    Collapse a (C, H, W) patch into per-channel mean/std/min/max — useful for the
    baseline (non-deep) model in 03_baseline.ipynb.
    """
    c = patch.shape[0]
    flat = patch.reshape(c, -1)
    return np.concatenate([flat.mean(1), flat.std(1), flat.min(1), flat.max(1)])


def temporal_deltas(feature_sequence):
    """
    feature_sequence: (T, F) array of per-timestep features (e.g. mean NDVI per date).
    Returns (T, F) array of first differences (rate of change), zero-padded at t=0.
    Captures "is this field trending worse" which raw snapshots miss.
    """
    deltas = np.diff(feature_sequence, axis=0, prepend=feature_sequence[:1])
    return deltas


def build_feature_table(patches, index_names=("ndvi", "ndwi", "evi", "savi", "ndmi")):
    """
    Build a tabular feature matrix (N, F) from a batch of raw-band patches (N, C, H, W),
    for use by the RandomForest baseline. F = per-index mean/std over each patch.
    """
    fns = {"ndvi": ndvi, "ndwi": ndwi, "evi": evi, "savi": savi, "ndmi": ndmi}
    rows = []
    for patch in patches:
        feats = []
        for name in index_names:
            idx_map = fns[name](patch)  # (H, W)
            feats.extend([np.nanmean(idx_map), np.nanstd(idx_map)])
        rows.append(feats)
    cols = [f"{n}_{stat}" for n in index_names for stat in ("mean", "std")]
    return np.array(rows, dtype=np.float32), cols
