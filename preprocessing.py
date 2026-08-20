"""
Preprocessing operations applied to raw Sentinel-2 imagery before it becomes
model input: resampling to a common resolution, cloud masking, reflectance
scaling, and vegetation index computation.
"""

import ee

from . import config


def resample_to_target_resolution(image, bands=config.BANDS_OF_INTEREST,
                                   crs=config.TARGET_CRS,
                                   scale=config.TARGET_RESOLUTION_M):
    """Resample selected bands to a common spatial resolution using bilinear interpolation."""
    return (
        image.select(bands)
        .resample("bilinear")
        .reproject(crs=crs, scale=scale)
    )


def mask_clouds(image, scl_mask_values=config.SCL_MASK_VALUES):
    """
    Mask out cloud, cloud shadow, cirrus, and snow pixels using the
    Scene Classification Layer (SCL) band.
    """
    scl = image.select("SCL")
    mask = ee.Image(1)
    for value in scl_mask_values:
        mask = mask.And(scl.neq(value))
    return image.updateMask(mask)


def scale_reflectance(image, bands=config.BANDS_OF_INTEREST,
                       factor=config.REFLECTANCE_SCALE_FACTOR):
    """Convert raw integer digital numbers to physical reflectance values (0-1 range)."""
    scaled = image.select(bands).divide(factor)
    return scaled.copyProperties(image, ["system:time_start"])


def add_vegetation_indices(image):
    """
    Compute NDVI, NDMI, NDRE, NDWI, and SAVI and add them as new bands.
    Assumes the image already has reflectance-scaled bands B2-B12.
    """
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndmi = image.normalizedDifference(["B8", "B11"]).rename("NDMI")
    ndre = image.normalizedDifference(["B8", "B5"]).rename("NDRE")
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")

    L = 0.5  # SAVI soil brightness correction factor
    savi = image.expression(
        "((NIR - RED) / (NIR + RED + L)) * (1 + L)",
        {"NIR": image.select("B8"), "RED": image.select("B4"), "L": L},
    ).rename("SAVI")

    return image.addBands([ndvi, ndmi, ndre, ndwi, savi])


def preprocess_image(image):
    """Full preprocessing chain for a single image: mask clouds, resample, scale, add indices."""
    masked = mask_clouds(image)
    resampled = resample_to_target_resolution(masked)
    scaled = scale_reflectance(resampled)
    scaled = ee.Image(scaled)
    with_indices = add_vegetation_indices(scaled)
    return with_indices


def preprocess_collection(collection):
    """Apply the full preprocessing chain to every image in a collection."""
    return collection.map(preprocess_image)


def clip_to_field(image, field_polygon):
    """Clip an image to the field boundary, excluding surrounding non-field land."""
    return image.clip(field_polygon)
