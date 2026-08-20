"""
Generates training labels for crop stress.

IMPORTANT: In the absence of ground-truth field observations, agronomist
annotations, or yield records, this module produces PROXY labels derived
from a heuristic combination of vegetation index decline and environmental
anomalies. These are not verified ground truth. Replace with real field
observations before drawing operational conclusions from model output.
"""

import numpy as np
import pandas as pd

from . import config


def compute_proxy_labels(satellite_tensor, sensor_sequence, all_channels,
                          patch_coords,
                          ndvi_threshold=config.NDVI_DECLINE_THRESHOLD,
                          ndmi_threshold=config.NDMI_DECLINE_THRESHOLD,
                          soil_moisture_threshold=config.SOIL_MOISTURE_DROP_THRESHOLD,
                          temp_threshold=config.TEMP_SPIKE_THRESHOLD):
    """
    Label a patch as stressed (1) if NDVI declined significantly AND NDMI
    declined significantly AND a field-level environmental anomaly occurred
    between the first and last observation date.

    Returns a DataFrame with per-patch diagnostics and the binary label,
    plus a standalone numpy array of labels for direct model training use.
    """
    ndvi_idx = all_channels.index("NDVI")
    ndmi_idx = all_channels.index("NDMI")

    patch_ndvi_mean = satellite_tensor[:, :, :, :, ndvi_idx].mean(axis=(2, 3))
    patch_ndmi_mean = satellite_tensor[:, :, :, :, ndmi_idx].mean(axis=(2, 3))

    ndvi_change = patch_ndvi_mean[:, -1] - patch_ndvi_mean[:, 0]
    ndmi_change = patch_ndmi_mean[:, -1] - patch_ndmi_mean[:, 0]

    soil_moisture_change = sensor_sequence[-1, 0] - sensor_sequence[0, 0]
    temp_change = sensor_sequence[-1, 1] - sensor_sequence[0, 1]
    env_anomaly = bool(
        (soil_moisture_change < soil_moisture_threshold) or (temp_change > temp_threshold)
    )

    labels = (
        (ndvi_change < ndvi_threshold)
        & (ndmi_change < ndmi_threshold)
        & env_anomaly
    ).astype(int)

    labels_df = pd.DataFrame({
        "patch_idx": np.arange(len(labels)),
        "row_start": [c[0] for c in patch_coords],
        "col_start": [c[1] for c in patch_coords],
        "ndvi_change": ndvi_change,
        "ndmi_change": ndmi_change,
        "env_anomaly": env_anomaly,
        "stress_label": labels,
        "label_type": "PROXY_LABEL_NOT_GROUND_TRUTH",
    })

    return labels_df, labels
