"""
inference.py
============
Load a trained CNNLSTMRiskModel checkpoint and turn a field's patch sequence +
sensor sequence into a spatial risk map (one raster per risk type), by predicting
per-patch risk and stitching results back to their (row, col) coordinates.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from models import CNNLSTMRiskModel, CNNRiskModel


def load_model(checkpoint_path, model_cls=CNNLSTMRiskModel, device="cpu", **model_kwargs):
    model = model_cls(**model_kwargs)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_patch_risks(model, image_seqs, sensor_seqs, device="cpu", batch_size=64):
    """
    image_seqs: (N, T, C, H, W) numpy array — N patch locations, T timesteps each
    sensor_seqs: (N, T, F) numpy array
    Returns dict of numpy arrays, each (N,): stress_risk, water_risk, pest_risk
    """
    model.eval()
    all_preds = {"stress_risk": [], "water_risk": [], "pest_risk": []}
    n = len(image_seqs)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        imgs = torch.from_numpy(image_seqs[start:end]).float().to(device)
        sensors = torch.from_numpy(sensor_seqs[start:end]).float().to(device)
        out = model(imgs, sensors)
        for k in all_preds:
            all_preds[k].append(out[k].cpu().numpy())
    return {k: np.concatenate(v) for k, v in all_preds.items()}


def stitch_risk_map(patch_risks, coords, scene_shape, patch_size):
    """
    patch_risks: (N,) risk score per patch
    coords: (N, 2) top-left (row, col) pixel coordinate of each patch, from preprocessing.extract_patches
    scene_shape: (H, W) of the full scene
    Returns an (H, W) risk raster, averaging overlapping patches and leaving
    unobserved pixels as NaN.
    """
    H, W = scene_shape
    risk_sum = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    for score, (row, col) in zip(patch_risks, coords):
        risk_sum[row:row + patch_size, col:col + patch_size] += score
        count[row:row + patch_size, col:col + patch_size] += 1

    with np.errstate(invalid="ignore"):
        risk_map = risk_sum / count
    risk_map[count == 0] = np.nan
    return risk_map


def generate_field_risk_maps(model, image_seqs, sensor_seqs, coords, scene_shape,
                              patch_size=64, device="cpu"):
    """
    Full pipeline: model predictions -> three stitched (H, W) risk maps.
    Returns dict: {"stress_risk": (H,W), "water_risk": (H,W), "pest_risk": (H,W)}
    """
    preds = predict_patch_risks(model, image_seqs, sensor_seqs, device=device)
    maps = {}
    for risk_type, scores in preds.items():
        maps[risk_type] = stitch_risk_map(scores, coords, scene_shape, patch_size)
    return maps


def plot_risk_maps(risk_maps, cmap="RdYlGn_r", figsize=(15, 5)):
    """Quick-look plot of the three risk rasters side by side."""
    fig, axes = plt.subplots(1, len(risk_maps), figsize=figsize)
    if len(risk_maps) == 1:
        axes = [axes]
    for ax, (name, raster) in zip(axes, risk_maps.items()):
        im = ax.imshow(raster, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(name.replace("_", " ").title())
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig
