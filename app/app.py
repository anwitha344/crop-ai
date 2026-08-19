"""
app.py
======
Minimal Streamlit demo that loads a trained CNNLSTMRiskModel checkpoint and
displays the stress/water/pest risk maps for a chosen field.

Run with:
    streamlit run app/app.py
"""

import os
import sys
import glob

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from models import CNNLSTMRiskModel
from dataset import load_processed_field
from preprocessing import load_sentinel2_scene
from inference import generate_field_risk_maps, plot_risk_maps

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
CKPT_PATH = os.path.join(PROCESSED_DIR, "checkpoints", "cnn_lstm_best.pt")

st.set_page_config(page_title="Crop Risk Map", layout="wide")
st.title("🌾 Crop Stress / Water / Pest Risk Map")


@st.cache_resource
def load_trained_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = CNNLSTMRiskModel(
        in_channels=ckpt["in_channels"],
        sensor_features=ckpt["sensor_features"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main():
    if not os.path.exists(CKPT_PATH):
        st.warning(f"No checkpoint found at {CKPT_PATH}. Train a model in "
                   f"`notebooks/05_cnn_lstm.ipynb` first.")
        return

    field_ids = sorted(os.listdir(PROCESSED_DIR)) if os.path.exists(PROCESSED_DIR) else []
    field_ids = [f for f in field_ids if f != "checkpoints"]
    if not field_ids:
        st.warning("No processed fields found under data/processed/.")
        return

    field_id = st.sidebar.selectbox("Field", field_ids)
    seq_len = st.sidebar.slider("Sequence length (timesteps)", 2, 12, 6)
    patch_size = st.sidebar.number_input("Patch size", value=64, step=8)

    with st.spinner("Loading model..."):
        model = load_trained_model(CKPT_PATH)

    with st.spinner("Loading processed data..."):
        dates, patches_by_date, sensors, coords = load_processed_field(field_id, PROCESSED_DIR)

    if len(dates) < seq_len:
        st.error(f"Field has only {len(dates)} dates, need at least {seq_len}.")
        return

    recent = range(len(dates) - seq_len, len(dates))
    n_locations = patches_by_date[0].shape[0]
    sensor_features = sensors.shape[1] if sensors is not None else 8

    image_seqs = np.stack(
        [np.stack([patches_by_date[t][loc] for t in recent]) for loc in range(n_locations)]
    ).astype(np.float32)

    sensor_seq = (np.stack([sensors[t] for t in recent]) if sensors is not None
                  else np.zeros((seq_len, sensor_features), dtype=np.float32))
    sensor_seqs = np.tile(sensor_seq, (n_locations, 1, 1)).astype(np.float32)

    scene_path = sorted(glob.glob(os.path.join(RAW_DIR, "sentinel2", field_id, "*.tif")))[-1]
    scene, _ = load_sentinel2_scene(scene_path)
    scene_shape = scene.shape[1:]

    with st.spinner("Running inference..."):
        risk_maps = generate_field_risk_maps(
            model, image_seqs, sensor_seqs, coords, scene_shape,
            patch_size=patch_size, device="cpu",
        )

    fig = plot_risk_maps(risk_maps)
    st.pyplot(fig)

    cols = st.columns(3)
    for col, (name, raster) in zip(cols, risk_maps.items()):
        col.metric(name.replace("_", " ").title(), f"{np.nanmean(raster):.2f}")


if __name__ == "__main__":
    main()
