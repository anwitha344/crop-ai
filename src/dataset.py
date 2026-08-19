"""
dataset.py
==========
PyTorch Dataset/DataLoader plumbing for the CNN -> MLP -> Fusion -> LSTM -> 3-head model.

Two datasets are provided:
  - PatchDataset        : single time-step (image patch, sensor vector) -> 3 risk labels.
                           Used by 04_cnn.ipynb (CNN-only baseline, no temporal modeling).
  - SequenceCropDataset : a sequence of T time-steps per field/patch location
                           (images[T,C,H,W], sensors[T,F]) -> 3 risk labels at the last step.
                           Used by 05_cnn_lstm.ipynb (full spatio-temporal model).

Both expect data already produced by src/preprocessing.py -> data/processed/<field_id>/.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """
    patches: (N, C, H, W) float32, already normalized to [0,1]
    sensors: (N, F) float32, already z-scored
    labels:  (N, 3) float32  -> [stress_risk, water_risk, pest_risk] in [0, 1]
    """

    def __init__(self, patches, sensors, labels, transform=None):
        assert len(patches) == len(sensors) == len(labels)
        self.patches = patches
        self.sensors = sensors
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, i):
        img = torch.from_numpy(self.patches[i]).float()
        if self.transform:
            img = self.transform(img)
        sensor = torch.from_numpy(self.sensors[i]).float()
        label = torch.from_numpy(self.labels[i]).float()
        return {"image": img, "sensor": sensor, "label": label}


class SequenceCropDataset(Dataset):
    """
    image_seqs:  (N, T, C, H, W) float32
    sensor_seqs: (N, T, F) float32
    labels:      (N, 3) float32   -> risk at the final timestep of each sequence
    seq_len:     T, fixed sequence length (pad/truncate upstream when building the array)
    """

    def __init__(self, image_seqs, sensor_seqs, labels, transform=None):
        assert len(image_seqs) == len(sensor_seqs) == len(labels)
        self.image_seqs = image_seqs
        self.sensor_seqs = sensor_seqs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_seqs)

    def __getitem__(self, i):
        imgs = torch.from_numpy(self.image_seqs[i]).float()      # (T, C, H, W)
        if self.transform:
            imgs = torch.stack([self.transform(imgs[t]) for t in range(imgs.shape[0])])
        sensors = torch.from_numpy(self.sensor_seqs[i]).float()  # (T, F)
        label = torch.from_numpy(self.labels[i]).float()         # (3,)
        return {"image_seq": imgs, "sensor_seq": sensors, "label": label}


def build_sequences(patches_by_date, sensors_by_date, labels, seq_len=6):
    """
    Turn per-date arrays into fixed-length sliding-window sequences per spatial location.

    patches_by_date: list of (N_locations, C, H, W) arrays, one per date, in chronological order
    sensors_by_date: list of (F,) arrays, one per date (broadcast to all locations that date)
    labels:          (N_locations, 3) array, risk labels at the *final* date in the window
    seq_len:         number of timesteps per training sequence

    Returns image_seqs (M, T, C, H, W), sensor_seqs (M, T, F), seq_labels (M, 3)
    where M = N_locations * max(1, n_dates - seq_len + 1)
    """
    n_dates = len(patches_by_date)
    n_locations = patches_by_date[0].shape[0]
    if n_dates < seq_len:
        raise ValueError(f"Need at least {seq_len} dates, got {n_dates}")

    image_seqs, sensor_seqs, seq_labels = [], [], []
    for start in range(0, n_dates - seq_len + 1):
        window = range(start, start + seq_len)
        for loc in range(n_locations):
            img_seq = np.stack([patches_by_date[t][loc] for t in window])       # (T, C, H, W)
            sensor_seq = np.stack([sensors_by_date[t] for t in window])          # (T, F)
            image_seqs.append(img_seq)
            sensor_seqs.append(sensor_seq)
            seq_labels.append(labels[loc])

    return (np.stack(image_seqs).astype(np.float32),
            np.stack(sensor_seqs).astype(np.float32),
            np.stack(seq_labels).astype(np.float32))


def train_val_test_split(n, val_frac=0.15, test_frac=0.15, seed=42):
    """Return shuffled index arrays (train_idx, val_idx, test_idx)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    val_idx = idx[:n_val]
    test_idx = idx[n_val:n_val + n_test]
    train_idx = idx[n_val + n_test:]
    return train_idx, val_idx, test_idx


def load_processed_field(field_id, processed_root="../data/processed"):
    """Convenience loader for everything build_processed_field() saved for one field."""
    field_dir = os.path.join(processed_root, field_id)
    with open(os.path.join(field_dir, "dates.txt")) as f:
        dates = f.read().splitlines()
    patches_by_date = [np.load(os.path.join(field_dir, f"patches_{d}.npy")) for d in dates]
    sensors = np.load(os.path.join(field_dir, "sensors.npy")) if os.path.exists(
        os.path.join(field_dir, "sensors.npy")) else None
    coords = np.load(os.path.join(field_dir, "patch_coords.npy"))
    return dates, patches_by_date, sensors, coords
