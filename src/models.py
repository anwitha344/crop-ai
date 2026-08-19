"""
models.py
=========
Implements the architecture from the design diagram:

    Sentinel-2 patches -> CNN -> spatial/spectral features ---\
                                                                 Feature Fusion -> LSTM -> [Stress, Water, Pest] heads
    Sensor data -> MLP -> sensor features ----------------------/

Two model classes are exported:
  - CNNRiskModel      : CNN + MLP + Fusion + 3 heads, NO temporal component.
                         Used for the 04_cnn.ipynb single-timestep baseline.
  - CNNLSTMRiskModel   : full pipeline above, consumes (B, T, C, H, W) image sequences and
                         (B, T, F) sensor sequences. Used in 05_cnn_lstm.ipynb.
"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """Extracts spatial/spectral features from a Sentinel-2 patch."""

    def __init__(self, in_channels=6, out_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # -> (B, 128, 1, 1)
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        h = self.features(x).flatten(1)  # (B, 128)
        return self.proj(h)              # (B, out_dim)


class SensorMLP(nn.Module):
    """Processes numeric ground-sensor readings (soil moisture, temp, humidity, etc.)."""

    def __init__(self, in_features, out_dim=32, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class FeatureFusion(nn.Module):
    """Concatenates CNN + MLP features and projects to a shared fused representation."""

    def __init__(self, img_dim=128, sensor_dim=32, fused_dim=128):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(img_dim + sensor_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, img_feat, sensor_feat):
        return self.fuse(torch.cat([img_feat, sensor_feat], dim=-1))


class RiskHeads(nn.Module):
    """Three parallel output heads: Stress Risk, Water Risk, Pest Risk (each in [0,1])."""

    def __init__(self, in_dim=64):
        super().__init__()
        def head():
            return nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(inplace=True), nn.Linear(32, 1), nn.Sigmoid())
        self.stress_head = head()
        self.water_head = head()
        self.pest_head = head()

    def forward(self, x):
        return {
            "stress_risk": self.stress_head(x).squeeze(-1),
            "water_risk": self.water_head(x).squeeze(-1),
            "pest_risk": self.pest_head(x).squeeze(-1),
        }


class CNNRiskModel(nn.Module):
    """CNN + MLP + Fusion + risk heads, single timestep (no LSTM). Baseline for 04_cnn.ipynb."""

    def __init__(self, in_channels=6, sensor_features=8, img_dim=128, sensor_dim=32, fused_dim=128):
        super().__init__()
        self.cnn = CNNEncoder(in_channels, img_dim)
        self.mlp = SensorMLP(sensor_features, sensor_dim)
        self.fusion = FeatureFusion(img_dim, sensor_dim, fused_dim)
        self.heads = RiskHeads(fused_dim)

    def forward(self, image, sensor):
        img_feat = self.cnn(image)
        sensor_feat = self.mlp(sensor)
        fused = self.fusion(img_feat, sensor_feat)
        return self.heads(fused)


class CNNLSTMRiskModel(nn.Module):
    """
    Full architecture: per-timestep CNN+MLP+Fusion features are fed into an LSTM
    to model temporal trends, and the final hidden state drives the 3 risk heads.
    """

    def __init__(self, in_channels=6, sensor_features=8, img_dim=128, sensor_dim=32,
                 fused_dim=128, lstm_hidden=64, lstm_layers=1, bidirectional=False):
        super().__init__()
        self.cnn = CNNEncoder(in_channels, img_dim)
        self.mlp = SensorMLP(sensor_features, sensor_dim)
        self.fusion = FeatureFusion(img_dim, sensor_dim, fused_dim)
        self.lstm = nn.LSTM(
            input_size=fused_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        head_in = lstm_hidden * (2 if bidirectional else 1)
        self.heads = RiskHeads(head_in)

    def forward(self, image_seq, sensor_seq):
        # image_seq: (B, T, C, H, W), sensor_seq: (B, T, F)
        B, T, C, H, W = image_seq.shape
        imgs_flat = image_seq.reshape(B * T, C, H, W)
        img_feat = self.cnn(imgs_flat).reshape(B, T, -1)          # (B, T, img_dim)

        sensors_flat = sensor_seq.reshape(B * T, -1)
        sensor_feat = self.mlp(sensors_flat).reshape(B, T, -1)    # (B, T, sensor_dim)

        fused = self.fusion(img_feat, sensor_feat)                # (B, T, fused_dim)

        lstm_out, (h_n, c_n) = self.lstm(fused)                   # lstm_out: (B, T, hidden)
        last_hidden = lstm_out[:, -1, :]                          # use final timestep's hidden state

        return self.heads(last_hidden)


class MultiTaskRiskLoss(nn.Module):
    """
    Weighted sum of BCE losses across the three risk heads (each treated as a
    probability in [0,1], e.g. from expert-labeled or thresholded ground truth).
    Swap nn.BCELoss for nn.MSELoss if labels are continuous risk scores instead of probabilities.
    """

    def __init__(self, weights=(1.0, 1.0, 1.0)):
        super().__init__()
        self.w_stress, self.w_water, self.w_pest = weights
        self.loss_fn = nn.BCELoss()

    def forward(self, preds, targets):
        # targets: (B, 3) -> columns [stress_risk, water_risk, pest_risk]
        loss_stress = self.loss_fn(preds["stress_risk"], targets[:, 0])
        loss_water = self.loss_fn(preds["water_risk"], targets[:, 1])
        loss_pest = self.loss_fn(preds["pest_risk"], targets[:, 2])
        total = self.w_stress * loss_stress + self.w_water * loss_water + self.w_pest * loss_pest
        return total, {"stress": loss_stress.item(), "water": loss_water.item(), "pest": loss_pest.item()}


if __name__ == "__main__":
    # quick shape sanity check
    B, T, C, H, W, F = 4, 6, 6, 64, 64, 8
    model = CNNLSTMRiskModel(in_channels=C, sensor_features=F)
    imgs = torch.randn(B, T, C, H, W)
    sensors = torch.randn(B, T, F)
    out = model(imgs, sensors)
    for k, v in out.items():
        print(k, v.shape)
