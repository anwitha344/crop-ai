"""
Model architectures: a standalone spatial CNN baseline, and the full
CNN + Sensor-MLP + LSTM fusion model used for spatiotemporal crop stress
prediction.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from . import config


def build_cnn(input_shape=(config.PATCH_SIZE, config.PATCH_SIZE, len(config.ALL_CHANNELS))):
    """
    Standalone spatial-only CNN. Used as an initial baseline to verify the
    network can learn spatial/spectral stress patterns before adding the
    temporal (LSTM) component.
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation="relu", name="cnn_features")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs, outputs, name="spatial_cnn_baseline")


def build_cnn_encoder_block(img_shape):
    """Returns a Sequential CNN block used as the shared per-timestep encoder in the full model."""
    return models.Sequential([
        layers.Conv2D(32, 3, padding="same", input_shape=img_shape),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same"),
        layers.ReLU(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
    ], name="cnn_encoder")


def build_full_model(
    img_shape=(config.PATCH_SIZE, config.PATCH_SIZE, len(config.ALL_CHANNELS)),
    sensor_shape=(config.N_DATES, len(config.SENSOR_FEATURES)),
    n_timesteps=config.N_DATES,
):
    """
    Full end-to-end CNN + Sensor-MLP + LSTM fusion model.

    Satellite branch: a shared CNN encoder is applied to each date's image
    (via TimeDistributed), producing a 128-dim feature vector per date.

    Sensor branch: a small MLP is applied to each date's sensor reading,
    producing a 64-dim feature vector per date.

    These are concatenated per-timestep into 192-dim vectors and fed to an
    LSTM, which outputs a single stress probability for the patch.
    """
    sat_input = layers.Input(shape=(n_timesteps, *img_shape), name="satellite_input")
    cnn_base = build_cnn_encoder_block(img_shape)
    sat_features = layers.TimeDistributed(cnn_base)(sat_input)  # (T, 128)

    sensor_input = layers.Input(shape=sensor_shape, name="sensor_input")
    sensor_features = layers.TimeDistributed(layers.Dense(32, activation="relu"))(sensor_input)
    sensor_features = layers.TimeDistributed(layers.Dense(64, activation="relu"))(sensor_features)  # (T, 64)

    fused = layers.Concatenate(axis=-1)([sat_features, sensor_features])  # (T, 192)

    x = layers.LSTM(128)(fused)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation="sigmoid", name="stress_probability")(x)

    model = models.Model(
        inputs=[sat_input, sensor_input],
        outputs=output,
        name="crop_stress_cnn_lstm",
    )
    return model


def compile_model(model, learning_rate=config.LEARNING_RATE):
    """Compile a model with the standard loss/metrics used throughout this project."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model
