# Crop AI

Spatiotemporal multimodal crop stress and risk prediction from Sentinel-2 satellite
imagery and environmental sensor data.

The system pulls a time series of Sentinel-2 images for a field, combines them with
soil/weather sensor readings, and trains a model to predict which patches of the
field are under crop stress. Predictions are turned into a spatial risk map that can
be applied to fields the model has never seen before.

## How It Works

At a high level, the pipeline has five stages: acquire imagery, preprocess it,
build patch-level features, label the patches, and train/evaluate a model. A
separate inference path reuses the same preprocessing and feature logic to score
new fields with an already-trained model.

```
Google Earth Engine (Sentinel-2 L2A)
          |
          v
  data_acquisition.py   -> query collection, pick evenly-spaced dates
          |
          v
  preprocessing.py      -> cloud mask, resample, scale, vegetation indices
          |
          v
  features.py            -> image -> numpy, split field into patches,
                             build temporal tensor (patches x dates x H x W x channels)
          |
          v
  sensor_data.py          -> environmental readings synced to the same dates,
                              tiled per patch (patches x dates x sensor features)
          |
          v
  labeling.py              -> proxy stress labels from vegetation-index decline
          |                    + environmental anomaly (no ground truth yet)
          v
  dataset.py + models.py   -> train/test split, Random Forest baseline,
                              spatial CNN baseline, full CNN + MLP + LSTM model
          |
          v
  evaluate.py + risk_map.py + visualize.py  -> metrics, per-patch risk map, plots
```

`inference.py` runs the acquisition/preprocessing/feature steps above against a new
bounding box and date range, then feeds the result through an already-trained model
to produce a risk map for that field.

## Modules

- **config.py** - Single place for all tunable settings: the Earth Engine project
  ID, area of interest, date range, Sentinel-2 bands and vegetation indices used,
  cloud-filtering thresholds, patch size, label thresholds, train/test split,
  training hyperparameters, and output paths.

- **data_acquisition.py** - Authenticates with Google Earth Engine, builds an area
  of interest geometry, queries the `COPERNICUS/S2_SR_HARMONIZED` collection for a
  date range with a cloud-cover filter, and selects a fixed number of evenly-spaced
  observation dates from the results.

- **preprocessing.py** - Per-image preprocessing chain: mask cloud/shadow/cirrus/
  snow pixels using the Scene Classification Layer, resample bands to a common
  resolution, scale raw digital numbers to reflectance, and compute vegetation
  indices (NDVI, NDMI, NDRE, NDWI, SAVI) as additional bands.

- **features.py** - Converts Earth Engine images into numpy arrays, splits a field
  into fixed-size non-overlapping patches while tracking each patch's spatial
  coordinates, stacks patches across dates into a temporal tensor, and computes/
  applies per-band normalization statistics.

- **sensor_data.py** - Generates a clearly-labeled simulated environmental sensor
  dataset (soil moisture, temperature, humidity, leaf wetness) for development,
  synchronizes sensor readings to the satellite observation dates, and builds the
  per-patch sensor tensor plus its normalization statistics. Meant to be replaced
  with real sensor or weather API data in production.

- **labeling.py** - Generates proxy stress labels: a patch is labeled stressed if
  NDVI and NDMI both decline beyond a threshold between the first and last
  observation date, together with a soil-moisture drop or temperature spike. These
  are heuristic proxies, not verified ground truth.

- **dataset.py** - Splits patches into train/test sets, builds a flat tabular
  feature table (mean/std/change per channel and sensor variable) for the Random
  Forest baseline, and flattens the time dimension so each date becomes an
  independent sample for the spatial-only CNN baseline.

- **models.py** - Defines the model architectures:
  - A standalone spatial CNN baseline (Conv2D blocks -> global average pooling ->
    dense -> sigmoid) used to sanity-check that the network can learn spatial and
    spectral stress patterns before adding a temporal component.
  - The full model: a shared CNN encoder applied per date via `TimeDistributed` to
    the satellite tensor, a small per-date MLP applied to the sensor tensor, the
    two feature vectors concatenated per timestep, and an LSTM over the resulting
    sequence producing a single stress probability per patch.

- **evaluate.py** - Computes accuracy, precision, recall, F1, and ROC-AUC (when
  both classes are present) and prints them in a consistent format.

- **risk_map.py** - Converts per-patch stress probabilities into risk categories
  (LOW/MED/HIGH) and reassembles per-patch predictions into a 2D grid matching the
  field's spatial layout for visualization.

- **visualize.py** - Plotting helpers: NDVI/NDMI time series, per-patch NDVI over
  time, training curves (loss/accuracy/AUC), Random Forest feature importances, and
  the final spatial risk map heatmap.

- **inference.py** - Applies an already-trained model to a new field: runs the same
  acquisition, cloud masking, resampling, scaling, and vegetation-index steps, cuts
  the field into patches, substitutes placeholder sensor values (or real ones if
  supplied), runs the model, and returns a risk dataframe, a risk grid, and the
  dates used.

- **train.py** - End-to-end training script that wires all of the above together:
  queries and preprocesses imagery, builds satellite and sensor tensors, generates
  proxy labels, splits train/test, trains the Random Forest baseline, the CNN
  baseline, and the full CNN-LSTM model, evaluates each, saves the trained model,
  and writes out a field-wide risk map.

## Model Architecture

The full model takes two inputs per patch:

- `satellite_input`: shape `(T, patch_size, patch_size, C)`, where `T` is the
  number of observation dates and `C` is the number of spectral bands plus
  vegetation indices (15 by default).
- `sensor_input`: shape `(T, F)`, where `F` is the number of sensor features
  (soil moisture, temperature, humidity, leaf wetness).

Each date's image is passed through a shared CNN encoder (three Conv2D blocks with
batch normalization, max pooling, and global average pooling, ending in a 128-dim
dense layer). Each date's sensor reading is passed through a small two-layer MLP
producing a 64-dim vector. The two per-date vectors are concatenated into a
192-dim vector per timestep, and the resulting sequence is fed into an LSTM(128)
followed by a dense layer and a sigmoid output, giving one stress probability per
patch.

## Setup

1. Install dependencies:

   ```
   pip install earthengine-api geemap numpy pandas scikit-learn tensorflow matplotlib
   ```

   (There is no `requirements.txt` in the repository yet; the above covers every
   import used across the modules.)

2. Set your Google Earth Engine project ID in `config.py`:

   ```python
   GEE_PROJECT_ID = "your-gee-project-id"
   ```

3. Authenticate with Earth Engine once, either by running
   `earthengine authenticate` on the command line, or by calling
   `ee.Authenticate()` interactively (for example in a notebook) before running
   the pipeline.

## Usage

Train the full pipeline on the default field and date range defined in
`config.py`:

```
python train.py
```

This will:

- query and preprocess Sentinel-2 imagery for the configured area of interest,
- build the satellite and sensor tensors and generate proxy labels,
- train and evaluate the Random Forest baseline, the spatial CNN baseline, and
  the full CNN-LSTM model,
- save the trained model to `crop_ai_data/processed/crop_stress_model.keras`,
- write training curves, a risk map image, a risk map CSV, and normalization
  statistics into `crop_ai_data/processed/`.

To score a new field with an already-trained model, use `predict_crop_stress` from
`inference.py`, passing a bounding box, a date range, and the loaded model:

```python
from tensorflow.keras.models import load_model
from inference import predict_crop_stress
from features import load_json
import config

model = load_model(config.MODEL_PATH)
sensor_stats = load_json(config.SENSOR_STATS_PATH)

risk_df, risk_grid, dates_used = predict_crop_stress(
    lon_min=-119.85, lat_min=36.55, lon_max=-119.79, lat_max=36.60,
    start_date="2024-04-01", end_date="2024-09-30",
    model=model,
    sensor_stats=sensor_stats,
)
```

## Project Structure

```
crop-ai/
  config.py
  data_acquisition.py
  preprocessing.py
  features.py
  sensor_data.py
  labeling.py
  dataset.py
  models.py
  evaluate.py
  risk_map.py
  visualize.py
  inference.py
  train.py
```

Note: the modules use relative imports (`from . import config`) and `train.py`
imports them as `from src import config`, `from src.data_acquisition import ...`,
etc. As currently committed, the files sit at the repository root rather than in a
`src/` package, so `train.py` will need either the files moved into a `src/`
directory with an `__init__.py`, or its imports updated to match the flat layout,
before it will run as-is.

## Known Limitations

- **Labels are proxies, not ground truth.** `labeling.py` derives stress labels
  from vegetation-index decline and environmental anomalies in the absence of
  field observations, agronomist annotations, or yield records. Treat model output
  as exploratory until validated against real observations.
- **Sensor data is simulated.** `sensor_data.py` generates synthetic soil
  moisture/temperature/humidity/leaf-wetness readings for development. Replace
  this with a real IoT feed or weather API before production use.
- **Train/test split is not spatially or temporally independent** when working
  with patches from a single field, as noted in `dataset.py`. For a more rigorous
  evaluation, split by field or by time range instead.
