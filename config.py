"""
Central configuration for the crop stress prediction pipeline.
Edit values here rather than scattering magic numbers across modules.
"""

# --- Google Earth Engine ---
GEE_PROJECT_ID = "your-gee-project-id"  # replace with your GEE-linked cloud project

# --- Field of interest (default: Central Valley, CA farmland near Caruthers) ---
DEFAULT_AOI_BOUNDS = {
    "lon_min": -119.85,
    "lat_min": 36.55,
    "lon_max": -119.79,
    "lat_max": 36.60,
}

# --- Date range for training data ---
DEFAULT_START_DATE = "2024-04-01"
DEFAULT_END_DATE = "2024-09-30"
N_DATES = 6  # number of evenly-spaced dates to sample from the collection

# --- Sentinel-2 bands used ---
BANDS_OF_INTEREST = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
VEGETATION_INDICES = ["NDVI", "NDMI", "NDRE", "NDWI", "SAVI"]
ALL_CHANNELS = BANDS_OF_INTEREST + VEGETATION_INDICES  # 15 channels total

# --- Cloud filtering ---
MAX_CLOUDY_PIXEL_PERCENTAGE = 30

# --- SCL (Scene Classification Layer) values to mask out ---
# 3 = cloud shadow, 8 = cloud medium probability, 9 = cloud high probability,
# 10 = thin cirrus, 11 = snow
SCL_MASK_VALUES = [3, 8, 9, 10, 11]

# --- Reflectance scaling ---
REFLECTANCE_SCALE_FACTOR = 10000  # Sentinel-2 SR values are scaled by this factor

# --- Resolution ---
TARGET_RESOLUTION_M = 10
TARGET_CRS = "EPSG:4326"

# --- Patching ---
PATCH_SIZE = 64  # pixels; at 10m resolution this is 640m x 640m per patch

# --- Sensor feature names (order matters, must match model input) ---
SENSOR_FEATURES = ["soil_moisture", "temp", "humidity", "leaf_wetness"]

# --- Label thresholds for proxy stress labeling ---
NDVI_DECLINE_THRESHOLD = -0.1
NDMI_DECLINE_THRESHOLD = -0.1
SOIL_MOISTURE_DROP_THRESHOLD = -10
TEMP_SPIKE_THRESHOLD = 5

# --- Train/test split ---
TEST_SIZE_FRACTION = 0.25
RANDOM_SEED = 42

# --- Model training ---
CNN_EPOCHS = 20
CNN_BATCH_SIZE = 8
FULL_MODEL_EPOCHS = 30
FULL_MODEL_BATCH_SIZE = 2
LEARNING_RATE = 1e-3

# --- Risk map bins ---
RISK_BINS = [0, 0.33, 0.66, 1.0]
RISK_LABELS = ["LOW", "MED", "HIGH"]

# --- Paths ---
BASE_DIR = "crop_ai_data"
RAW_DIR = f"{BASE_DIR}/raw"
PROCESSED_DIR = f"{BASE_DIR}/processed"
LABELS_DIR = f"{BASE_DIR}/labels"
MODEL_PATH = f"{PROCESSED_DIR}/crop_stress_model.keras"
BAND_STATS_PATH = f"{PROCESSED_DIR}/band_stats.json"
SENSOR_STATS_PATH = f"{PROCESSED_DIR}/sensor_stats.json"
