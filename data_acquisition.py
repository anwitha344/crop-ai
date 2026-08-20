"""
Handles all interaction with Google Earth Engine: authentication, querying
the Sentinel-2 collection, and selecting a set of well-spaced dates.
"""

import datetime
import ee

from . import config


def initialize_earth_engine(project_id=config.GEE_PROJECT_ID):
    """Authenticate and initialize the Earth Engine API. Run once per session."""
    ee.Authenticate()
    ee.Initialize(project=project_id)


def define_aoi(lon_min, lat_min, lon_max, lat_max):
    """Build an Earth Engine rectangle geometry from a bounding box."""
    return ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])


def get_sentinel2_collection(aoi, start_date, end_date,
                              max_cloud_pct=config.MAX_CLOUDY_PIXEL_PERCENTAGE):
    """Query the Sentinel-2 Level-2A (surface reflectance) collection for an AOI/date range."""
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .sort("system:time_start")
    )
    return collection


def list_available_dates(collection):
    """Return a list of human-readable date strings for every image in the collection."""
    count = collection.size().getInfo()
    dates = collection.aggregate_array("system:time_start").getInfo()
    readable = [
        datetime.datetime.utcfromtimestamp(d / 1000).strftime("%Y-%m-%d")
        for d in dates
    ]
    return readable, count


def select_evenly_spaced_dates(collection, count, n_dates=config.N_DATES):
    """
    From a collection, select n_dates unique images spaced as evenly as
    possible across the available date range. Returns a new ImageCollection.
    """
    img_list = collection.toList(count)
    seen_dates = set()
    unique_pairs = []

    for i in range(count):
        img = ee.Image(img_list.get(i))
        date_ms = img.get("system:time_start").getInfo()
        date_str = datetime.datetime.utcfromtimestamp(date_ms / 1000).strftime("%Y-%m-%d")
        if date_str not in seen_dates:
            unique_pairs.append((date_str, img))
            seen_dates.add(date_str)

    if len(unique_pairs) < n_dates:
        raise ValueError(
            f"Only found {len(unique_pairs)} unique dates, need at least {n_dates}. "
            "Widen the date range or loosen the cloud filter."
        )

    step = len(unique_pairs) / n_dates
    sampled = [unique_pairs[int(i * step)] for i in range(n_dates)]

    dates_used = [d for d, _ in sampled]
    images = [img for _, img in sampled]
    new_collection = ee.ImageCollection.fromImages(images).sort("system:time_start")

    return new_collection, dates_used
