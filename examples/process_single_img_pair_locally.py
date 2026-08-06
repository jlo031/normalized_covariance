# ---- This is <process_single_img_pair_locally.py> ----

"""
Fully process a single image pair:
    - List all images in GEOTIFF_DIR
    - Select first two images as img_pair
    - Pre-process img_pair: normcovar_utils.check_and_trim_image_pair
    - Comnpute deviation from local mean, local std, normCoVar: normcovar.fully_process_single_image_pair

You need to define your main data directory ('DATA_DIR') and a test site ('site').
The code expects a 'SITE_DIR' in your main data directore ('DATA_SIR/site').
The originally processed geotiff files from GA must be stored in a 'GEOTIFF_DIR' called 'GA_geotiffs' within that 'SITE_DIR'.
This is the same file structure that is expected for the batch processing code.
This script will select the first two tif files as an image pair and trigger the processing chain.
All output files will be written to a newly created 'IMG_PAIR_DIR', which is defined based on image time stamps.
NB: In case GA changes their naming convention (or if you work with different tif files), you must adjust the date extraction and naming of the IMG_PAIR_DIR.
""" 

import pathlib
from loguru import logger
import sys
import shutil

import numpy as np

from osgeo import gdal

from normalized_covariance import normcovar, normcovar_utils

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# Define loglevel ["DEBUG" or "INFO"]
loglevel = "DEBUG"
##loglevel = "INFO"

logger.remove()
logger.add(sys.stderr, level=loglevel)

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# Define the main data dir
DATA_DIR = pathlib.Path("/g/data/jk72/jl0818/DATA/fast_ice_tests")

# Define your current test site
site = "normcovar_test__Atka"

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# Build path site-specific data dir
SITE_DIR =  DATA_DIR / f"{site}"

# Build full path to intensity geotiff dir
GEOTIFF_DIR = SITE_DIR / "GA_geotiffs"

logger.debug(f"DATA_DIR:    {DATA_DIR}")
logger.debug(f"SITE_DIR:    {SITE_DIR}")
logger.debug(f"GEOTIFF_DIR: {GEOTIFF_DIR}")

if not GEOTIFF_DIR.is_dir():
    logger.error(f"Could not find GEOTIFF_DIR: {GEOTIFF_DIR}")

# Find all S1 tif files in GEOTIFF_DIR
img_list = [ f.name for f in GEOTIFF_DIR.iterdir() if f.name.startswith("S1") and f.name.endswith("tif") ]

if not any(img_list):
    logger.error("No 'S1*tif' files found in GEOTIFF_DIR")
    pair = False
    sys.exit()

# select one image pair to test functions
if len(img_list) == 1:
    logger.warning("Only found one image in GEOTIFF_DIR, cannot test funtions working on image pairs")
    pair = False
    sys.exit()
else:
    img_pair = img_list[0:2]
    img_pair = [ GEOTIFF_DIR/f for f in img_pair ]
    pair = True

logger.info("Selected the folling image pair:")
logger.info(f"{img_pair[0]}")
logger.info(f"{img_pair[1]}")

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# Get image datestrings (set manual/from image pair if this fails due to changed file name
date1 = (normcovar_utils.extract_date_from_filename(img_pair[0].stem)).strftime("%Y%m%dT%H%M%S")
date2 = (normcovar_utils.extract_date_from_filename(img_pair[1].stem)).strftime("%Y%m%dT%H%M%S")

# Define output_dir
IMG_PAIR_DIR = SITE_DIR / f"S1_image_pair_{date1}_{date2}"

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# Temporal baseline
min_temp_baseline = 11.9
max_temp_baseline = 12.1

# Output epsg
output_epsg = 3031

normcovar_utils.check_and_trim_image_pair(
    img_pair,
    IMG_PAIR_DIR,
    min_temp_baseline = min_temp_baseline,
    max_temp_baseline = max_temp_baseline,
    output_epsg = output_epsg,
    date1 = None,
    date2 = None,
    overwrite = False,
)

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# Define window sizes to process
window_list = [11,21,33]

# Save intermediate normCoVar steps?
save_intermediate_products = True

# Define min/max values for normCoVar scaling to RGB image
NP_min = -0.5
NP_max = 1.0

# Define path to landmask shapefile (set to None if no landmask needed/available)
landmask_shapefile_path = "/g/data/jk72/jl0818/DATA/fast_ice_tests/coastline_shapefiles/add_coastline_high_res_polygon_v7_5/add_coastline_high_res_polygon_v7_5.shp"

# Set number of pixels for landmask erosion (set to None if no erosion needed)
erode_landmask = 100

# Resample NP RGB image
resample = True

# Set resamping interval for NP RGB image and landmask
resample_interval = 10

normcovar.fully_process_single_image_pair(
    IMG_PAIR_DIR,
    windows = window_list,
    save_intermediate_products = save_intermediate_products,
    NP_min = NP_min,
    NP_max = NP_max,
    landmask_shapefile_path = landmask_shapefile_path,
    erode_landmask = erode_landmask,
    resample = resample,
    resample_interval = resample_interval,
)

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <process_single_img_pair_locally.py> ----

