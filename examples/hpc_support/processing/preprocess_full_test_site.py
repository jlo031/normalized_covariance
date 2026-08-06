# ---- This is <preprocess_full_test_site.py> ----

"""
Loop over test site specified in config.yaml file.

- Find all S1*tif files for given test site
- Loop over all combinations of image pairs
- If an image pair meets temporal baseline criteria:
    - Check and trim pair and write 'georeg*tif' files to IMG_PAIR_DIR
    - Add IMG_PAIR_DIR to list of valid folders
    - If 'compute_normcovar_locally' is set to 'True' in config.yaml:
        - compute normcovar locally
        - Not recommended if processing many image pairs (~40minutes per pair)
""" 

import pathlib
from loguru import logger
import sys
import shutil

import math
from itertools import combinations

import numpy as np
from osgeo import gdal

from normalized_covariance import normcovar, normcovar_utils

from utils.config_loader import load_config

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def preprocess_full_test_site():

    # Load config settings
    cfg = load_config()

    if cfg == False:
        logger.error(f"Failed to read config")
        return

    # Set logger
    logger.remove()
    logger.add(sys.stderr, level=cfg["loglevel"])

    # Extract parameters from config
    DATA_DIR                   = cfg["DATA_DIR"]
    SITE_DIR                   = cfg["SITE_DIR"]
    GEOTIFF_DIR                = cfg["GEOTIFF_DIR"]
    site                       = cfg["site"]
    min_temp_baseline          = cfg["min_temp_baseline"]
    max_temp_baseline          = cfg["max_temp_baseline"]
    output_epsg                = cfg["output_epsg"]
    window_list                = cfg["window_list"]
    compute_normcovar_locally   = cfg["compute_normcovar_locally"]
    save_intermediate_products = cfg["save_intermediate_products"]
    NP_min                     = cfg["NP_min"]
    NP_max                     = cfg["NP_max"]
    LANDMASK_SHAPEFILE_PATH    = cfg["LANDMASK_SHAPEFILE_PATH"]
    erode_landmask             = cfg["erode_landmask"]
    overwrite                  = cfg["overwrite"]
    resample                   = cfg["resample"]
    resample_interval          = cfg["resample_interval"]
    loglevel                   = cfg["loglevel"]
    conda_sh                   = cfg["conda_sh"]
    conda_env                  = cfg["conda_env"]
    PBS_TEMPLATE_FILE          = cfg["PBS_TEMPLATE_FILE"]
    PBS_RUN_SCRIPT             = cfg["PBS_RUN_SCRIPT"]
    PBS_LOG_DIR                = cfg["PBS_LOG_DIR"]

    for key in cfg.keys():
        logger.debug(f"cfg key/value pair: '{key}': {cfg[key]}")

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# NOTHING TO CHANGE FROM HERE

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

    logger.info(f"Starting processing of full test site...")

    logger.info(f"DATA_DIR:    {DATA_DIR}")
    logger.info(f"SITE_DIR:    {SITE_DIR}")
    logger.info(f"GEOTIFF_DIR: {GEOTIFF_DIR}")

    if not GEOTIFF_DIR.is_dir():
        logger.error(f"Could not find GEOTIFF_DIR: {GEOTIFF_DIR}")

    # GEOTIFF_DIR should contain the GA processed geotiff files with backscatter intensity
    # List all 'S1*tif' files for current test site
    img_list = [ f.name for f in GEOTIFF_DIR.iterdir() if f.name.startswith("S1") and f.name.endswith("tif") ]
    img_list.sort()

    logger.info(f"Found {len(img_list)} 'S1*tif' images in GEOTIFF_DIR")

    # Report error if not at least two images found
    if not any(img_list):
        logger.error("No 'S1*tif' files found in GEOTIFF_DIR")
        sys.exit()

    if len(img_list) == 1:
        logger.warning("Only found one image in GEOTIFF_DIR, no pairs available")
        sys.exit()

    total_img_pairs = math.comb(len(img_list), 2)
    logger.info(f"Total number of img_pairs to process: {total_img_pairs}")

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

    # Loop over all possible image pair combinations

    # Initialize a list of image pair folders that match the temporal baseline settings
    valid_image_pair_folders = []

    # Build path to file where list of valid img pairs will be saved
    valid_pairs_list = SITE_DIR / f"{site}__valid_img_pair_list.txt"
    logger.info(f"List of valid image pairs will be saved to: {valid_pairs_list}")

    for count,img_pair in enumerate(combinations(img_list, 2),1):

        # Convert tuples to pairs (expexted by normcovar functions)
        img_pair = list(img_pair)
        logger.info(f"Processing image pair {count}/{total_img_pairs}")
        logger.info(f"img1: {img_pair[0]}")
        logger.info(f"img1: {img_pair[1]}")

        # Add full path to img_pair
        img_pair = [ GEOTIFF_DIR/f for f in img_pair ]

        # Get image datestrings
        # Set these manually from image file names if this fails
        # If you need to do this, then you also need
        date1 = (normcovar_utils.extract_date_from_filename(img_pair[0].stem)).strftime("%Y%m%dT%H%M%S")
        date2 = (normcovar_utils.extract_date_from_filename(img_pair[1].stem)).strftime("%Y%m%dT%H%M%S")

        # Define IMG_PAIR_DIR
        IMG_PAIR_DIR = SITE_DIR / f"S1_image_pair_{date1}_{date2}"

        logger.debug(f"date1: {date1}")
        logger.debug(f"date2: {date2}")
        logger.debug(f"IMG_PAIR_DIR: {IMG_PAIR_DIR}")

        # Check and trim image pair
        valid_pair = normcovar_utils.check_and_trim_image_pair(
            img_pair,
            IMG_PAIR_DIR,
            min_temp_baseline = min_temp_baseline,
            max_temp_baseline = max_temp_baseline,
            output_epsg = output_epsg,
            date1 = date1,
            date2 = date2,
            overwrite = overwrite,
        )

        # Add IMG_PAIR_DIR to list if it met the temporal baseline criteria and was succesfully processed
        # If this does not work properly, check the False/True returns from 'normcovar_utils.check_and_trim_image_pair'
        if valid_pair:
            valid_image_pair_folders.append(IMG_PAIR_DIR.name)

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

        if valid_pair:

            if not compute_normcovar_locally:
                logger.info("Not performing actual normcovar computation at this point")
                logger.info("You can compute the normcovar by setting 'compute_normcovar_locally=True'")
                logger.info("For many image pairs, it is recommended to do compute normcovar in a separate job")

            else:
                logger.info("Preparing normcovar computation")
                logger.warning("This may take a long time, especially for many valid image pairs")
                logger.warning("Consider distributed processing instead: 'compute_normcovar_locally: false'")

                normcovar.fully_process_single_image_pair(
                    IMG_PAIR_DIR,
                    windows = window_list,
                    save_intermediate_products = save_intermediate_products,
                    NP_min = NP_min,
                    NP_max = NP_max,
                    landmask_shapefile_path = LANDMASK_SHAPEFILE_PATH,
                    erode_landmask = erode_landmask,
                    resample = resample,
                    resample_interval = resample_interval,
                )

    # Write list with valid image pair folders to file for later processing
    logger.info(f"Writing list of valid_image_pair_folders to: {valid_pairs_list}")

    with open(valid_pairs_list, 'w') as f:
        f.write('\n'.join(valid_image_pair_folders) + '\n')
    logger.info(f"Successfully wrote {len(valid_image_pair_folders)} entries to {valid_pairs_list}")

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

if __name__ == "__main__":
    preprocess_full_test_site()

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <preprocess_full_test_site.py> ----



