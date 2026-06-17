# ---- This is <batch_process_normprod_smovar.py> ----

"""
Batch process normprod_smovar for test site specified in config.yaml file.
Requires pre-processing of the test site using the same config settings.

- Loop over txt file with pre-processed IMG_PAIR_DIR entries
- Create an individual PBS job for each image pair and start a separate job for:
    - DoB
    - local_std
    - normprod_smovar
""" 

import pathlib
from loguru import logger
import sys
import shutil

import math
from itertools import combinations
import subprocess

import numpy as np
from osgeo import gdal

from normalized_product import normprod, normprod_utils

from utils.config_loader import load_config

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def batch_process_img_pair_folders_distributed():

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
    compute_normprod_locally   = cfg["compute_normprod_locally"]
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

    logger.info(f"Starting batch processing of full test site...")

    logger.info(f"DATA_DIR:    {DATA_DIR}")
    logger.info(f"SITE_DIR:    {SITE_DIR}")
    logger.info(f"GEOTIFF_DIR: {GEOTIFF_DIR}")

    # Build path to file with list of valid img pairs
    LIST_FILE_PATH = SITE_DIR / f"{site}__valid_img_pair_list.txt"
    logger.info(f"Reading list of valid image pairs from: {LIST_FILE_PATH}")

    if not LIST_FILE_PATH.is_file():
        logger.error(f"Could not find LIST_FILE_PATH: {LIST_FILE_PATH}")

    if not PBS_TEMPLATE_FILE.is_file():
         logger.error(f"Could not find PBS_TEMPLATE_FILE: {PBS_TEMPLATE_FILE}")

    if not PBS_RUN_SCRIPT.is_file():
        logger.error(f"Could not find PBS_RUN_SCRIPT: {PBS_RUN_SCRIPT}")

    # Make sure that PBS_LOG_DIR exists
    PBS_LOG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

    # Read the image directories from the list file
    with open(LIST_FILE_PATH, 'r') as f:
        image_dirs = [line.strip() for line in f if line.strip()]

    if not any(image_dirs):
        logger.error("Could not read any image_dirs from LIST_FILE_PATH")

    # Get number of image pairs to process
    n_pairs = len(image_dirs)
    logger.info(f"Read {n_pairs} IMG_PAIR_DIRs from LIST_FILE_PATH")

    # image_dirs must contain the full paths 
    image_dirs = [ SITE_DIR/image_dir for image_dir in image_dirs ]

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

    # Read the PBS template
    with open(PBS_TEMPLATE_FILE, 'r') as f:
        pbs_template = f.read()

    # Create emtpty job list
    pbs_job_list = []

    # Generate the PBS scripts
    for IMG_PAIR_DIR in image_dirs:

        if not IMG_PAIR_DIR.is_dir():
            logger.error(f"Could not find IMG_PAIR_DIR: {IMG_PAIR_DIR}")

        img_pair_name = IMG_PAIR_DIR.name

        # Set parameters for processing in dictionary
        params = {
            "jobname": f"PBS_JOB_{img_pair_name}",
            "log_dir": PBS_LOG_DIR,
            "conda_sh": conda_sh,
            "conda_env": conda_env,
            "run_script_path": PBS_RUN_SCRIPT,
            "img_pair_dir": IMG_PAIR_DIR,
            "windows": window_list,
            "save_intermediate_products": save_intermediate_products,
            "NP_min": NP_min,
            "NP_max": NP_max,
            "landmask_shapefile_path": LANDMASK_SHAPEFILE_PATH,
            "erode_landmask": erode_landmask,
            "resample": resample,
            "resample_interval": resample_interval,
            "loglevel": loglevel
        }

        # Fill PBS template and write to file
        pbs_template_filled = pbs_template.format(**params)

        pbs_script_path = PBS_LOG_DIR / f"submit_{img_pair_name}.pbs"
    
        with open(pbs_script_path, 'w') as f:
            f.write(pbs_template_filled)
    
        logger.info(f"Generated PBS script for {img_pair_name}: {pbs_script_path}")

        pbs_job_list.append(pbs_script_path)


    for pbs_job in pbs_job_list:

        subprocess.run(["qsub", pbs_job])

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

if __name__ == "__main__":
    batch_process_img_pair_folders_distributed()

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <batch_process_img_pair_folders_distributed.py> ----

