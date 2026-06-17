# ---- This is <config_loader.py> ----

"""
Facilitate loading of config.yaml in batch processing.
cfg dict is explicitly unpacked in batch processing scripts.
"""

import yaml
import pathlib
from loguru import logger

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def load_config(config_path="../config/config.yaml"):

    config_path = pathlib.Path(config_path)

    if not config_path.is_file():
        logger.warning(f"Could not find config_path: {config_path}")
        logger.warning(f"Trying to sert absolute paths automatically...")

        base_path = pathlib.Path(__file__).resolve().parent.parent.parent
        config_path = base_path / "config" / "config.yaml"

        logger.debug(f"Set base_path:   {base_path}")
        logger.debug(f"Set config_path: {config_path}")

        if not config_path.is_file():
            logger.error(f"Could not find re-set config_path: {config_path}")
            return False

        else:
            logger.warning(f"Reset config_path to: {config_path}")
            logger.warning(f"Ensure that this is your correct config")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Convert string paths to Path objects
    cfg['DATA_DIR']    = pathlib.Path(cfg['data_dir']).resolve()
    cfg['SITE_DIR']    = cfg['DATA_DIR'] / cfg['site']
    cfg['GEOTIFF_DIR'] = cfg['SITE_DIR'] / "GA_geotiffs"

    cfg['LANDMASK_SHAPEFILE_PATH'] = pathlib.Path(cfg['landmask_shapefile_path']).resolve()

    cfg['PBS_TEMPLATE_FILE'] = pathlib.Path(cfg['pbs_template']).resolve()
    cfg['PBS_RUN_SCRIPT']    = pathlib.Path(cfg['pbs_run_script']).resolve()
    cfg['PBS_LOG_DIR']       = pathlib.Path(cfg['pbs_log_dir']).resolve()

    return cfg

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <config_loader.py> ----

