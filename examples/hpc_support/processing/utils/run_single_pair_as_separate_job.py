# ---- This is <run_single_pair.py> ----

"""
Process normnprod_smovar for input IMG_PAIR_DIR.
Desgined to be called as pbs job in disctirbuted processing for a full test site.
"""

import sys
import ast
import argparse
from pathlib import Path

from loguru import logger

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def str_to_bool(value: str) -> bool:
    """Helper to handle various boolean string representations."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (True/False).")

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def run_single_pair():
    parser = argparse.ArgumentParser(description="Process a single image pair folder.")

    # Positional Arguments
    parser.add_argument("IMG_PAIR_DIR", type=Path, help="Path to the image pair directory")
    parser.add_argument("windows", type=str, help="Python-style list string of window sizes")
    parser.add_argument("save_intermediate_products", type=str_to_bool, help="True/False")
    parser.add_argument("NP_min", type=float, help="Min value for NP scaling to RGB")
    parser.add_argument("NP_max", type=float, help="Max value for NP scaling to RGB")
    parser.add_argument("landmask_shapefile_path", type=Path, help="Path to landmask shapefile")
    parser.add_argument("erode_landmask", type=int, help="Number of pixels for landmask erosion")
    parser.add_argument("resample", type=str_to_bool, help="True/False")
    parser.add_argument("resample_interval", type=int, help="Resampling interval")
    parser.add_argument("loglevel", type=str, help="Set loglevel")

    args = parser.parse_args()

    # ---------------------
    # --- Logging Setup ---

    logger.remove()
    logger.add(sys.stdout, level=args.loglevel.upper())

    # -------------------------------------
    # --- Argument Parsing & Validation ---
    
    # Check if directory exists before doing heavy lifting
    if not args.IMG_PAIR_DIR.is_dir():
        logger.error(f"Directory not found: {args.IMG_PAIR_DIR}")
        sys.exit(1)

    # Check that landmask_shapefile exists
    if not args.landmask_shapefile_path.is_file():
        logger.error(f"File not found: {args.landmask_shapefile_path}")
        sys.exit(1)

    # Parse the window list string
    try:
        window_list = ast.literal_eval(args.windows)
        if not isinstance(window_list, list):
            raise ValueError("Input must be a list.")
        # Ensure elements are integers
        window_list = [int(x) for x in window_list]
    except (ValueError, SyntaxError) as e:
        logger.error(f"Failed to parse windows list '{args.windows}': {e}")
        sys.exit(1)

    # -----------------
    # --- Execution ---

    logger.info(f"IMG_PAIR_DIR: {args.IMG_PAIR_DIR.resolve()}")
    logger.info(f"window_list: {window_list}")
    logger.info(f"Save intermediates: {args.save_intermediate_products}")
    logger.info(f"NP_min: {args.NP_min}")
    logger.info(f"NP_max: {args.NP_max}")
    logger.info(f"landmask_shapefile_path: {args.landmask_shapefile_path}")
    logger.info(f"erode_landmask: {args.erode_landmask}")
    logger.info(f"resample: {args.resample}")
    logger.info(f"resample_interval: {args.resample_interval}")


    try:
        # Import only when needed to keep the CLI snappy
        from normalized_product import normprod
        
        normprod.fully_process_single_image_pair(
            args.IMG_PAIR_DIR,
            windows = window_list,
            save_intermediate_products = args.save_intermediate_products,
            NP_min = args.NP_min,
            NP_max = args.NP_max,
            landmask_shapefile_path = args.landmask_shapefile_path,
            erode_landmask = args.erode_landmask,
            resample = args.resample,
            resample_interval = args.resample_interval,
        )

        logger.success("Processing complete.")
        
    except Exception as e:
        logger.error("A runtime error occurred during processing.")
        sys.exit(1)

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

if __name__ == "__main__":
    run_single_pair()

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <run_single_pair.py> ----


