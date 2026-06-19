# ---- This is <normprod.py> ----

"""
Module for normprod (normalized product) computation.
Developed as part of the AAPP/UTAS tool for Antarctic fast ice mapping.

Initial developments by A.P. Doulgeris, G. Burke, A. Fraser.

Packaged by J. Lohse.
(johannes.lohse@utas.edu.au)
"""

import pathlib
from loguru import logger

import numpy as np
from scipy.ndimage import uniform_filter, generic_filter

from osgeo import gdal

from normalized_product import normprod_utils

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def compute_DoB(image_path, output_path, window):
    """Compute the difference from 2D boxcar smoothing (boxcar DoG-like) while preventing NaN spread."""

    logger.info(f"Starting DoB computation for w={window}...")

    image_path  = pathlib.Path(image_path)
    output_path = pathlib.Path(output_path)

    logger.debug(f"image_path:  {image_path}")
    logger.debug(f"output_path: {output_path}")
    logger.debug(f"window:      {window}")

    if output_path.is_file():
        logger.info(f"Skipping, {output_path} already exists.")
        return True

    if not image_path.is_file():
        logger.error(f"Could not find image_path: {image_path}.")
        return False

    ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    if ds is None:
        logger.error(f"Cannot open image_path: {image_path}")
        return False

    # Get input band (can be HH or HV, input image, but should just have one single band
    band = ds.GetRasterBand(1).ReadAsArray()

    # Fill NaNs before filtering
    band_filled = normprod_utils.fill_nans(band)

    # Apply 2D boxcar filter
    smoothed = uniform_filter(band_filled, size=window, mode="nearest")

    # Difference from smoothed version
    DoB = band - smoothed

# --------------------- #

    # Save result
    driver = gdal.GetDriverByName("GTIFF")
    out_ds = driver.Create(output_path, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Float32, options=["COMPRESS=DEFLATE", "BIGTIFF=YES"])
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(DoB)
    out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    out_ds.FlushCache()

    # Clean up
    out_ds = None
    ds = None

    logger.info(f"Saved DoB image: {output_path}")

    return output_path

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def compute_local_std(image_path, output_path, window):
    """Compute the local standard deviation in a boxcar window."""

    logger.info(f"Starting local std computation for w={window}...")

    image_path  = pathlib.Path(image_path)
    output_path = pathlib.Path(output_path)

    logger.debug(f"image_path:  {image_path}")
    logger.debug(f"output_path: {output_path}")
    logger.debug(f"window:      {window}")

    if output_path.is_file():
        logger.info(f"Skipping, {output_path} already exists.")
        return output_path

    if not image_path.is_file():
        logger.error(f"Could not find image_path: {image_path}.")
        return

    ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    if ds is None:
        logger.error(f"Cannot open image_path: {image_path}")
        return

    # Get input band (can be HH or HV, input image, but should just have one single band
    band = ds.GetRasterBand(1).ReadAsArray()

    # Fill NaNs before filtering
    band_filled = normprod_utils.fill_nans(band)

    # Compute mean and mean of squares in local window
    local_mean = uniform_filter(band_filled, size=window, mode="nearest")
    local_mean_sq = uniform_filter(band_filled**2, size=window, mode="nearest")

    # Standard deviation: sqrt(E[x^2] - (E[x])^2)
    local_std = np.sqrt(local_mean_sq - local_mean**2)

# --------------------- #

    # Save result
    driver = gdal.GetDriverByName("GTIFF")
    out_ds = driver.Create(output_path, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Float32, options=["COMPRESS=DEFLATE", "BIGTIFF=YES"])
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(local_std)
    out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    out_ds.FlushCache()

    # Clean up
    out_ds = None
    ds = None

    logger.info(f"Saved local std image: {output_path}")

    return output_path

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def compute_normprod(
    dob1,
    dob2,
    std1,
    std2,
    normprod_smovar_output_path,
    window,
    save_intermediate_products=False
):
    """Compute Normalized Product (NormProd) using precomputed smoothed images, with NaN-safe summation.

    Parameters
    ----------
    dob1 : path to input DoB image 1
    dob2 : path to input DoB image 2
    std1 : path to input local std image 1
    std2 : path to input local std image 2
    normprod_smovar_output_path : path to output file, normprod divided by smoothed variance
    window : window size for normalized product (e.g. 11, 21, 33)
    save_intermediate_products : save intermediate products as tif files (default=False)
    """

    logger.info(f"Starting normprod_smovar computation for w={window}...")

    logger.debug(f"dob1: {dob1}")
    logger.debug(f"dob2: {dob2}")
    logger.debug(f"std1: {std1}")
    logger.debug(f"std2: {std2}")

    dob1 = pathlib.Path(dob1)
    dob2 = pathlib.Path(dob2)
    std1 = pathlib.Path(std1)
    std2 = pathlib.Path(std2)
    normprod_smovar_output_path = pathlib.Path(normprod_smovar_output_path)

    if normprod_smovar_output_path.is_file():
        logger.info(f"Skipping, {normprod_smovar_output_path} already exists.")
        return True

    if not dob1.is_file():
        logger.error(f"Could not find dob1: {dob1}.")
        return False

    if not dob2.is_file():
        logger.error(f"Could not find dob2: {dob2}.")
        return False

    if not std1.is_file():
        logger.error(f"Could not find std1: {std1}.")
        return False

    if not std2.is_file():
        logger.error(f"Could not find std2: {std2}.")
        return False

# --------------------- #

    # Read all input data

    ds_dob1 = gdal.Open(dob1, gdal.GA_ReadOnly)
    ds_dob2 = gdal.Open(dob2, gdal.GA_ReadOnly)
    ds_std1 = gdal.Open(std1, gdal.GA_ReadOnly)
    ds_std2 = gdal.Open(std2, gdal.GA_ReadOnly)

    if not all([ds_dob1, ds_dob2, ds_std1, ds_std2]):
        logger.error(f"Could not open all required input files.")
        return False

    logger.debug("Reading input data.")
    dob1 = ds_dob1.GetRasterBand(1).ReadAsArray()
    dob2 = ds_dob2.GetRasterBand(1).ReadAsArray()
    std1 = ds_std1.GetRasterBand(1).ReadAsArray()
    std2 = ds_std2.GetRasterBand(1).ReadAsArray()    

# --------------------- #

    # OLD

    # Compute mean of std images
    logger.debug("Computing stdmean.")
    stdmean = np.mean(np.stack([std1, std2], axis=0), axis=0)

    if save_intermediate_products:
        logger.debug("Saving stdmean...")
        intermediate_output_path = normprod_smovar_output_path.parent / f"stdmean_window{window}.tif"
        driver = gdal.GetDriverByName("GTIFF")
        out_ds = driver.Create(intermediate_output_path, ds_dob1.RasterXSize, ds_dob1.RasterYSize, 1, gdal.GDT_Float32, options=["COMPRESS=DEFLATE", "BIGTIFF=YES"])
        out_ds.SetGeoTransform(ds_dob1.GetGeoTransform())
        out_ds.SetProjection(ds_dob1.GetProjection())
        out_ds.GetRasterBand(1).WriteArray(stdmean)
        out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
        out_ds.FlushCache()
        out_ds = None
        logger.debug(f"Saved stdmean image: {intermediate_output_path}")

# --------------------- #

    # Square stdmean to get varianve
    logger.debug("Computing variance.")
    variance = stdmean*stdmean

    # Fill NaNs (again, just in case)
    logger.debug("Filling nans.")
    variance_filled = normprod_utils.fill_nans(variance)

    # Clean up
    stdmean = variance = None

    if save_intermediate_products:
        logger.debug("Saving intermediate output: variance.")
        intermediate_output_path = normprod_smovar_output_path.parent / f"variance_window{window}.tif"
        driver = gdal.GetDriverByName("GTIFF")
        out_ds = driver.Create(intermediate_output_path, ds_dob1.RasterXSize, ds_dob1.RasterYSize, 1, gdal.GDT_Float32, options=["COMPRESS=DEFLATE", "BIGTIFF=YES"])
        out_ds.SetGeoTransform(ds_dob1.GetGeoTransform())
        out_ds.SetProjection(ds_dob1.GetProjection())
        out_ds.GetRasterBand(1).WriteArray(variance_filled)
        out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
        out_ds.FlushCache()
        out_ds = None
        logger.debug(f"Saved variance image: {intermediate_output_path}")

# --------------------- #

    # Compute smothed variance with boxcar of size window
    logger.debug("Computing smoothed variance.")
    smoothed_variance = uniform_filter(variance_filled, size=window, mode="nearest")

    if save_intermediate_products:
        logger.debug("Saving intermediate output: smoothed_variance.")
        intermediate_output_path = normprod_smovar_output_path.parent / f"smoothed_variance_window{window}.tif"
        driver = gdal.GetDriverByName("GTIFF")
        out_ds = driver.Create(intermediate_output_path, ds_dob1.RasterXSize, ds_dob1.RasterYSize, 1, gdal.GDT_Float32, options=["COMPRESS=DEFLATE", "BIGTIFF=YES"])
        out_ds.SetGeoTransform(ds_dob1.GetGeoTransform())
        out_ds.SetProjection(ds_dob1.GetProjection())
        out_ds.GetRasterBand(1).WriteArray(smoothed_variance)
        out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
        out_ds.FlushCache()
        out_ds = None
        logger.debug(f"Saved smoothed_variance image: {intermediate_output_path}")

    # Clean up
    variance_filled = None

# --------------------- #
# --------------------- #
# --------------------- #

    # NEW

    # Compute square of std images
    logger.debug("Computing stdsquare (var).")
    var1 = std1 * std1
    var2 = std2 * std2

    if save_intermediate_products:
        logger.debug("Saving var1...")
        intermediate_output_path = normprod_smovar_output_path.parent / f"var1_window{window}.tif"
        driver = gdal.GetDriverByName("GTIFF")
        out_ds = driver.Create(intermediate_output_path, ds_dob1.RasterXSize, ds_dob1.RasterYSize, 1, gdal.GDT_Float32, options=["COMPRESS=DEFLATE", "BIGTIFF=YES"])
        out_ds.SetGeoTransform(ds_dob1.GetGeoTransform())
        out_ds.SetProjection(ds_dob1.GetProjection())
        out_ds.GetRasterBand(1).WriteArray(var1)
        out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
        out_ds.FlushCache()
        out_ds = None
        logger.debug(f"Saved var1 image: {intermediate_output_path}")

        logger.debug("Saving var2...")
        intermediate_output_path = normprod_smovar_output_path.parent / f"var2_window{window}.tif"
        driver = gdal.GetDriverByName("GTIFF")
        out_ds = driver.Create(intermediate_output_path, ds_dob1.RasterXSize, ds_dob1.RasterYSize, 1, gdal.GDT_Float32, options=["COMPRESS=DEFLATE", "BIGTIFF=YES"])
        out_ds.SetGeoTransform(ds_dob1.GetGeoTransform())
        out_ds.SetProjection(ds_dob1.GetProjection())
        out_ds.GetRasterBand(1).WriteArray(var2)
        out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
        out_ds.FlushCache()
        out_ds = None
        logger.debug(f"Saved var2 image: {intermediate_output_path}")

    # Clean up
    std1 = std2 = None

# --------------------- #

    # Compute mean of var images
    logger.debug("Computing mean_var.")
    mean_variance = np.mean(np.stack([var1, var2], axis=0), axis=0)

    if save_intermediate_products:
        logger.debug("Saving mean_variance...")
        intermediate_output_path = normprod_smovar_output_path.parent / f"mean_variance_window{window}.tif"
        driver = gdal.GetDriverByName("GTIFF")
        out_ds = driver.Create(intermediate_output_path, ds_dob1.RasterXSize, ds_dob1.RasterYSize, 1, gdal.GDT_Float32, options=["COMPRESS=DEFLATE", "BIGTIFF=YES"])
        out_ds.SetGeoTransform(ds_dob1.GetGeoTransform())
        out_ds.SetProjection(ds_dob1.GetProjection())
        out_ds.GetRasterBand(1).WriteArray(mean_variance)
        out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
        out_ds.FlushCache()
        out_ds = None
        logger.debug(f"Saved mean_variance image: {intermediate_output_path}")

    # Clean up
    var1 = var2 = None

    # Fill NaNs (again, just in case)
    logger.debug("Filling nans.")
    mean_variance_filled = normprod_utils.fill_nans(mean_variance)

    # Clean up
    mean_variance = None

# --------------------- #

    # Compute smothed mean_variance with boxcar of size window
    logger.debug("Computing smoothed variance.")
    smoothed_mean_variance = uniform_filter(mean_variance_filled, size=window, mode="nearest")

    if save_intermediate_products:
        logger.debug("Saving intermediate output: smoothed_mean_variance.")
        intermediate_output_path = normprod_smovar_output_path.parent / f"smoothed_mean_variance_window{window}.tif"
        driver = gdal.GetDriverByName("GTIFF")
        out_ds = driver.Create(intermediate_output_path, ds_dob1.RasterXSize, ds_dob1.RasterYSize, 1, gdal.GDT_Float32, options=["COMPRESS=DEFLATE", "BIGTIFF=YES"])
        out_ds.SetGeoTransform(ds_dob1.GetGeoTransform())
        out_ds.SetProjection(ds_dob1.GetProjection())
        out_ds.GetRasterBand(1).WriteArray(smoothed_mean_variance)
        out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
        out_ds.FlushCache()
        out_ds = None
        logger.debug(f"Saved smoothed_mean_variance image: {intermediate_output_path}")

    # Clean up
    mean_variance_filled = None

# --------------------- #
# --------------------- #
# --------------------- #

    # Compute NormProd: DoB1*DoB2
    logger.debug("Computing normprod.")
    normprod = dob1 * dob2

    if save_intermediate_products:
        logger.debug("Saving intermediate output: normprod.")
        intermediate_output_path = normprod_smovar_output_path.parent / f"normprod_window{window}.tif"
        driver = gdal.GetDriverByName("GTIFF")
        out_ds = driver.Create(intermediate_output_path, ds_dob1.RasterXSize, ds_dob1.RasterYSize, 1, gdal.GDT_Float32, options=["COMPRESS=DEFLATE", "BIGTIFF=YES"])
        out_ds.SetGeoTransform(ds_dob1.GetGeoTransform())
        out_ds.SetProjection(ds_dob1.GetProjection())
        out_ds.GetRasterBand(1).WriteArray(normprod)
        out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
        out_ds.FlushCache()
        out_ds = None
        logger.debug(f"Saved normprod image: {intermediate_output_path}")

# --------------------- #

    # Apply a NaN-safe mean using generic_filter
    kernel = np.ones((window, window))  # Window for summation

    logger.debug("Starting generic_filter... about 10 mins for 11*11, 12 mins for 21*21.")

    mean_normprod = generic_filter(normprod, normprod_utils.nan_safe_mean_filter, footprint=kernel, mode='constant', cval=np.nan)

    logger.debug("Finished generic_filter")

    if save_intermediate_products:
        logger.debug("Saving intermediate output: mean_normprod.")
        intermediate_output_path = normprod_smovar_output_path.parent / f"mean_normprod_window{window}.tif"
        driver = gdal.GetDriverByName("GTIFF")
        out_ds = driver.Create(intermediate_output_path, ds_dob1.RasterXSize, ds_dob1.RasterYSize, 1, gdal.GDT_Float32, options=["COMPRESS=DEFLATE", "BIGTIFF=YES"])
        out_ds.SetGeoTransform(ds_dob1.GetGeoTransform())
        out_ds.SetProjection(ds_dob1.GetProjection())
        out_ds.GetRasterBand(1).WriteArray(mean_normprod)
        out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
        out_ds.FlushCache()
        out_ds = None
        logger.debug(f"Saved mean_normprod image: {intermediate_output_path}")

    # Clean up
    variance_filled = None

# --------------------- #

    # Divide normprod by smoothed variance
    normprod_smovar = mean_normprod/smoothed_mean_variance
    normprod_smovar_OLD = mean_normprod/smoothed_variance

    # Clean up
    summed_normprod = smoothed_variance = smoothed_mean_variance= None

# --------------------- #

    # Write normprod_smovar to disk

    logger.debug("Saving normprod_smovar.")

    driver = gdal.GetDriverByName("GTIFF")
    out_ds = driver.Create(normprod_smovar_output_path, ds_dob1.RasterXSize, ds_dob1.RasterYSize, 1, gdal.GDT_Float32, options=["COMPRESS=DEFLATE", "BIGTIFF=YES"])
    out_ds.SetGeoTransform(ds_dob1.GetGeoTransform())
    out_ds.SetProjection(ds_dob1.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(normprod_smovar)
    out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    out_ds.FlushCache()
    out_ds = None


    intermediate_output_path = normprod_smovar_output_path.parent / f"OLD_normprod_smovar_window{window}.tif"
    driver = gdal.GetDriverByName("GTIFF")
    out_ds = driver.Create(intermediate_output_path, ds_dob1.RasterXSize, ds_dob1.RasterYSize, 1, gdal.GDT_Float32, options=["COMPRESS=DEFLATE", "BIGTIFF=YES"])
    out_ds.SetGeoTransform(ds_dob1.GetGeoTransform())
    out_ds.SetProjection(ds_dob1.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(normprod_smovar_OLD)
    out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    out_ds.FlushCache()
    out_ds = None



    logger.info(f"Saved normprod_smovar: {normprod_smovar_output_path}.")

    # Clean up
    logger.debug("Freeing memory.")
    dob1 = dob2 = std1 = std2 = ds_dob1 = ds_dob2 = ds_std1 = ds_sdt2 = None
    normprod_smovar = None

# --------------------- #

    return True

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def fully_process_single_image_pair(
    img_pair_dir,
    windows = [11,21,33],
    save_intermediate_products = False,
    NP_min = -0.5,
    NP_max = 1.0,
    landmask_shapefile_path = None,
    erode_landmask = None,
    resample = True,
    resample_interval = 10,
):
    """
    Full Normprod processing for single image pair that has already been checked and trimmed.
        - DoB for each image
        - local std for each image
        - normprod_smovar for image pair
        - stack normprod_smovar to RGB image

    Parameters
    ----------
    img_pair_dir : Path to image pair directory
    windows : List of window sizes for normprod processing (default=[11,21,33])
    save_intermediate_products : Save intermediate products as tif files (default=False)
    NP_min : Min NP value for scaling to RGB (default=-0.5)
    NP_max : Max NP value for scaling to RGB (default=1.0)
    landmask_shapefile_path : Path to landmask shapefile
    erode_landmask : Erode landmask by number of pixels (default=None)
    resample : Resample NP RGB image for processing with SAM (default=True)
    resample_interval : Resamping interval (default=10)
    Returns
    -------
    """

    logger.info(f"Starting full normprod processing chain for img_pair_dir...")

    # Ensure that img_pair_dir is pathlib.Path object    
    img_pair_dir = pathlib.Path(img_pair_dir)

    if not img_pair_dir.exists():
        logger.error(f"Could not find img_pair_dir: {img_pair_dir}")
        return False
        if not img_pair_dir.is_dir():
            logger.error(f"img_pair_dir must be folder {img_pair_dir}")
            return False

# --------------------- #

    # List all files in img_pair_dir
    tif_file_list = [ f for f in img_pair_dir.glob("*.tif") ]

    logger.debug(f"Found {len(tif_file_list)} tif files in img_pair_dir")
    for ii,tif_file in enumerate(tif_file_list):
        logger.debug(f"tif_file {ii+1}: {tif_file}")

    # Find the original georeg files
    # Make sure to exclude previously processed DoB or std images
    exclude_list = ["DoB", "dob", "std"]

    # List the georeg files for the IMG_PAIR_DIR
    georeg_pair = [ f for f in tif_file_list if f.name.startswith("georeg") and not any(excluded in f.name for excluded in exclude_list) ]
    georeg_pair.sort(key=lambda p: p.name)

    logger.info(f"Found {len(georeg_pair)} georeg*tif files in img_pair_dir:")
    for i, georeg_img in enumerate(georeg_pair):
        logger.info(f"georeg_{i+1}: {georeg_img}")

    if not len(georeg_pair)==2:
        logger.error(f"Expected exactly 2 files in georeg_pair, but found {len(georeg_pair)}.")
        return False

# --------------------- #

    logger.info(f"Computing DoB, local_std, and normprod_smovar for the following window sizes: {windows}")

    georeg_path_1 = georeg_pair[0]
    georeg_path_2 = georeg_pair[1]
    georeg_basename_1 = georeg_path_1.stem
    georeg_basename_2 = georeg_path_2.stem

    logger.debug(f"georeg_path_1: {georeg_path_1}")
    logger.debug(f"georeg_path_2: {georeg_path_2}")
    logger.debug(f"georeg_basename_1: {georeg_basename_1}")
    logger.debug(f"georeg_basename_2: {georeg_basename_2}")

    for window in windows:
        logger.info(f"Computing DoB and local std for window: {window}")

        dob_path_1 = img_pair_dir / f"{georeg_basename_1}_DoB_window{window}.tif"
        dob_path_2 = img_pair_dir / f"{georeg_basename_2}_DoB_window{window}.tif"
        std_path_1 = img_pair_dir / f"{georeg_basename_1}_local_std_window{window}.tif"
        std_path_2 = img_pair_dir / f"{georeg_basename_2}_local_std_window{window}.tif"
        normprod_smovar_path = img_pair_dir / f"normprod_smovar_window{window}.tif"

        logger.debug(f"dob_path_1: {dob_path_1}")
        logger.debug(f"dob_path_2: {dob_path_2}")
        logger.debug(f"std_path_1: {std_path_1}")
        logger.debug(f"std_path_2: {std_path_2}")
        logger.debug(f"normprod_smovar_path: {normprod_smovar_path}")

        compute_DoB(georeg_path_1, dob_path_1, window)
        compute_DoB(georeg_path_2, dob_path_2, window)

        compute_local_std(georeg_path_1, std_path_1, window)
        compute_local_std(georeg_path_2, std_path_2, window)

        logger.info(f"Computing normprod_smovar for window: {window}")

        compute_normprod(
            dob_path_1,
            dob_path_2,
            std_path_1,
            std_path_2,
            normprod_smovar_path,
            window,
            save_intermediate_products=save_intermediate_products
        )

    # --------------------- #

    logger.info("Stacking to false-color RGB")

    if not len(windows)==3:
        logger.error(f"Expected three different window sizes for RGB stack, but len(windows) is {len(windows)}")
        return False

    img1_path   = img_pair_dir / f"normprod_smovar_window{windows[0]}.tif"
    img2_path   = img_pair_dir / f"normprod_smovar_window{windows[1]}.tif"
    img3_path   = img_pair_dir / f"normprod_smovar_window{windows[2]}.tif"
    output_path = img_pair_dir / f"normprod_smovar_RGB.tif"

    logger.debug(f"NP_min:{NP_min}")
    logger.debug(f"NP_min:{NP_max}")
    logger.debug(f"img1_path:{img1_path}")
    logger.debug(f"img2_path:{img2_path}")
    logger.debug(f"img3_path:{img3_path}")

    normprod_utils.stack_2_RGB(
        img1_path,
        img2_path,
        img3_path,
        output_path,
        img_min = NP_min,
        img_max = NP_max,
        new_min = 0,
        new_max = 255,
        overwrite = False
    )




    # REMOVE THIS PART LATER

    img1_path   = img_pair_dir / f"OLD_normprod_smovar_window{windows[0]}.tif"
    img2_path   = img_pair_dir / f"OLD_normprod_smovar_window{windows[1]}.tif"
    img3_path   = img_pair_dir / f"OLD_normprod_smovar_window{windows[2]}.tif"
    output_path = img_pair_dir / f"OLD_normprod_smovar_RGB.tif"

    logger.debug(f"NP_min:{NP_min}")
    logger.debug(f"NP_min:{NP_max}")
    logger.debug(f"img1_path:{img1_path}")
    logger.debug(f"img2_path:{img2_path}")
    logger.debug(f"img3_path:{img3_path}")

    normprod_utils.stack_2_RGB(
        img1_path,
        img2_path,
        img3_path,
        output_path,
        img_min = NP_min,
        img_max = NP_max,
        new_min = 0,
        new_max = 255,
        overwrite = False
    )













    if resample:

        logger.info("Resampling RGB image")

        geotiff_path = img_pair_dir / f"normprod_smovar_RGB.tif"
        output_path  = img_pair_dir / f"normprod_smovar_RGB_resampled_{resample_interval}_{resample_interval}.tif"

        logger.debug(f"geotiff_path:      {geotiff_path}")
        logger.debug(f"output_path:       {output_path}")
        logger.debug(f"resample_interval: {resample_interval}")

        normprod_utils.resample_geotiff(
            geotiff_path,
            output_path,
            zoom_x=resample_interval,
            zoom_y=resample_interval,
            order=1,
            overwrite=False,
        )

    # --------------------- #

    if landmask_shapefile_path is not None:

        logger.info("Creating landmask image")

        geotiff_path = img_pair_dir / f"normprod_smovar_RGB.tif"
        output_path  = img_pair_dir / f"landmask.tif"

        logger.debug(f"geotiff_path:            {geotiff_path}")
        logger.debug(f"output_path:             {output_path}")
        logger.debug(f"landmask_shapefile_path: {landmask_shapefile_path}")

        normprod_utils.save_landmask_file_4_geotiff(
            geotiff_path,
            landmask_shapefile_path,
            output_path,
            erode_landmask=None,
        )


        if resample:

            logger.info("Creating resampled landmask image")

            geotiff_path = img_pair_dir / f"normprod_smovar_RGB_resampled_{resample_interval}_{resample_interval}.tif"
            output_path  = img_pair_dir / f"landmask_resampled_{resample_interval}_{resample_interval}.tif"

            logger.debug(f"geotiff_path:      {geotiff_path}")
            logger.debug(f"output_path:       {output_path}")
            logger.debug(f"resample_interval: {resample_interval}")

            normprod_utils.save_landmask_file_4_geotiff(
                geotiff_path,
                landmask_shapefile_path,
                output_path,
                erode_landmask=None,
            )

        # --------------------- #

        if erode_landmask is not None:

            logger.info("Creating eroded landmask image")

            geotiff_path = img_pair_dir / f"normprod_smovar_RGB.tif"
            output_path  = img_pair_dir / f"landmask_eroded_{erode_landmask}.tif"

            logger.debug(f"geotiff_path:            {geotiff_path}")
            logger.debug(f"output_path:             {output_path}")
            logger.debug(f"landmask_shapefile_path: {landmask_shapefile_path}")

            normprod_utils.save_landmask_file_4_geotiff(
                geotiff_path,
                landmask_shapefile_path,
                output_path,
                erode_landmask=erode_landmask,
            )


            if resample:

                logger.info("Resampling eroded resampled landmask image")

                geotiff_path = img_pair_dir / f"landmask_eroded_{erode_landmask}.tif"
                output_path  = img_pair_dir / f"landmask_eroded_{erode_landmask}_resampled_{resample_interval}_{resample_interval}.tif"

                logger.debug(f"geotiff_path:      {geotiff_path}")
                logger.debug(f"output_path:       {output_path}")
                logger.debug(f"resample_interval: {resample_interval}")

                normprod_utils.resample_geotiff(
                    geotiff_path,
                    output_path,
                    zoom_x=resample_interval,
                    zoom_y=resample_interval,
                    order=1,
                    overwrite=False,
                )

    # --------------------- #

    return True

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <normprod.py> ----
