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

import xarray as xr

from normalized_product import normprod_utils

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# numpy array based functions for both file based and xarray processing
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #


def _compute_dob_arr(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute DoB on a 2-D numpy array."""
    arr_filled = normprod_utils.fill_nans(arr)
    smoothed = uniform_filter(arr_filled, size=window, mode="nearest")
    return arr_filled - smoothed


def _compute_local_std_arr(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute local std on a 2-D numpy array using E[x²] - (E[x])²."""
    arr_filled = normprod_utils.fill_nans(arr)
    local_mean = uniform_filter(arr_filled, size=window, mode="nearest")
    local_mean_sq = uniform_filter(arr_filled**2, size=window, mode="nearest")

    # compute variance and clamp to prevent negative floats
    local_variance = local_mean_sq - local_mean**2
    local_variance_clamped = np.clip(local_variance, a_min=0.0, a_max=None)

    return np.sqrt(local_variance_clamped)


def _compute_normprod_smovar_arr(
    dob1: np.ndarray,
    dob2: np.ndarray,
    std1: np.ndarray,
    std2: np.ndarray,
    window: int,
    save_intermediate_products: bool = False,
    intermediate_dir: pathlib.Path = None,
) -> np.ndarray:
    """
    Compute normprod_smovar on 2-D numpy arrays.

    Parameters
    ----------
    dob1, dob2 : DoB arrays for the two images
    std1, std2 : Local std arrays for the two images
    window     : Boxcar window size
    save_intermediate_products : Write intermediate arrays to disk (requires intermediate_dir).
    intermediate_dir : Directory for intermediate GeoTIFF files (ignored when save_intermediate_products=False).
    """

    def _save(arr, name):
        if not save_intermediate_products or intermediate_dir is None:
            return
        path = intermediate_dir / f"{name}_window{window}.tif"
        drv = gdal.GetDriverByName("GTIFF")
        out = drv.Create(
            str(path),
            arr.shape[1],
            arr.shape[0],
            1,
            gdal.GDT_Float32,
            options=["COMPRESS=DEFLATE", "BIGTIFF=YES"],
        )
        out.GetRasterBand(1).WriteArray(arr)
        out.GetRasterBand(1).SetNoDataValue(np.nan)
        out.FlushCache()
        out = None
        logger.debug(f"Saved intermediate: {path}")

    logger.debug("Computing local variance (var).")
    var1 = std1 * std1
    _save(var1, "local_variance_1")
    var2 = std2 * std2
    _save(var2, "local_variance_2")

    # Clean up
    std1 = std2 = None

    logger.debug("Computing man_var.")
    mean_var = (var1 + var2) * 0.5
    _save(mean_var, "local_mean_variance")

    # Clean up
    var1 = var2 = None

    logger.debug("Filling nans.")
    mean_var_filled = normprod_utils.fill_nans(mean_var)
    _save(mean_var_filled, "smoothed_mean_variance")

    # Clean up
    mean_var = None

    logger.debug("Computing smoothed variance.")
    smoothed_mean_var = uniform_filter(mean_var_filled, size=window, mode="nearest")
    _save(smoothed_mean_var, "smoothed_mean_variance")

    # Clean up
    mean_var_filled = None

    logger.debug("Computing normprod.")
    normprod = dob1 * dob2
    _save(normprod, "normprod")

    logger.debug("Average normprod.")
    mean_normprod = uniform_filter(normprod, size=window, mode="nearest")
    _save(mean_normprod, "mean_normprod")

    # Clean up
    normprod = None

    return mean_normprod / smoothed_mean_var


# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# File-based NormProd
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #


def compute_DoB(image_path, output_path, window):
    """Compute the difference from 2D boxcar smoothing (boxcar DoG-like) while preventing NaN spread."""

    logger.info(f"Starting DoB computation for w={window}...")

    image_path = pathlib.Path(image_path)
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

    band = ds.GetRasterBand(1).ReadAsArray()
    DoB = _compute_dob_arr(band, window)

    # --------------------- #

    # Save result
    driver = gdal.GetDriverByName("GTIFF")
    out_ds = driver.Create(
        output_path,
        ds.RasterXSize,
        ds.RasterYSize,
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "BIGTIFF=YES"],
    )
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

    image_path = pathlib.Path(image_path)
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
    local_std = _compute_local_std_arr(band, window)

    # --------------------- #

    # Save result
    driver = gdal.GetDriverByName("GTIFF")
    out_ds = driver.Create(
        output_path,
        ds.RasterXSize,
        ds.RasterYSize,
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "BIGTIFF=YES"],
    )
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
# --------------------------------------------------------------------------


def compute_normprod(
    dob1,
    dob2,
    std1,
    std2,
    normprod_smovar_output_path,
    window,
    save_intermediate_products=False,
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
    arr_dob1 = ds_dob1.GetRasterBand(1).ReadAsArray()
    arr_dob2 = ds_dob2.GetRasterBand(1).ReadAsArray()
    arr_std1 = ds_std1.GetRasterBand(1).ReadAsArray()
    arr_std2 = ds_std2.GetRasterBand(1).ReadAsArray()

    # --------------------- #

    normprod_smovar = _compute_normprod_smovar_arr(
        arr_dob1,
        arr_dob2,
        arr_std1,
        arr_std2,
        window=window,
        save_intermediate_products=save_intermediate_products,
        intermediate_dir=normprod_smovar_output_path.parent,
    )

    # --------------------- #

    # Write normprod_smovar to disk

    logger.debug("Saving normprod_smovar.")

    driver = gdal.GetDriverByName("GTIFF")
    out_ds = driver.Create(
        normprod_smovar_output_path,
        ds_dob1.RasterXSize,
        ds_dob1.RasterYSize,
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "BIGTIFF=YES"],
    )
    out_ds.SetGeoTransform(ds_dob1.GetGeoTransform())
    out_ds.SetProjection(ds_dob1.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(normprod_smovar)
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
    windows=[11, 21, 33],
    save_intermediate_products=False,
    NP_min=-0.5,
    NP_max=1.0,
    landmask_shapefile_path=None,
    erode_landmask=None,
    resample=True,
    resample_interval=10,
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
    tif_file_list = [f for f in img_pair_dir.glob("*.tif")]

    logger.debug(f"Found {len(tif_file_list)} tif files in img_pair_dir")
    for ii, tif_file in enumerate(tif_file_list):
        logger.debug(f"tif_file {ii+1}: {tif_file}")

    # Find the original georeg files
    # Make sure to exclude previously processed DoB or std images
    exclude_list = ["DoB", "dob", "std"]

    # List the georeg files for the IMG_PAIR_DIR
    georeg_pair = [
        f
        for f in tif_file_list
        if f.name.startswith("georeg")
        and not any(excluded in f.name for excluded in exclude_list)
    ]
    georeg_pair.sort(key=lambda p: p.name)

    logger.info(f"Found {len(georeg_pair)} georeg*tif files in img_pair_dir:")
    for i, georeg_img in enumerate(georeg_pair):
        logger.info(f"georeg_{i+1}: {georeg_img}")

    if not len(georeg_pair) == 2:
        logger.error(
            f"Expected exactly 2 files in georeg_pair, but found {len(georeg_pair)}."
        )
        return False

    # --------------------- #

    logger.info(
        f"Computing DoB, local_std, and normprod_smovar for the following window sizes: {windows}"
    )

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
            save_intermediate_products=save_intermediate_products,
        )

    # --------------------- #

    logger.info("Stacking to false-color RGB")

    if not len(windows) == 3:
        logger.error(
            f"Expected three different window sizes for RGB stack, but len(windows) is {len(windows)}"
        )
        return False

    img1_path = img_pair_dir / f"normprod_smovar_window{windows[0]}.tif"
    img2_path = img_pair_dir / f"normprod_smovar_window{windows[1]}.tif"
    img3_path = img_pair_dir / f"normprod_smovar_window{windows[2]}.tif"
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
        img_min=NP_min,
        img_max=NP_max,
        new_min=0,
        new_max=255,
        overwrite=False,
    )

    if resample:

        logger.info("Resampling RGB image")

        geotiff_path = img_pair_dir / f"normprod_smovar_RGB.tif"
        output_path = (
            img_pair_dir
            / f"normprod_smovar_RGB_resampled_{resample_interval}_{resample_interval}.tif"
        )

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
        output_path = img_pair_dir / f"landmask.tif"

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

            geotiff_path = (
                img_pair_dir
                / f"normprod_smovar_RGB_resampled_{resample_interval}_{resample_interval}.tif"
            )
            output_path = (
                img_pair_dir
                / f"landmask_resampled_{resample_interval}_{resample_interval}.tif"
            )

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
            output_path = img_pair_dir / f"landmask_eroded_{erode_landmask}.tif"

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
                output_path = (
                    img_pair_dir
                    / f"landmask_eroded_{erode_landmask}_resampled_{resample_interval}_{resample_interval}.tif"
                )

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
# xarray NormProd
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #


def compute_DoB_xr(da, window):
    """Compute DoB for an xarray DataArray.

    Parameters
    ----------
    da     : xr.DataArray — single 2-D spatial slice (no time dimension)
    window : boxcar window size

    Returns
    -------
    xr.DataArray with the same coordinates and dimensions as `da`
    """
    logger.info(f"Starting DoB computation (xarray) for w={window}...")
    result = _compute_dob_arr(da.values.astype(np.float32), window)
    return da.copy(data=result)


def compute_local_std_xr(da, window):
    """Compute local standard deviation for an xarray DataArray.

    Parameters
    ----------
    da     : xr.DataArray — single 2-D spatial slice (no time dimension)
    window : boxcar window size

    Returns
    -------
    xr.DataArray with the same coordinates and dimensions as `da`
    """
    logger.info(f"Starting local std computation (xarray) for w={window}...")
    result = _compute_local_std_arr(da.values.astype(np.float32), window)
    return da.copy(data=result)


def compute_normprod_smovar_xr(dob1, dob2, std1, std2, window):
    """Compute normprod_smovar for xarray DataArrays.

    Parameters
    ----------
    dob1, dob2 : xr.DataArray — DoB images for the two acquisitions
    std1, std2 : xr.DataArray — local std images for the two acquisitions
    window     : boxcar window size

    Returns
    -------
    xr.DataArray with the same coordinates and dimensions as `dob1`
    """
    logger.info(f"Starting normprod_smovar computation (xarray) for w={window}")
    result = _compute_normprod_smovar_arr(
        dob1.values.astype(np.float32),
        dob2.values.astype(np.float32),
        std1.values.astype(np.float32),
        std2.values.astype(np.float32),
        window=window,
    )
    return dob1.copy(data=result.astype(np.float32)).assign_attrs({"window": window})


def fully_process_image_pair_xr(
    img1,
    img2,
    windows = [11, 21, 33],
    NP_min = -0.5,
    NP_max = 1.0,
    rgb_min = 0,
    rgb_max = 255,
    resample_rgb = False,
    zoom_x = 1,
    zoom_y = 1,
    resample_method = "linear",
):
    """
    Full NormProd processing for a pair of xarray DataArrays.
    Equivalent to fully_process_single_image_pair but operates entirely
    in-memory — no files are read or written.

    Parameters
    ----------
    img1, img2 : xr.DataArray — single 2-D spatial slices (no time dimension)
    windows    : list of boxcar window sizes; must be length 3 for RGB output
    NP_min     : minimum NormProd value for RGB scaling (default=-0.5)
    NP_max     : maximum NormProd value for RGB scaling (default=1.0)
    rgb_min    : minimum value for RGB image. Defaults to 0.
    rgb_max    : maximum value for RGB image. Defaults to 255.
    resample_rgb : if True, resample the rgb output by zoom_x/zoom_y after
                 scaling. normprod_smovar is left at full resolution
                 (default=False)
    zoom_x     : resampling factor in x-direction for rgb; >1 coarsens
                 resolution (default=1)
    zoom_y     : resampling factor in y-direction for rgb; >1 coarsens
                 resolution (default=1)
    resample_method : interpolation method for resampling rgb, mapped to a
                 scipy.ndimage.zoom order ("nearest"->0, "linear"/"bilinear"->1,
                 "cubic"->3). Default="linear".

    Returns
    -------
    dict with keys:
        "normprod_smovar" : dict[int, xr.DataArray]  — one DataArray per window
        "rgb"             : np.ndarray (H, W, 3) uint8 — false-colour composite
                           (only present when len(windows) == 3). Scaled between 0
                           and 255 by default.
    """

    logger.info(f"Starting full normprod processing chain (xarray) for windows={windows}...")

    normprod_smovar_results = {}

    for window in windows:
        logger.info(f"Window {window}: computing DoB, local std, normprod_smovar...")

        dob1 = compute_DoB_xr(img1, window)
        dob2 = compute_DoB_xr(img2, window)
        std1 = compute_local_std_xr(img1, window)
        std2 = compute_local_std_xr(img2, window)

        normprod_smovar_results[window] = compute_normprod_smovar_xr(
            dob1, dob2, std1, std2, window,
        )

    results = {"normprod_smovar": normprod_smovar_results}

    # False-colour RGB stack (requires exactly 3 windows)
    if len(windows) == 3:
        logger.info("Stacking normprod_smovar to false-colour RGB...")

        def _scale(arr):
            return ((np.clip(arr, NP_min, NP_max) - NP_min) / (NP_max - NP_min) * (rgb_max-rgb_min)).astype(np.uint8)

        results["rgb"] = np.dstack([
            _scale(normprod_smovar_results[windows[0]].values),
            _scale(normprod_smovar_results[windows[1]].values),
            _scale(normprod_smovar_results[windows[2]].values),
        ])

        if resample_rgb and (zoom_x != 1 or zoom_y != 1):
            logger.info(f"Resampling rgb by zoom_x={zoom_x}, zoom_y={zoom_y} (method={resample_method})...")
            order_map = {"nearest": 0, "linear": 1, "bilinear": 1, "cubic": 3}
            order = order_map.get(resample_method, 1)
            rgb_resampled = zoom(results["rgb"], zoom=(1 / zoom_y, 1 / zoom_x, 1), order=order)
            results["rgb"] = np.clip(rgb_resampled, 0, 255).astype(np.uint8)

    else:
        logger.warning(f"Expected 3 windows for RGB stack, got {len(windows)} — skipping RGB.")

    logger.info("Finished full normprod processing chain (xarray).")

    return results


# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <normprod.py> ----
