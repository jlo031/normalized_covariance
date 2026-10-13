# ---- This is <normcovar.py> ----

"""
Module for NormCoVar computation.
Developed as part of the AAPP/UTAS tool for Antarctic fast ice mapping.

Initial developments by A.P. Doulgeris, G. Burke, A. Fraser.

Packaged by J. Lohse, A. Bradley, C. Adams.
(johannes.lohse@utas.edu.au)
"""

import pathlib
from loguru import logger

import numpy as np
from scipy.ndimage import uniform_filter, zoom

from osgeo import gdal

import xarray as xr

from odc.geo.xr import xr_reproject

from normalized_covariance import normcovar_utils
from normalized_covariance.xarray_utils import zoomed_geobox

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# numpy array based functions for both file based and xarray processing
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def _compute_deviation_from_local_mean_arr(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Compute local mean values of within boxcar window.
    Subtract local_mean from central point of window.
    """
    arr_filled = normcovar_utils.fill_nans(arr)
    local_mean = uniform_filter(arr_filled, size=window, mode="nearest")
    return arr_filled - local_mean


def _compute_local_var_arr(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Compute local var on a 2-D numpy array using E[x²] - (E[x])².
    """
    arr_filled = normcovar_utils.fill_nans(arr)
    local_mean = uniform_filter(arr_filled, size=window, mode="nearest")
    local_mean_sq = uniform_filter(arr_filled**2, size=window, mode="nearest")

    # compute variance and clamp to prevent negative floats
    local_variance = local_mean_sq - local_mean**2
    local_variance_clamped = np.clip(local_variance, a_min=0.0, a_max=None)

    return local_variance_clamped


def _compute_normcovar_arr(
    dev_from_local_mean1: np.ndarray,
    dev_from_local_mean2: np.ndarray,
    local_var1: np.ndarray,
    local_var2: np.ndarray,
    window: int,
    save_intermediate_products: bool = False,
    intermediate_dir: pathlib.Path = None,
) -> np.ndarray:
    """
    Compute normcovar on 2-D numpy arrays.

    Parameters
    ----------
    dev_from_local_mean1, dev_from_local_mean2 : Arrays with deviation from local mean
    local_var1, local_var2 : Local variance arrays for the two images
    window : Boxcar window size
    save_intermediate_products : Write intermediate arrays to disk (requires intermediate_dir).
    intermediate_dir : Directory for intermediate GeoTIFF files (ignored when save_intermediate_products=False).
    """

    def _save(arr, name):
        if not save_intermediate_products or intermediate_dir is None:
            return
        path = intermediate_dir / f"{name}__window{window}.tif"
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


    logger.debug("Computing local_mean_var.")
    local_mean_var = (local_var1 + local_var2) * 0.5
    _save(local_mean_var, "local_mean_variance")

    logger.debug("Filling nans.")
    local_mean_var_filled = normcovar_utils.fill_nans(local_mean_var)

    # Clean up
    var1 = var2 = local_mean_var = None


    logger.debug("Computing smoothed variance.")
    smoothed_local_mean_var = uniform_filter(local_mean_var_filled, size=window, mode="nearest")
    _save(smoothed_local_mean_var, "smoothed_local_mean_variance")

    # Clean up
    mean_var_filled = None


    logger.debug("Computing image covariance.")
    product_of_local_devs = dev_from_local_mean1 * dev_from_local_mean2
    _save(product_of_local_devs, "product_of_local_devs")

    logger.debug("Average product_of_local_devs.")
    covar = uniform_filter(product_of_local_devs, size=window, mode="nearest")
    _save(covar, "covar")

    # Clean up
    product_of_local_devs = None

    return covar / smoothed_local_mean_var

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# File-based NormProd
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def compute_deviation_from_local_mean(image_path, output_path, window):
    """
    Read input image and compute local mean values within boxcar window.
    Subtract local_mean from central point of window and write to output_path.
    """

    logger.info(f"Starting 'deviation_from_local_mean' computation for w={window}...")

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

    band = ds.GetRasterBand(1).ReadAsArray()
    deviation_from_local_mean = _compute_deviation_from_local_mean_arr(band, window)

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
    out_ds.GetRasterBand(1).WriteArray(deviation_from_local_mean)
    out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    out_ds.FlushCache()

    # Clean up
    out_ds = None
    ds     = None

    logger.info(f"Saved 'deviation_from_local_mean' image: {output_path}")

    return output_path

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def compute_local_var(image_path, output_path, window):
    """
    Read input image and compute local variance using E[x²] - (E[x])².
    Write variance image to output_path.
    """

    logger.info(f"Starting 'local_var' computation for w={window}...")

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
    local_var = _compute_local_var_arr(band, window)

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
    out_ds.GetRasterBand(1).WriteArray(local_var)
    out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    out_ds.FlushCache()

    # Clean up
    out_ds = None
    ds     = None

    logger.info(f"Saved 'local_var' image: {output_path}")

    return output_path

# -------------------------------------------------------------------------- #
# --------------------------------------------------------------------------

def compute_normcovar(
    dev_from_local_mean1_path,
    dev_from_local_mean2_path,
    local_var1_path,
    local_var2_path,
    normcovar_output_path,
    window,
    save_intermediate_products=False,
):
    """
    Compute normalised covariance (NormCoVar) using precomputed smoothed images.

    Parameters
    ----------
    dev_from_local_mean1_path : path to 'deviation_from_local_mean' input image 1
    dev_from_local_mean2_path : path to 'deviation_from_local_mean' input image 2
    local_var1_path : path to 'local_variance' input image 1
    local_var2_path : path to 'local_variance' input image 2
    normcovar_output_path : path to output file (normcovar image)
    window : window size for normalized product (e.g. 11, 21, 33)
    save_intermediate_products : save intermediate products as tif files (default=False)
    """

    logger.info(f"Starting 'normcovar' computation for w={window}...")

    logger.debug(f"dev_from_local_mean1_path: {dev_from_local_mean1_path}")
    logger.debug(f"dev_from_local_mean2_path: {dev_from_local_mean2_path}")
    logger.debug(f"local_var1_path: {local_var1_path}")
    logger.debug(f"local_var2_path: {local_var2_path}")

    dev_from_local_mean1_path = pathlib.Path(dev_from_local_mean1_path)
    dev_from_local_mean2_path = pathlib.Path(dev_from_local_mean2_path)
    local_var1_path = pathlib.Path(local_var1_path)
    local_var2_path = pathlib.Path(local_var2_path)
    normcovar_output_path = pathlib.Path(normcovar_output_path)

    if normcovar_output_path.is_file():
        logger.info(f"Skipping, {normcovar_output_path} already exists.")
        return True

    if not dev_from_local_mean1_path.is_file():
        logger.error(f"Could not find dev_from_local_mean1_path: {dev_from_local_mean1_path}.")
        return False

    if not dev_from_local_mean2_path.is_file():
        logger.error(f"Could not find dev_from_local_mean2_path: {dev_from_local_mean2_path}.")
        return False

    if not local_var1_path.is_file():
        logger.error(f"Could not find local_var1_path: {local_var1_path}.")
        return False

    if not local_var2_path.is_file():
        logger.error(f"Could not find local_var2_path: {local_var2_path}.")
        return False

    # --------------------- #

    # Read all input data

    ds_dev_from_local_mean1 = gdal.Open(dev_from_local_mean1_path, gdal.GA_ReadOnly)
    ds_dev_from_local_mean2 = gdal.Open(dev_from_local_mean2_path, gdal.GA_ReadOnly)
    ds_local_var1 = gdal.Open(local_var1_path, gdal.GA_ReadOnly)
    ds_local_var2 = gdal.Open(local_var2_path, gdal.GA_ReadOnly)

    if not all([ds_dev_from_local_mean1, ds_dev_from_local_mean2, ds_local_var1, ds_local_var2]):
        logger.error(f"Could not open all required input files.")
        return False

    logger.debug("Reading input data.")
    dev_from_local_mean1 = ds_dev_from_local_mean1.GetRasterBand(1).ReadAsArray()
    dev_from_local_mean2 = ds_dev_from_local_mean2.GetRasterBand(1).ReadAsArray()
    local_var1 = ds_local_var1.GetRasterBand(1).ReadAsArray()
    local_var2 = ds_local_var2.GetRasterBand(1).ReadAsArray()

    # --------------------- #

    normcovar = _compute_normcovar_arr(
        dev_from_local_mean1,
        dev_from_local_mean2,
        local_var1,
        local_var2,
        window=window,
        save_intermediate_products=save_intermediate_products,
        intermediate_dir=normcovar_output_path.parent,
    )

    # --------------------- #

    # Write normcovar to disk

    logger.debug("Saving normcovar...")

    driver = gdal.GetDriverByName("GTIFF")
    out_ds = driver.Create(
        normcovar_output_path,
        ds_local_var1.RasterXSize,
        ds_local_var1.RasterYSize,
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "BIGTIFF=YES"],
    )
    out_ds.SetGeoTransform(ds_local_var1.GetGeoTransform())
    out_ds.SetProjection(ds_local_var1.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(normcovar)
    out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    out_ds.FlushCache()
    out_ds = None

    logger.info(f"Saved 'normcovar' image: {normcovar_output_path}.")

    # Clean up
    logger.debug("Freeing memory.")
    dev_from_local_mean1 = dev_from_local_mean1 = local_var1 = local_var1 = None
    ds_dev_from_local_mean1 = ds_dev_from_local_mean1 = ds_local_var1 = ds_local_var1 = None
    normcovar = None

    # --------------------- #

    return True

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# Full workflows put together
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
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
    Full NormCoVar processing for single image pair that has already been checked and trimmed.
        - dev_from_local_mean for each image
        - local_var for each image
        - normcovar for image pair
        - stack normcovar to RGB image

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
    exclude_list = ["local_var", "dev_from_local_mean", "local_std"]  ##UPDATE THIS TO DO

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

    logger.info(f"Computing dev_from_local_mean, local_var, and normcovar for the following window sizes: {windows}")

    georeg_path_1 = georeg_pair[0]
    georeg_path_2 = georeg_pair[1]
    georeg_basename_1 = georeg_path_1.stem
    georeg_basename_2 = georeg_path_2.stem

    logger.debug(f"georeg_path_1: {georeg_path_1}")
    logger.debug(f"georeg_path_2: {georeg_path_2}")
    logger.debug(f"georeg_basename_1: {georeg_basename_1}")
    logger.debug(f"georeg_basename_2: {georeg_basename_2}")

    for window in windows:
        logger.info(f"Computing 'dev_from_local_mean' and 'local_var' for window: {window}")

        dev_from_local_mean_path_1 = img_pair_dir / f"{georeg_basename_1}__dev_from_local_mean__window{window}.tif"
        dev_from_local_mean_path_2 = img_pair_dir / f"{georeg_basename_2}__dev_from_local_mean__window{window}.tif"
        local_var_path_1 = img_pair_dir / f"{georeg_basename_1}__local_var__window{window}.tif"
        local_var_path_2 = img_pair_dir / f"{georeg_basename_2}__local_var__window{window}.tif"
        normcovar_path = img_pair_dir / f"normcovar__window{window}.tif"

        logger.debug(f"dev_from_local_mean_path_1: {dev_from_local_mean_path_1}")
        logger.debug(f"dev_from_local_mean_path_2: {dev_from_local_mean_path_2}")
        logger.debug(f"local_var_path_1: {local_var_path_1}")
        logger.debug(f"local_var_path_2: {local_var_path_2}")
        logger.debug(f"normcovar_path: {normcovar_path}")

        compute_deviation_from_local_mean(georeg_path_1, dev_from_local_mean_path_1, window)
        compute_deviation_from_local_mean(georeg_path_2, dev_from_local_mean_path_2, window)

        compute_local_var(georeg_path_1, local_var_path_1, window)
        compute_local_var(georeg_path_2, local_var_path_2, window)

        logger.info(f"Computing 'normcovar' for window: {window}")

        compute_normcovar(
            dev_from_local_mean_path_1,
            dev_from_local_mean_path_2,
            local_var_path_1,
            local_var_path_2,
            normcovar_path,
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

    img1_path = img_pair_dir / f"normcovar__window{windows[0]}.tif"
    img2_path = img_pair_dir / f"normcovar__window{windows[1]}.tif"
    img3_path = img_pair_dir / f"normcovar__window{windows[2]}.tif"
    output_path = img_pair_dir / f"normcovar__RGB.tif"

    logger.debug(f"NP_min:{NP_min}")
    logger.debug(f"NP_min:{NP_max}")
    logger.debug(f"img1_path:{img1_path}")
    logger.debug(f"img2_path:{img2_path}")
    logger.debug(f"img3_path:{img3_path}")

    normcovar_utils.stack_2_RGB(
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

        geotiff_path = img_pair_dir / f"normcovar__RGB.tif"
        output_path = (
            img_pair_dir
            / f"normcovar__RGB__resampled_{resample_interval}_{resample_interval}.tif"
        )

        logger.debug(f"geotiff_path:      {geotiff_path}")
        logger.debug(f"output_path:       {output_path}")
        logger.debug(f"resample_interval: {resample_interval}")

        normcovar_utils.resample_geotiff(
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

        geotiff_path = img_pair_dir / f"normcovar__RGB.tif"
        output_path  = img_pair_dir / f"landmask.tif"

        logger.debug(f"geotiff_path:            {geotiff_path}")
        logger.debug(f"output_path:             {output_path}")
        logger.debug(f"landmask_shapefile_path: {landmask_shapefile_path}")

        normcovar_utils.save_landmask_file_4_geotiff(
            geotiff_path,
            landmask_shapefile_path,
            output_path,
            erode_landmask=None,
        )

        if resample:

            logger.info("Creating resampled landmask image")

            geotiff_path = (
                img_pair_dir
                / f"normcovar__RGB__resampled_{resample_interval}_{resample_interval}.tif"
            )
            output_path = (
                img_pair_dir
                / f"landmask__resampled_{resample_interval}_{resample_interval}.tif"
            )

            logger.debug(f"geotiff_path:      {geotiff_path}")
            logger.debug(f"output_path:       {output_path}")
            logger.debug(f"resample_interval: {resample_interval}")

            normcovar_utils.save_landmask_file_4_geotiff(
                geotiff_path,
                landmask_shapefile_path,
                output_path,
                erode_landmask=None,
            )

        # --------------------- #

        if erode_landmask is not None:

            logger.info("Creating eroded landmask image")

            geotiff_path = img_pair_dir / f"normcovar__RGB.tif"
            output_path  = img_pair_dir / f"landmask__eroded_{erode_landmask}.tif"

            logger.debug(f"geotiff_path:            {geotiff_path}")
            logger.debug(f"output_path:             {output_path}")
            logger.debug(f"landmask_shapefile_path: {landmask_shapefile_path}")

            normcovar_utils.save_landmask_file_4_geotiff(
                geotiff_path,
                landmask_shapefile_path,
                output_path,
                erode_landmask=erode_landmask,
            )

            if resample:

                logger.info("Resampling eroded resampled landmask image")

                geotiff_path = img_pair_dir / f"landmask__eroded_{erode_landmask}.tif"
                output_path = (
                    img_pair_dir
                    / f"landmask__eroded_{erode_landmask}__resampled_{resample_interval}_{resample_interval}.tif"
                )

                logger.debug(f"geotiff_path:      {geotiff_path}")
                logger.debug(f"output_path:       {output_path}")
                logger.debug(f"resample_interval: {resample_interval}")

                normcovar_utils.resample_geotiff(
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
# xarray NormProd
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #


def compute_deviation_from_local_mean_xr(da, window):
    """Compute deviation from local mean value for an xarray DataArray.

    Parameters
    ----------
    da     : xr.DataArray — single 2-D spatial slice (no time dimension)
    window : boxcar window size

    Returns
    -------
    xr.DataArray with the same coordinates and dimensions as `da`
    """
    logger.info(f"Starting 'deviation_from_local_mean' computation (xarray) for w={window}...")
    result = _compute_deviation_from_local_mean_arr(da.values.astype(np.float32), window)
    return da.copy(data=result)


def compute_local_var_xr(da, window):
    """Compute local variance for an xarray DataArray.

    Parameters
    ----------
    da     : xr.DataArray — single 2-D spatial slice (no time dimension)
    window : boxcar window size

    Returns
    -------
    xr.DataArray with the same coordinates and dimensions as `da`
    """
    logger.info(f"Starting local std computation (xarray) for w={window}...")
    result = _compute_local_var_arr(da.values.astype(np.float32), window)
    return da.copy(data=result)


# Should be fine
def compute_landmask_xr(da, landmask_shapefile_path, erode_landmask=None):
    """Compute a landmask for an xarray DataArray by rasterizing a shapefile.

    Parameters
    ----------
    da     : xr.DataArray — single 2-D spatial slice with geospatial metadata
             (CRS + transform, e.g. as attached by rioxarray or odc-stac)
    landmask_shapefile_path : path to landmask shapefile
    erode_landmask : erode landmask by number of pixels (default=None, no erosion)

    Returns
    -------
    xr.DataArray with the same coordinates and dimensions as `da`.
    uint8 landmask (1=land, 0=not land).
    """
    logger.info(f"Starting landmask computation (xarray), erode_landmask={erode_landmask}...")
    result = normcovar_utils.rasterize_landmask_4_xr(
        da, landmask_shapefile_path, erode_landmask=erode_landmask,
    )
    return da.copy(data=result.astype(np.uint8))


def compute_normcovar_xr(dev1, dev2, var1, var2, window):
    """Compute normcovar for xarray DataArrays.

    Parameters
    ----------
    dev1, dev2 : xr.DataArray — deviation from local mean images for the two acquisitions
    var1, var2 : xr.DataArray — local var images for the two acquisitions
    window     : boxcar window size

    Returns
    -------
    xr.DataArray with the same coordinates and dimensions as `dob1`
    """
    logger.info(f"Starting normcovar computation (xarray) for w={window}")
    result = _compute_normcovar_arr(
        dev1.values.astype(np.float32),
        dev2.values.astype(np.float32),
        var1.values.astype(np.float32),
        var2.values.astype(np.float32),
        window=window,
    )
    return dev1.copy(data=result.astype(np.float32)).assign_attrs({"window": window})


def fully_process_image_pair_xr(
    img1,
    img2,
    windows = [11, 21, 33],
    NP_min = -0.5,
    NP_max = 1.0,
    rgb_min = 0,
    rgb_max = 255,
    resample_rgb = True,
    zoom_x = 10,
    zoom_y = 10,
    resample_method = "bilinear",
    landmask_shapefile_path = None,
    erode_landmask = None,
    apply_landmask_to_rgb = False,
):
    """
    Full NormCoVar processing for a pair of xarray DataArrays.
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
                 scaling. normcovar is left at full resolution
                 (default=False)
    zoom_x     : resampling factor in x-direction for rgb; >1 coarsens
                 resolution (default=1)
    zoom_y     : resampling factor in y-direction for rgb; >1 coarsens
                 resolution (default=1)
    resample_method : interpolation method for resampling rgb, one of 
                [“average”, “bilinear”, “cubic”, “cubic_spline”, “lanczos”, 
                “mode”, “gauss”, “max”, “min”, “med”, “q1”, “q3”]. Default="bilinear".
    landmask_shapefile_path : path to landmask shapefile. If given, a landmask
                 is rasterized onto img1's grid (default=None, no landmask).
    erode_landmask : erode landmask by number of pixels (default=None). Only
                 applied if landmask_shapefile_path is also given.
    apply_landmask_to_rgb : apply the mask to the rgb and rgb_resampled image.

    Returns
    -------
    dict with keys (all values are xr.DataArray, preserving spatial coords/CRS):
        "normcovar" : dict[int, xr.DataArray]  — one DataArray per window
        "rgb"             : xr.DataArray (y, x, band) uint8 — false-colour composite
                           (only present when len(windows) == 3). Scaled between 0
                           and 255 by default.
        "rgb_resampled"   : rgb DataArray resampled based on setting (only present
                            when resample_rgb = True.
        "landmask"        : xr.DataArray uint8 — rasterized landmask (only present
                           when landmask_shapefile_path is given)
        "landmask_resampled"   : landmask DataArray resampled based on setting (only present
                            when resample_rgb = True and landmask_shapefile_path
                            is given).
        "rgb_landmasked"  : The rgb DataArray with the landmask applied. (only present
                            when landmask and apply_landmask_to_rgb = True)
        "rgb_resampled_landmasked"  : The rgb resampled DataArray with the landmask applied.
                            (only present when landmask, resample_rgb = True,
                            apply_landmask_to_rgb = True)
    """

    logger.info(f"Starting full normcovar processing chain (xarray) for windows={windows}...")

    normcovar_results = {}

    for window in windows:
        logger.info(f"Window {window}: computing DoB, local std, normcovar...")

        dev1 = compute_deviation_from_local_mean_xr(img1, window)
        dev2 = compute_deviation_from_local_mean_xr(img2, window)
        var1 = compute_local_var_xr(img1, window)
        var2 = compute_local_var_xr(img2, window)

        normcovar_results[window] = compute_normcovar_xr(
            dev1, dev2, var1, var2, window,
        )

    results = {"normcovar": normcovar_results}

    y_dim, x_dim = img1.dims[0], img1.dims[1]

    # False-colour RGB stack (requires exactly 3 windows)
    if len(windows) == 3:
        logger.info("Stacking normcovar to false-colour RGB...")

        def _scale(arr):
            return ((np.clip(arr, NP_min, NP_max) - NP_min) / (NP_max - NP_min) * (rgb_max - rgb_min)).astype(np.uint8)

        rgb_arr = np.dstack([
            _scale(normcovar_results[windows[0]].values),
            _scale(normcovar_results[windows[1]].values),
            _scale(normcovar_results[windows[2]].values),
        ])

        logger.info("Converting rgb to xr.DataArray...")
        results["rgb"] = xr.DataArray(
            rgb_arr,
            dims=(y_dim, x_dim, "band"),
            coords={
                y_dim: img1.coords[y_dim],
                x_dim: img1.coords[x_dim],
                "band": ["R", "G", "B"],
            },
            attrs=img1.attrs,
        )
        if hasattr(img1, "rio"):
            results["rgb"] = results["rgb"].rio.write_crs(img1.rio.crs)

        if resample_rgb and (zoom_x != 1 or zoom_y != 1):
            logger.info(f"Resampling rgb by zoom_x={zoom_x}, zoom_y={zoom_y} (method={resample_method})...")
            results["rgb_resampled"] = xr_reproject(
                results["rgb"],
                how=zoomed_geobox(results["rgb"], zoom_y, zoom_x),
                resampling=resample_method,
                dst_nodata=None,
                dtype=np.uint8,
            )
    else:
        logger.warning(f"Expected 3 windows for RGB stack, got {len(windows)} — skipping RGB.")

    # Landmask
    if landmask_shapefile_path is not None:
        logger.info("Computing landmask...")
        results["landmask"] = compute_landmask_xr(img1, landmask_shapefile_path, erode_landmask=erode_landmask)
        if apply_landmask_to_rgb and "rgb" not in results:
            logger.warning("apply_landmask_to_rgb=True but no 'rgb' was built (windows != 3), skipping rgb_landmasked.")
        if apply_landmask_to_rgb and "rgb" in results:
            rgb_landmasked = results["rgb"].copy()
            rgb_landmasked = rgb_landmasked.where(results["landmask"] != 1, 0).astype(np.uint8)
            results["rgb_landmasked"] = rgb_landmasked
        if resample_rgb and (zoom_x != 1 or zoom_y != 1):
            logger.info(f"Resampling landmask by zoom_x={zoom_x}, zoom_y={zoom_y} (method=nearest)...")
            results["landmask_resampled"] = xr_reproject(
                results["landmask"],
                how=zoomed_geobox(results["landmask"], zoom_y, zoom_x),
                resampling="nearest",  # nearest neighbour for binary landmask
                dst_nodata=None,
                dtype=np.uint8,
            )

            if apply_landmask_to_rgb and "rgb_resampled" in results:
                rgb_resampled_landmasked = results["rgb_resampled"].copy()
                rgb_resampled_landmasked = rgb_resampled_landmasked.where(
                    results["landmask_resampled"] != 1, 0
                ).astype(np.uint8)
                results["rgb_resampled_landmasked"] = rgb_resampled_landmasked

    logger.info("Finished full normprod processing chain (xarray).")

    return results


# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <normcovar.py> ----
