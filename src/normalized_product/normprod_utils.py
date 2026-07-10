# ---- This is <normprod_utils.py> ----

"""
Module with helpers for normprod (normalized product) computation.
Developed as part of the AAPP/UTAS tool for Antarctic fast ice mapping.

Initial developments by A.P. Doulgeris, G. Burke, A. Fraser.

Packaged by J. Lohse.
(johannes.lohse@utas.edu.au)
"""

import pathlib
from loguru import logger

from datetime import datetime

import numpy as np
from scipy.ndimage import zoom, binary_erosion

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from osgeo import gdal, osr

import rioxarray  # registers the .rio accessor on xr.DataArray

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def check_raster_stats(filepath):
    """Prints min, max, mean, and std of a raster file."""

    filepath = pathlib.Path(filepath).resolve()

    if not filepath.is_file():
        logger.error(f"Cannot find filepath: {filepath}.")
        return False
    
    ds = gdal.Open(filepath, gdal.GA_ReadOnly)
    if ds is None:
        logger.error(f"Cannot open filepath: {filepath}.")
        return False
    
    arr = ds.GetRasterBand(1).ReadAsArray()
    print(f"Stats for: {filepath.name}:")
    print(f"    Full path: {filepath}")
    print(f"    Min: {np.nanmin(arr)}")
    print(f"    Max: {np.nanmax(arr)}")
    print(f"    Mean: {np.nanmean(arr)}")
    print(f"    Std Dev: {np.nanstd(arr)}\n")
    ds = None

    return True

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def fill_nans(image):
    """Fills NaNs before filtering to prevent them from growing."""

    nan_mask = np.isnan(image)

    # If no NaNs, return original
    if not np.any(nan_mask):
        return image

    logger.debug("Filling NaNs of input image.")

    # Simple approach: replace NaNs with the local mean of valid pixels
    mean_value = np.nanmean(image)
    image_filled = np.where(nan_mask, mean_value, image)

    return image_filled

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# NEEDED? SAME AS np.nanmean ?
# Let's keep it for now, it works and is probably not the bottleneck for processing

def nan_safe_mean_filter(values):
    """Computes the mean of non-NaN values."""

    valid_values = values[~np.isnan(values)]

    return np.nan if valid_values.size == 0 else np.mean(valid_values)

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# VERY SPECIFIC TO FILENAME CONVENTION, WILL BREAK AT THE SLIGHTEST CHANGE
# SHOULD IDEALLY NOT BE PART OF THE PACKAGE
# IMPLEMENTED ALTERNATIVE 'date1', 'date2' inputs in 'check_and_trim_image_pair'

def extract_date_from_filename(filename):
    """Convert date from GA S1 filename to datetime.datetime"""

    # Extract datestring from filename
    datestring = filename[12:27]

    date = extract_date_from_datestring(datestring)

    return date

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def extract_date_from_datestring(datestring):
    """Convert datestring (YYYYMMDDThhmmss) to datetime.datetime"""

    logger.debug(f"datestring: {datestring}")

    # Convert to datetime
    try:
        date = datetime.strptime(datestring, "%Y%m%dT%H%M%S")
    except ValueError:
        logger.error(f"Extracted datestring does not match the expected format.")
        date = None

    return date

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def get_valid_data_extent(ds):
    """Compute the bounding box of the valid (non-NaN) data in a GDAL dataset."""

    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    
    arr = band.ReadAsArray()
    
    # Create a valid mask: Exclude nodata and NaN values
    if nodata is not None:
        mask = (arr != nodata) & ~np.isnan(arr)
    else:
        mask = ~np.isnan(arr)  # Assuming float data with NaNs

    if not np.any(mask):  # No valid data
        return None

    # Get min/max row/col indices where data is valid
    row_indices = np.where(np.any(mask, axis=1))[0]
    col_indices = np.where(np.any(mask, axis=0))[0]

    if row_indices.size == 0 or col_indices.size == 0:
        return None  # No valid pixels found

    min_row, max_row = row_indices[0], row_indices[-1]
    min_col, max_col = col_indices[0], col_indices[-1]

    # Convert pixel coordinates to geographic coordinates
    geotransform = ds.GetGeoTransform()
    xRes, yRes = abs(geotransform[1]), abs(geotransform[5])  # Pixel size
    min_x = geotransform[0] + geotransform[1] * min_col
    max_x = geotransform[0] + geotransform[1] * (max_col + 1)
    max_y = geotransform[3] + geotransform[5] * min_row
    min_y = geotransform[3] + geotransform[5] * (max_row + 1)

    return (min_x, min_y, max_x, max_y, xRes, yRes)

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def check_and_trim_image_pair(
    img_pair,
    output_dir,
    min_temp_baseline = 11.9,
    max_temp_baseline = 12.1,
    output_epsg = 3031,
    date1 = None,
    date2 = None,
    overwrite = False
):
    """
    Check input image pair for spatial overlap and temporal baseline requirements.
    Trim to common valid overlap region and write resulting geotiff images to output dir.

    Parameters
    ----------
    img_pair : Image pair list with full paths to input images
    output_dir : path to directory where trimmed outputs are stored
    min_temp_baseline : minimum temporal baseline in days (default=11.9)
    min_temp_baseline : maximum temporal baseline in days (default=12.1)
    output_epsg : EPSG code for output image projection (default=3031)
    date1 : manual datestring for first image of pair (default=None)
    date2 : manual datestring for second image of pair (default=None)
    overwrite : overwrite previous existing results (default=False)

    Returns
    -------
    """

    logger.info("Starting to check and trim input image pair...")

    # Check input image pair
    if not isinstance(img_pair, list):
        logger.error(f"Expected input img_pair of type 'list' but received {type(image_pair).__name__}.")
        return False

    if not len(img_pair)==2:
        logger.error(f"Input img_pair list must have exactly 2 entries.")
        return False

    # Set inddividual paths to both input images
    img_path_1 = pathlib.Path(img_pair[0]).resolve()
    img_path_2 = pathlib.Path(img_pair[1]).resolve()

    logger.debug(f"img_path_1: {img_path_1}")
    logger.debug(f"img_path_2: {img_path_2}")

    # Check that both image files exist
    if not img_path_1.is_file():
        logger.error(f"Could not find first input image: {img_path_1}")
        return False

    if not img_path_2.is_file():
        logger.error(f"Could not find second input image: {img_path_2}")
        return False

    # --------------------- #

    # Set/extract dates
    # 'extract_date' implemented in this module is VERY specific to GA naming convention
    # If this fails, 'date1' and 'date2' can be defined as external input parameters

    if date1==None:
        date1 = extract_date_from_filename(img_path_1.name)
    else:
        date1 = extract_date_from_datestring(date1)

    if date2==None:
        date2 = extract_date_from_filename(img_path_2.name)
    else:
        date2 = extract_date_from_datestring(date2)

    logger.debug(f"date1: {date1}")
    logger.debug(f"date2: {date2}")

    if date1==None or date2==None:
        logger.error("Could not succesfully convert dates to datetime objects.")
        logger.error("Check filename convenction or provide datestring as 'YYYYMMDDThhmmss'.")
        return False	

    # Calculate temporal baseline in days
    temp_baseline = date2 - date1
    temp_baseline_days = np.abs(temp_baseline.total_seconds()) / 86400

    logger.debug(f"temp_baseline_days: {temp_baseline_days}")

    if not (min_temp_baseline <= temp_baseline_days <= max_temp_baseline):
        logger.info(f"skipping this pair because temporal baseline is outside of defined range.")
        return False

    # --------------------- #

    # Ensure that output_dir exists
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build full output paths
    georeg1_path = output_dir / f"georeg_1_{date1.strftime('%Y%m%d')}_EPSG{output_epsg}.tif"
    georeg2_path = output_dir / f"georeg_2_{date2.strftime('%Y%m%d')}_EPSG{output_epsg}.tif"

    logger.debug(f"georeg1_path: {georeg1_path}")
    logger.debug(f"georeg2_path: {georeg2_path}")

    if georeg1_path.is_file() and georeg2_path.is_file() and not overwrite:
        logger.info(f"Trimmed output files already exist. Set 'overwrite' option to reprocess.")
        return True

    elif georeg1_path.is_file() or georeg2_path.is_file() and overwrite:
        logger.info(f"Removing existing output files before reprocessing")
        georeg1_path.unlink(missing_ok=True)
        georeg1_path.unlink(missing_ok=True)

    # --------------------- #

    logger.info("Passed initial checks, reading data and checking valid extent...")

    # Open data sets
    DS1 = gdal.Open(img_path_1, gdal.GA_ReadOnly)
    DS2 = gdal.Open(img_path_2, gdal.GA_ReadOnly)

    # Get image extent
    extent1 = get_valid_data_extent(DS1)
    extent2 = get_valid_data_extent(DS2)

    logger.debug(f"extent1: {extent1}")
    logger.debug(f"extent2: {extent2}")

    # Compute the intersection of the extents
    if extent1 and extent2:
        min_x = max(extent1[0], extent2[0])  # Max of min_x
        min_y = max(extent1[1], extent2[1])  # Max of min_y
        max_x = min(extent1[2], extent2[2])  # Min of max_x
        max_y = min(extent1[3], extent2[3])  # Min of max_y
        xRes = min(extent1[4], extent2[4])   # Use the finer resolution
        yRes = min(extent1[5], extent2[5])  

        # Check if valid intersection exists
        if min_x >= max_x or min_y >= max_y:
            logger.info(f"Skipping this pair because of no overlapping valid region.")
            return False

    else:
        logger.info(f"Skipping this pair because one image does not contain valid data.")
        return False

   # --------------------- #

    logger.info("Warping both images to common projection and overlapping footprint...")

    # Specify output spatial reference system    
    Psrs = osr.SpatialReference()
    Psrs.ImportFromEPSG(output_epsg)
    logger.debug(f"Initialized spatial reference system: EPSG {Psrs.GetAuthorityCode(None)}, {Psrs.GetName()}")

    # Gdalwarp the input bands to the new projection and extent
    gdal.Warp(
        georeg1_path,
        DS1,
        format = "GTiff",
        dstSRS = Psrs,
        outputBounds = (min_x, min_y, max_x, max_y), 
        xRes = xRes,
        yRes = yRes,
        dstNodata = np.nan,
        outputType=gdal.GDT_Float32,
        creationOptions=["COMPRESS=DEFLATE", "BIGTIFF=YES"]
    )

    # Gdalwarp the input bands to the new projection and extent
    gdal.Warp(
        georeg2_path,
        DS2,
        format = "GTiff",
        dstSRS = Psrs,
        outputBounds = (min_x, min_y, max_x, max_y), 
        xRes = xRes,
        yRes = yRes,
        dstNodata = np.nan,
        outputType=gdal.GDT_Float32,
        creationOptions=["COMPRESS=DEFLATE", "BIGTIFF=YES"]
    )

    # Clean up
    DS1, DS2 = None, None

    # --------------------- #

    logger.info("Trimming reprojected images and to common valid points...")

    # Read in warped images
    DS1 = gdal.Open(georeg1_path, gdal.GA_Update)
    DS2 = gdal.Open(georeg2_path, gdal.GA_Update)
    Dat1, Dat2 = DS1.ReadAsArray(), DS2.ReadAsArray()

    # Get number of bands and expand dims if just one band
    NBnds1, NBnds2 = DS1.RasterCount, DS2.RasterCount
    if NBnds1 == 1:
        Dat1 = np.expand_dims(Dat1, 0)
    if NBnds2 == 1:
        Dat2 = np.expand_dims(Dat2, 0)

    # Find all indices that are valid in both images
    Intersection = ~np.isnan(Dat1[0]) & ~np.isnan(Dat2[0])

    # Check all bands in both images
    # Not tested - GA images only contain one single band per file
    if NBnds1 > 1:
        Intersection &= ~np.isnan(Dat1[1]) & ~np.isnan(Dat2[1])

    # Set all non-valid data points in all bands to np.nan
    for di in range(NBnds1):
        Dat1[di][~Intersection] = np.nan
        Bnd1 = DS1.GetRasterBand(di+1)
        Bnd1.WriteArray(np.float32(Dat1[di]))
        Bnd1.SetNoDataValue(np.nan)
        Bnd1.FlushCache()

    # Set all non-valid data points in all bands to np.nan
    for di in range(NBnds2):
        Dat2[di][~Intersection] = np.nan
        Bnd2 = DS2.GetRasterBand(di+1)
        Bnd2.WriteArray(np.float32(Dat2[di]))
        Bnd2.SetNoDataValue(np.nan)
        Bnd2.FlushCache()

    # Clean up
    DS1.FlushCache()
    DS1 = None
    DS2.FlushCache()
    DS2 = None

    return True

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def stack_2_RGB(
    img1_path,
    img2_path,
    img3_path,
    output_path,
    img_min = -0.5,
    img_max = 1.0,
    new_min = 0,
    new_max = 255,
    overwrite = False
):
    """
    Stack input images to 8-bit integer RGB image.

    Parameters
    ----------
    img1_path : Path to 1st input image, used for red channel
    img2_path : Path to 2nd input image, used for green channel
    img3_path : Path to 3rd input image, used for blue channel
    output_path : Path to output RGB image
    img_min : Minimum value for input images (default=-0.5 for normprod_smovar)
    img_max : Maximum value for input images (default=1.0 normprod_smovar)
    newMin : New minimum for RGB image channels (default=0)
    newMax : New maximum for RGB image channels (default=255)
    overwrite : Overwrite previous existing results (default=False)

    Returns
    -------
    """

    logger.info("Starting to stack images to RGB...")

    img1_path = pathlib.Path(img1_path).resolve()
    img2_path = pathlib.Path(img2_path).resolve()
    img3_path = pathlib.Path(img3_path).resolve()
    output_path = pathlib.Path(output_path).resolve()

    logger.debug(f"img1_path: {img1_path}")
    logger.debug(f"img2_path: {img2_path}")
    logger.debug(f"img3_path: {img3_path}")
    logger.debug(f"output_path: {output_path}")
    
    if not img1_path.is_file():
        logger.error(f"Could not find first input image: {img1_path}")
        return False

    if not img2_path.is_file():
        logger.error(f"Could not find second input image: {img2_path}")
        return False

    if not img3_path.is_file():
        logger.error(f"Could not find third input image: {img3_path}")
        return False

    if output_path.is_file() and not overwrite:
        logger.info(f"RGB output file already exists: {output_path}")
        return True

    # --------------------- #

    # Read input images
    img1 = gdal.Open(img1_path).ReadAsArray()
    img2 = gdal.Open(img2_path).ReadAsArray()
    img3 = gdal.Open(img3_path).ReadAsArray()

    # Clean nan and +/-inf pixels
    img1_clean = np.nan_to_num(img1, nan=img_min, posinf=img_min, neginf=img_min)
    img2_clean = np.nan_to_num(img2, nan=img_min, posinf=img_min, neginf=img_min)
    img3_clean = np.nan_to_num(img3, nan=img_min, posinf=img_min, neginf=img_min)

    # Clip to min/max
    img1_clip = np.clip(img1_clean, img_min, img_max)
    img2_clip = np.clip(img2_clean, img_min, img_max)
    img3_clip = np.clip(img3_clean, img_min, img_max)

    # Normalize
    img1_norm = ((img1_clip - img_min) / (img_max - img_min) * (new_max-new_min)).astype(np.uint8)
    img2_norm = ((img2_clip - img_min) / (img_max - img_min) * (new_max-new_min)).astype(np.uint8)
    img3_norm = ((img3_clip - img_min) / (img_max - img_min) * (new_max-new_min)).astype(np.uint8)

    logger.debug("Finished normalization of input images")

    # Stack channels into RGB image
    img_rgb = np.dstack((img1_norm, img2_norm, img3_norm))

    logger.debug("Stacked images to RGB")

    # --------------------- #

    # Get meta-data from first input image
    ds = gdal.Open(img1_path, gdal.GA_ReadOnly)

    # Save RGB to disk
    driver = gdal.GetDriverByName("GTIFF")
    out_ds = driver.Create(output_path, ds.RasterXSize, ds.RasterYSize, 3, gdal.GDT_Byte, options=["COMPRESS=DEFLATE", "BIGTIFF=YES"])
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(img_rgb[:,:,0])
    out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    out_ds.GetRasterBand(2).WriteArray(img_rgb[:,:,1])
    out_ds.GetRasterBand(2).SetNoDataValue(np.nan)
    out_ds.GetRasterBand(3).WriteArray(img_rgb[:,:,2])
    out_ds.GetRasterBand(3).SetNoDataValue(np.nan)
    out_ds.FlushCache()

    # Clean up
    out_ds = None
    ds = None

    logger.info(f"Saved RGB image: {output_path}")

    return True

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def resample_geotiff(
    geotiff_path,
    output_path,
    zoom_x = 10,
    zoom_y = 10,
    order = 1,
    overwrite = False
):
    """
    Resample geotiff image and write output image to disk.

    Parameters
    ----------
    geotiff_path : Path to input geotiff file
    output_path : Path to resampled output file
    zoom_x : Resampling factor in x-direction (default=10)
    zoom_y : Resampling factor in y-direction (default=10)
    order : Order for interpolation  with scipy.ndimage.zoom (default=1)

    overwrite : Overwrite previous existing results (default=False)

    Returns
    -------
    """

    logger.info("Starting resampling of geotiff file...")

    geotiff_path = pathlib.Path(geotiff_path).resolve()
    output_path  = pathlib.Path(output_path).resolve()

    logger.debug(f"geotiff_path: {geotiff_path}")
    logger.debug(f"output_path:  {output_path}")
    
    if not geotiff_path.is_file():
        logger.error(f"Could not find geotiff_path: {geotiff_path}")
        return False

    if output_path.is_file() and not overwrite:
        logger.info(f"RGB output file already exists: {output_path}")
        return True

    # --------------------- #

    # Read and adjust the projection

    # Open the original file
    with gdal.Open(geotiff_path) as ds_orig:

        # Get geoinfo and number of bands
        n_bands = ds_orig.RasterCount
        gt = ds_orig.GetGeoTransform()
        proj = ds_orig.GetProjection()

        # Get data type from first band
        dtype_code = ds_orig.GetRasterBand(1).DataType
        dtype_name = gdal.GetDataTypeName(dtype_code)

        # Read data
        data = ds_orig.ReadAsArray()

    # Calculate the new GeoTransform
    # gt[1] is pixel width, gt[5] is pixel height
    new_gt = (
        gt[0],           # Top-left X remains the same
        gt[1] * zoom_x,  # Pixel width becomes 10x larger
        gt[2],           # Rotation 1
        gt[3],           # Top-left Y remains the same
        gt[4],           # Rotation 2
        gt[5] * zoom_y   # Pixel height becomes 10x larger
    )

    logger.debug(f"number of bands:       {n_bands}")
    logger.debug(f"dtype_code:            {dtype_code}")
    logger.debug(f"dtype_name:            {dtype_name}")
    logger.debug(f"original projection:   {proj}")
    logger.debug(f"original geotransform: {gt}")
    logger.debug(f"updated geotransform:  {new_gt}")

    # --------------------- #

    # Fix the zoom factors based on the number of input bands

    logger.debug(f"zoom_x: {zoom_x}")
    logger.debug(f"zoom_x: {zoom_x}")

    if n_bands==1 and len(data.shape)==2:

        logger.debug("Input geotiff only has 1 band, setting zoom_factors accordingly")
        zoom_factors = (1/zoom_x, 1/zoom_y)

    elif n_bands>1 and len(data.shape)==3:

        logger.debug("Input geotiff only has more than1 band, setting zoom_factors accordingly")
        zoom_factors = (1/zoom_x, 1/zoom_y, 1)

        logger.debug("Transposing image dimensions to (Nx, Ny, N_bands)")
        data = data.transpose(1,2,0)

    else:

        logger.error("Extracted number of bands and shape of read data do not match")
        #return False 
   
    logger.debug(f"data.shape: {data.shape}")
    logger.debug(f"zoom_factors: {zoom_factors}")

    # --------------------- #

    # Do the resampling
    data_resampled = zoom(data, zoom=zoom_factors, order=order)

    # --------------------- #

    # Create the new file

    rows, cols = data_resampled.shape[0:2]
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_path, cols, rows, n_bands, dtype_code)

    out_ds.SetGeoTransform(new_gt)
    out_ds.SetProjection(proj)

    if n_bands>1:
        for i in range(n_bands):
            out_ds.GetRasterBand(i + 1).WriteArray(data_resampled[:, :, i])

    elif n_bands==1:
        out_ds.GetRasterBand(1).WriteArray(data_resampled)

    # Clean up
    out_ds.FlushCache()
    out_ds = None

    return True

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def erode_coastline(landmask_raster, erode_pixels):
    """
    Shrink land inward from the coastline by `erode_pixels`.
    Interior land (far from any water) is untouched; the raster's own
    edges are treated as land (not water) so tile-boundary cropping
    doesn't get mistaken for a real coastline.
    """
    if not erode_pixels:
        return landmask_raster
    structure = np.ones((3, 3), dtype=bool)  # 8-connected
    eroded = binary_erosion(
        landmask_raster.astype(bool),
        structure=structure,
        iterations=erode_pixels,
        border_value=1,  # treat outside-the-array as land, not water
    )
    return eroded.astype(np.uint8)

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def rasterize_landmask_4_geotiff(
    geotiff_path,
    landmask_shapefile_path,
    erode_landmask=None,
):
    """
    Rasterize a shapefile landmask onto a geotiff raster

    Parameters
    ----------
    geotiff_path : Path to input geotiff raster file
    landmask_shapefile_path : Path to input landmask shapefile
    erode_landmask : Erode landmask by number of pixels (default=None)

    Returns
    -------
    landmask_raster : Array with rasterized landmask
    """

    logger.info("Starting to rasterize landmask...")

    geotiff_path             = pathlib.Path(geotiff_path).resolve()
    landmask_shapefile_path  = pathlib.Path(landmask_shapefile_path).resolve()

    logger.debug(f"geotiff_path:             {geotiff_path}")
    logger.debug(f"landmask_shapefile_path:  {landmask_shapefile_path}")
    
    if not geotiff_path.is_file():
        logger.error(f"Could not find geotiff_path: {geotiff_path}")
        return False

    if not landmask_shapefile_path.is_file():
        logger.error(f"Could not find landmask_shapefile_path: {landmask_shapefile_path}")
        return False

    # --------------------- #

    # Read the landmask shapefile and create the generator for burning
    landmask_gdf = gpd.read_file(landmask_shapefile_path)
    shapes = ((geom, 1) for geom in landmask_gdf.geometry)

    # Read required meta data from geotiff
    with rasterio.open(geotiff_path) as src:
        bounds = src.bounds
        xmin, ymin, xmax, ymax = bounds.left, bounds.bottom, bounds.right, bounds.top
        width = src.width
        height = src.height
        transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, width, height)

    logger.debug(f"bounds:    {bounds}")
    logger.debug(f"width:     {width}")
    logger.debug(f"height:    {height}")
    logger.debug(f"transform: {transform}")

    # Burn the landmask onto the raster
    landmask_raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        default_value=1,
        dtype="uint8"
    )

    logger.info("Finished landmask raster")

    # --------------------- #

    if erode_landmask == None:
        return landmask_raster

    else:

        logger.info(f"Eroding landmask: {erode_landmask} pixels")
        eroded_landmask_raster = erode_coastline(landmask_raster, erode_pixels=erode_landmask)

    return eroded_landmask_raster

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def save_landmask_file_4_geotiff(
    geotiff_path,
    landmask_shapefile_path,
    output_path,
    erode_landmask=None,
    overwrite=False
):
    """
    Rasterize a shapefile landmask onto a geotiff raster and save as new geotiff

    Parameters
    ----------
    geotiff_path : Path to input geotiff raster file
    landmask_shapefile_path : Path to input landmask shapefile
    output_path : Path to landmask output file
    erode_landmask : Erode landmask by number of pixels (default=None)

    Returns
    -------
    """

    geotiff_path             = pathlib.Path(geotiff_path).resolve()
    landmask_shapefile_path  = pathlib.Path(landmask_shapefile_path).resolve()
    output_path              = pathlib.Path(output_path).resolve()

    logger.debug(f"geotiff_path:             {geotiff_path}")
    logger.debug(f"landmask_shapefile_path:  {landmask_shapefile_path}")
    logger.debug(f"output_path:              {output_path}")
    
    if not geotiff_path.is_file():
        logger.error(f"Could not find geotiff_path: {geotiff_path}")
        return False

    if not landmask_shapefile_path.is_file():
        logger.error(f"Could not find landmask_shapefile_path: {landmask_shapefile_path}")
        return False

    if output_path.is_file() and not overwrite:
        logger.info(f"Output file already exists: {output_path}")
        return True

    # --------------------- #

    # create the landmask raster
    landmask = rasterize_landmask_4_geotiff(
        geotiff_path,
        landmask_shapefile_path,
        erode_landmask=erode_landmask
    )

    # --------------------- #

    # Get meta-data from first input image
    with gdal.Open(geotiff_path, gdal.GA_ReadOnly) as ds:
        Nx   = ds.RasterXSize
        Ny   = ds.RasterYSize
        gt   = ds.GetGeoTransform()
        proj = ds.GetProjection()

    # Save RGB to disk
    driver = gdal.GetDriverByName("GTIFF")
    out_ds = driver.Create(output_path, Nx, Ny, 1, gdal.GDT_Byte, options=["COMPRESS=DEFLATE", "BIGTIFF=YES"])
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    out_ds.GetRasterBand(1).WriteArray(landmask)
    out_ds.FlushCache()

    # Clean up
    out_ds = None
    ds = None

    logger.info(f"Saved landmask: {output_path}")

    return True

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def rasterize_landmask_4_xr(
    da,
    landmask_shapefile_path,
    erode_landmask=None,
):
    """
    Rasterize a shapefile landmask onto the grid of an xarray DataArray.

    Parameters
    ----------
    da : xr.DataArray with geospatial metadata (CRS + transform), e.g. as
         attached by rioxarray or odc-stac
    landmask_shapefile_path : Path to input landmask shapefile
    erode_landmask : Erode landmask by number of pixels (default=None)

    Returns
    -------
    landmask_raster : Array with rasterized landmask 
    (1 = land, 0 = water).
    """

    logger.info("Starting to rasterize landmask...")

    landmask_shapefile_path = pathlib.Path(landmask_shapefile_path).resolve()

    logger.debug(f"landmask_shapefile_path:  {landmask_shapefile_path}")

    if not landmask_shapefile_path.is_file():
        logger.error(f"Could not find landmask_shapefile_path: {landmask_shapefile_path}")
        return False

    # --------------------- #

    # Read the landmask shapefile and reproject to match the DataArray's CRS
    landmask_gdf = gpd.read_file(landmask_shapefile_path)

    da_crs = da.rio.crs
    if da_crs is not None and landmask_gdf.crs is not None and landmask_gdf.crs != da_crs:
        landmask_gdf = landmask_gdf.to_crs(da_crs)

    shapes = ((geom, 1) for geom in landmask_gdf.geometry)

    # Read required geospatial meta data from the DataArray
    height, width = da.rio.shape
    transform = da.rio.transform()

    logger.debug(f"width:     {width}")
    logger.debug(f"height:    {height}")
    logger.debug(f"transform: {transform}")

    # Burn the landmask onto the raster
    # (1 = land, 0 = water).
    landmask_raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        default_value=1,
        dtype="uint8"
    )

    logger.info("Finished landmask raster")

    # --------------------- #

    if erode_landmask == None:
        return landmask_raster
    else:
        logger.info(f"Eroding landmask: {erode_landmask} pixels")
        eroded_landmask_raster = erode_coastline(landmask_raster, erode_pixels=erode_landmask)

    return eroded_landmask_raster

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <normprod_utils.py> ----
