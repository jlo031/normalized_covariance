import numpy as np
from scipy.ndimage import uniform_filter, zoom
import xarray as xr
from affine import Affine

from loguru import logger

# Adapted from https://stackoverflow.com/questions/39785970/speckle-lee-filter-in-python
def lee_filter(img, size):
    """
    Applies the Lee filter to reduce speckle noise in an image.

    Parameters:
    img (ndarray): Input image to be filtered.
    size (int): Size of the uniform filter window.

    Returns:
    ndarray: The filtered image.
    """
    img_mean = uniform_filter(img, size)
    img_sqr_mean = uniform_filter(img**2, size)
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = np.var(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output


# Define a function to apply the Lee filter to a DataArray
def apply_lee_filter(data_array, size=7):
    """
    Applies the Lee filter to the provided DataArray.

    Parameters:
    data_array (xarray.DataArray): The data array to be filtered.
    size (int): Size of the uniform filter window. Default is 7.

    Returns:
    xarray.DataArray: The filtered data array.
    """
    data_array_filled = data_array.fillna(0)
    filtered_data = xr.apply_ufunc(
        lee_filter,
        data_array_filled,
        kwargs={"size": size},
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        dask_gufunc_kwargs={"allow_rechunk": True},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[data_array.dtype],
    )
    filtered_data_masked = xr.where(np.isnan(data_array), np.nan, filtered_data)

    return filtered_data_masked

def resample_xr(
    da,
    zoom_y,
    zoom_x,
    order,
    clip_range=None,
    dtype=None,
    trim_for_clean_pixel_size=True,
):
    """
    Resample a georeferenced xr.DataArray and return a new xr.DataArray on
    the resampled grid, with coordinates/CRS/transform derived from the
    input's affine transform and output shape.

    Parameters
    ----------
    da         : xr.DataArray — (y, x) or (y, x, band), with a working
                 .rio accessor (CRS + transform set)
    zoom_y     : zoom factor in y (>1 coarsens resolution)
    zoom_x     : zoom factor in x (>1 coarsens resolution)
    order      : scipy.ndimage.zoom interpolation order (0=nearest, 1=linear, 3=cubic)
    clip_range : optional (min, max) to clip resampled values to, e.g. (0, 255)
    dtype      : output dtype to cast the resampled array to. Defaults to
                 da.dtype (i.e. dtype is preserved) if None.

    Returns
    -------
    xr.DataArray — resampled array with coords/CRS/transform matching
    the new grid, same dims as the input.
    """
    y_dim, x_dim = da.dims[0], da.dims[1]
    has_band = "band" in da.dims

    if dtype is None:
        dtype = da.dtype

    orig_ny, orig_nx = da.sizes[y_dim], da.sizes[x_dim]


    # Crop input so dimensions divide evenly by the zoom factor —
    # guarantees zoom() produces an exact orig/zoom_factor output shape,
    # with no internal rounding, so pixel size stays clean.
    # stops the output xr having size of 399.7364x400.00263 for example
    if trim_for_clean_pixel_size:
        trim_ny = orig_ny - (orig_ny % zoom_y)
        trim_nx = orig_nx - (orig_nx % zoom_x)
        if trim_ny != orig_ny or trim_nx != orig_nx:
            logger.warning(
                f"Trimming {orig_ny - trim_ny} row(s) / {orig_nx - trim_nx} col(s) "
                f"so dimensions divide evenly by zoom factor (clean pixel size)."
            )
            da = da.isel({y_dim: slice(0, trim_ny), x_dim: slice(0, trim_nx)})
            orig_ny, orig_nx = trim_ny, trim_nx

    # Resample
    zoom_factors = (1 / zoom_y, 1 / zoom_x, 1) if has_band else (1 / zoom_y, 1 / zoom_x)
    resampled_arr = zoom(da.values, zoom=zoom_factors, order=order)

    if clip_range is not None:
        resampled_arr = np.clip(resampled_arr, *clip_range)
    resampled_arr = resampled_arr.astype(dtype)

    out_ny, out_nx = resampled_arr.shape[:2]

    # Build a new transform. Scale pixel size (using the output
    # shape, not the nominal zoom factors) so the same ground extent is
    # covered by out_ny x out_nx pixels instead of orig_ny x orig_nx.
    # Affine.scale() composition preserves origin/rotation correctly —
    # avoids manually indexing/reordering transform components.
    src_transform = da.rio.transform()
    x_scale = orig_nx / out_nx
    y_scale = orig_ny / out_ny
    new_transform = src_transform * Affine.scale(x_scale, y_scale)

    # Pixel-center coordinates: (col + 0.5, row + 0.5) -> (x, y).
    # Required as xarray values are centre referenced
    cols = np.arange(out_nx) + 0.5
    rows = np.arange(out_ny) + 0.5
    new_x, _ = new_transform * (cols, np.zeros_like(cols))
    _, new_y = new_transform * (np.zeros_like(rows), rows)

    # Wrap in DataArray, reusing dims/band coords from the input
    if has_band:
        dims = (y_dim, x_dim, "band")
        coords = {y_dim: new_y, x_dim: new_x, "band": da.coords["band"].values}
    else:
        dims = (y_dim, x_dim)
        coords = {y_dim: new_y, x_dim: new_x}

    resampled_da = xr.DataArray(resampled_arr, dims=dims, coords=coords, attrs=da.attrs)
    resampled_da = resampled_da.rio.write_crs(da.rio.crs)
    resampled_da = resampled_da.rio.write_transform(new_transform)

    return resampled_da