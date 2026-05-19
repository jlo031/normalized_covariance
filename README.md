# normalized_product

Python library for computation of the **normalized product** (normprod) for automated mapping of landfast sea ice around Antarctica in Sentinel-1 SAR imagery.

Original development was conducted in collaboration with the ***University of Tasmania***, ***UiT The Arctic University of Norway***, and ***Geoscience Australia***, as part of the ***Australian Antarctic Program Partnership ([AAPP])***.

---

## 🛠 Preparation

This library requires the **Geospatial Data Abstraction Layer ([GDAL])** library. The simplest way to manage GDAL and Python dependencies is via the **[Anaconda]** distribution.

### Environment Setup
It is recommended the code in a dedicated virtual environment:

    # Create and activate a new environment
    conda create -y --name NORMPROD -c conda-forge gdal rasterio geopandas numpy scipy loguru matplotlib pyyaml
    conda activate NORMPROD

    # Install additional dependencies
    pip install ipython xarray odc-stac odc-stac[botocore] odc-geo stac-geoparquet pystac pystac-client fsspec shapely aiohttp

---

## 📦 Installation

You can install this library directly from GitHub or locally after cloning the repository.

### 1. Installation from GitHub

     # install this package
     pip install git+https://github.com/jlo031/normalized_product

### 2. Local installation

    # clone the repository
     git clone git@github.com:jlo031/normalized_product

 Change into the main directory of the cloned repository (it should contain the *setup.py* file) and install the library:

       # installation
       pip install .

---

## 🚀 Usage

Test scripts and usage examples are provided in the `test/` and `examples/` folders.

* **examples/process_single_img_pair_locally.py:** Script that runs the entire processing chain for a single image pair from local GeoTIFF files.
* **examples/normprod_from_stac_xarray.ipynb:** Notebook that runs the full NormProd pipeline entirely in-memory on xarray DataArrays queried from STAC — no intermediate files are written.
* **examples/hpc_support:** Examples subfolder with setup for distributed batch processing, specifically designed for the **NCI/GADI** supercomputing environment.

Unless you are developing the code further, there is no need to run or modify the contents of the `test/` folder.

---

## 📊 Batch Processing

An example setup for HPC batch processing is provided in `examples/hpc_support/`.

For full batch processing of a complete test site, users only need to adjust the `config.yaml` file located in the `config/` folder.

The entire batch processing chain consists of two steps:
1.  `preprocess_full_test_site.py`: Handles initial data preparation, georegistration, checking of image pairs, and cropping to overlapping regions.
2.  `batch_process_normprod_smovar.py`: Computes the normalized product for each valid image pair.

Both scripts read settings from `config.yaml` and should not require manual code changes. All outputs are written to specific image-pair folders within the test site directory.

### Folder Structure
Your data **must** be organized according to the structure below for the batch scripts to function:

```text
DATA_DIR/
│
├── TestSite1/
│   ├── GA_geotiffs/
│   │   ├── original_GA_intensity_file1.tif    
│   │   ├── original_GA_intensity_file2.tif
│   │   └── ...
│   ├── IMG_PAIR_1/
│   │   ├── georeg_1_*tif
│   │   ├── georeg_1_*tif
│   │   └── normprod_smovar_window_*tif
│   └── ...
│
└── TestSiteN/
    ├── GA_geotiffs/
    │   └── ...
    ├── IMG_PAIR_1/
    │   └── ...
    └── ...
```

- `DATA_DIR/`: The main directory containing all test site subfolders.
- `TestSite1/`, `TestSite2/`, ..., `TestSiteN/`: Subfolders for individual test sites.
- `GA_geotiffs/`: A folder within each test site containing the original GeoTIFF files.
- `IMG_PAIR_1/`, `IMG_PAIR_2/`, ...: Folders for individual image pairs within each test site, containing processed or related files.

[GDAL]: https://gdal.org/
[AAPP]: https://aappartnership.org.au/
[Anaconda]: https://www.anaconda.com/
