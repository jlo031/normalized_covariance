# normalized_covariance

Python library for computation of the **normalized covariance** (NormCoVar) for automated mapping of landfast sea ice around Antarctica in Sentinel-1 SAR imagery.

Original development was conducted in collaboration with the ***University of Tasmania***, ***UiT The Arctic University of Norway***, and ***Geoscience Australia***, as part of the ***Australian Antarctic Program Partnership ([AAPP])***.

---

## 📦 Installation

This library requires the **Geospatial Data Abstraction Layer ([GDAL])** library.
As such, this library requires the use of Conda or Pixi for installation.

### Install with Conda or Mamba

Donwload the `environment.yaml` file and create an environment from it:

```bash
conda env create -f environment.yaml
conda activate NORMCOVAR
```

Then install the package from GitHub:

```bash
pip install git+https://github.com/jlo031/normalized_covariance.git
```

---

### Install with Pixi

Create a new Pixi project and add the package directly from GitHub:

```bash
mkdir normalized-covariance
cd normalized-covariance

pixi init

pixi add gdal
pixi add rasterio
pixi add --pypi "normalized-covariance @ git+https://github.com/jlo031/normalized_covariance.git"
```

Pixi will create an environment containing all required dependencies and install the package from GitHub.

---

## 🚀 Usage

Usage examples are provided in the `examples/` folder.

* **examples/process_single_img_pair_locally.py:** Script that runs the entire processing chain for a single image pair from local GeoTIFF files.
* **examples/normprod_from_stac_xarray.ipynb:** Notebook that runs the full NormCoVar pipeline entirely in-memory on xarray DataArrays queried from STAC — no intermediate files are written.
* **examples/hpc_support:** Examples subfolder with setup for distributed batch processing, specifically designed for the **NCI/GADI** supercomputing environment.

---

## 📊 Batch Processing

An example setup for HPC batch processing is provided in `examples/hpc_support/`.

For full batch processing of a complete test site, users only need to adjust the `config.yaml` file located in the `config/` folder.

The entire batch processing chain consists of two steps:
1.  `preprocess_full_test_site.py`: Handles initial data preparation, georegistration, checking of image pairs, and cropping to overlapping regions.
2.  `batch_process_normcovar.py`: Computes the normalized covariance for each valid image pair.

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
│   │   └── normCoVar__window_*tif
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

---

## 💻 For Developers

Python packages are challenging! 
We have put some thought into how we manage them for both development and general use.

For developers, we have picked [`pixi`](https://pixi.sh/latest/).
This is because:
* It allows us to keep track of explicit python dependencies from both conda and pypi using a single `pyproject.toml` file.
* It keeps a [lock file](https://pixi.sh/latest/workspace/lockfile/) that is always up-to-date, allowing for reproducible environments.
* It allows us to keep packages needed for development in their own [environment](https://pixi.sh/latest/workspace/environment/).
* It allows us to define useful [tasks](https://pixi.sh/latest/workspace/advanced_tasks/) (similar to a Makefile) all within the `pyproject.toml` file.

### Install pixi

Follow the [pixi installation guide](https://pixi.sh/latest/#installation).

### Install pixi environments
Environments are associated with the project.

* The `default` environment contains packages required for the code base (e.g. gdal, rasterio).

`cd` to the repository folder and install the environments:

To install both environments, run
```bash
pixi install --all
```

### Adding a package

We recommend using [`pixi add`](https://pixi.sh/latest/reference/cli/pixi/add/) because this will automatically update the lock file (`pixi.lock`).

#### From Pypi
Preference should be made to install packages from PyPi if they are available.
This is likely for common python packages.

To install or update a package from Pypi, run `pixi add --pypi <package-name>`

To remove a package from Pypi, run `pixi remove --pypi <package-name>`

#### From Conda
If the package is not available on PyPi, conda should be used.

To install a package from Conda, run `pixi add <package-name>`

Pixi defaults to using the `conda-forge` channel.
To add other channels, see [`pixi workspace channel`](https://pixi.sh/latest/reference/cli/pixi/workspace/channel/).

#### Directly editing the pyproject.toml file
You can manually add packages by adding them to the appropriate section of the `pyproject.toml` file:
* `[tool.pixi.dependencies]` for Conda
* `dependencies` for pip

However, this will not automatically update the `pixi.lock` file, so is not recommended.

### Tidying up the pyproject.toml file
After adding a package, it is worth doing a little extra work to make sure the `pyproject.toml` file is nicely formatted:

1. check the versions that were installed using `pixi list -x` (this shows the versions of packages explicitly listed in `pyproject.toml`)

1. Check and manually update the versions in the `pyproject.toml` if required (remove upper limits from conda packages, add versions for pypi packages)

### Running utility tasks

The following utility tasks can be run.

* `pixi run export-conda` -> Export the `default` pixi environment as a conda environment.yaml file
