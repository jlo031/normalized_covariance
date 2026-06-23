import os
from setuptools import setup, find_packages

def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()

setup(
    name = "normalized_product",
    version = "0.2.1",
    author = "Johannes Lohse",
    author_email = "johannes.lohse@utas.edu.au",
    description = ("Normprod computation on SAR data for Antarctic fast ice mapping."),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jlo031/normalized_product",
    license = "The Ask Johannes Before You Do Anything License",
    package_dir = {'': 'src'},
    packages = find_packages(where='src'),
    install_requires = [
        'pathlib',
        'loguru',
        'datetime',
        'numpy',
        'scipy',
        'gdal'
    ],
)
