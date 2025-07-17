from setuptools import setup, find_packages
from setuptools.extension import Extension

extensions = []
setup(
    name="emulate_era5_land",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    ext_modules=extensions,
    setup_requires=[],
)
