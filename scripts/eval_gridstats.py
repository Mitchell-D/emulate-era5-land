"""
Methods for interacting with 'timegrid' style HDF5s, which each cover 1/6 of
CONUS over a 3 month period, and store their data as a (T,P,Q,F) dynamic grid
with (P,Q,F) static grids and (T,1) timestamps
"""
import numpy as np
import pickle as pkl
import random as rand
import json
import gc
import h5py
from datetime import datetime
from pathlib import Path

from emulate_era5_land.extract_gridstats import make_gridstat_hdf5
from emulate_era5_land.helpers import sector_slices

if __name__=="__main__":
    data_dir = Path("/rstor/mdodson/era5")
    tg_dir = data_dir.joinpath("timegrids-new")
    #static_pkl_path = data_dir.joinpath("static/nldas_static_cropped.pkl")
    gridstat_dir = data_dir.joinpath("gridstats")

    '''
    ## Generate gridstats file over a single region
    substr = "timegrid_era5"
    timegrids = sorted([p for p in tg_dir.iterdir() if substr in p.name])
    print(timegrids)
    make_gridstat_hdf5(
            timegrids=timegrids,
            out_file=gridstat_dir.joinpath(
                f"gridstats_era5_2012-2023.h5"),
            depermute=True,
            time_sector_size=24*14,
            space_sector_chunks=16,
            nworkers=16,
            debug=True,
            )
    '''

    gs_path = gridstat_dir.joinpath("gridstats_era5_2012-2023.h5")
    F = h5py.File(gs_path, "r")
    dlabels  = json.loads(F["data"].attrs["gridstats"])["flabels"]
    slabels  = json.loads(F["data"].attrs["static"])["flabels"]
    print(dlabels)
    print(slabels)
    slices = sector_slices(
            sector_shape=(100, 24, [5, 7, 9], None),
            bounding_box=(16384, 336, 17, 4),
            iteration_order=(2,0,1),
            separate_sparse=True,
            )
    for s in slices:
        print(s)
