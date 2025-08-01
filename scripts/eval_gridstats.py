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
from time import perf_counter
from pathlib import Path

from emulate_era5_land.extract_gridstats import make_gridstat_hdf5
from emulate_era5_land.helpers import sector_slices
from emulate_era5_land.plotting import geo_quad_plot

if __name__=="__main__":
    data_dir = Path("/rstor/mdodson/era5")
    tg_dir = data_dir.joinpath("timegrids-new")
    #static_pkl_path = data_dir.joinpath("static/nldas_static_cropped.pkl")
    gridstat_dir = data_dir.joinpath("gridstats")
    fig_dir = Path("/rhome/mdodson/emulate-era5-land/figures/gridstats")

    ## contingency since full latlon array should be stored in gridstats and
    ## timegrid files, but currently only valid pixel coords are stored.
    static_pkl = Path("data/static/era5_static.pkl")
    slabels,sdata = pkl.load(static_pkl.open("rb"))
    lat = sdata[slabels.index("lat")]
    lon = sdata[slabels.index("lon")]

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
    mlabels  = json.loads(F["data"].attrs["gridstats"])["mlabels"]
    slabels  = json.loads(F["data"].attrs["static"])["flabels"]
    counts = F["/data/counts"][...]
    gridstats = F["/data/gridstats"]
    static = F["/data/static"][...]
    m_valid = F["/data/mask"][...]
    valid_ixs = np.where(m_valid)
    print(dlabels)
    print(slabels)

    ## generate gridded
    slices = sector_slices(
            #sector_shape=(None,None,16384,None,None),
            sector_shape=(None,None,32768,None,None),
            bounding_box=gridstats.shape,
            iteration_order=None,
            separate_sparse=True,
            )

    ix_valid = np.where(m_valid)
    #latlon = np.full((*m_valid.shape, 2), np.nan)
    #latlon[*ix_valid, 0] = static[...,slabels.index("lat")]
    #latlon[*ix_valid, 1] = static[...,slabels.index("lon")]

    spatial_stats = np.full(gridstats.shape[2:], np.nan)
    for s in slices:
        t0 = perf_counter()
        tmp_gs = gridstats[*s]
        t1 = perf_counter()
        ## global minimum per pixel
        print(s)
        print(tmp_gs.shape, spatial_stats.shape)
        spatial_stats[s[2],...,0] = np.min(tmp_gs[...,0], axis=(0,1))
        ## gloabl maximum per pixel
        spatial_stats[s[2],...,1] = np.max(tmp_gs[...,1], axis=(0,1))
        ## global mean per pixel
        spatial_stats[s[2],...,2] = np.sum(tmp_gs[...,2], axis=(0,1))
        spatial_stats[s[2],...,2] /= np.sum(counts)
        ## global standard deviation per pixel
        spatial_stats[s[2],...,3] = np.sum(tmp_gs[...,3], axis=(0,1)) \
                / np.sum(counts)
        spatial_stats[s[2],...,3] = spatial_stats[s[2],...,3] ** (1/2)
        t2 = perf_counter()
        print(f"{tmp_gs.shape[2]} {t1-t0} {t2-t1}")

    stats_2d = np.full((*m_valid.shape, *spatial_stats.shape[-2:]), np.nan)
    stats_2d[*ix_valid,...] = spatial_stats
    for i,dl in enumerate(dlabels):
        geo_quad_plot(
                data=[stats_2d[:,:,i,j] for j in range(len(mlabels))],
                flabels=[f"{dl} {ml}" for ml in mlabels],
                latitude=lat,
                longitude=lon,
                plot_spec={
                    "title":f"{dl}",
                    "cbar_shrink":.8,
                    "text_size":24,
                    "idx_ticks":False,
                    "show_ticks":False,
                    "cmap":"gnuplot2",
                    "xtick_freq":20,
                    "ytick_freq":20,
                    "figsize":(32,16),
                    "title_fontsize":32,
                    "use_pcolormesh":True,
                    "norm":"linear",
                    },
                show=False,
                fig_path=fig_dir.joinpath(f"{gs_path.stem}_spatial_{dl}.png"),
                )
