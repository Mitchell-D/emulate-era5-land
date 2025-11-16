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
from emulate_era5_land.extract_gridstats import make_gridhist_hdf5
from emulate_era5_land.helpers import sector_slices
from emulate_era5_land.plotting import geo_quad_plot,plot_hists

def get_gridstats_global(gridstat_path):
    F = h5py.File(gs_path, "r")
    dlabels  = json.loads(F["data"].attrs["gridstats"])["flabels"]
    mlabels  = json.loads(F["data"].attrs["gridstats"])["mlabels"]
    slabels  = json.loads(F["data"].attrs["static"])["flabels"]
    counts = F["/data/counts"][...]
    gridstats = F["/data/gridstats"]
    static = F["/data/static"][...]
    m_valid = F["/data/mask"][...]
    latlon = F["/data/latlon"][...]
    valid_ixs = np.where(m_valid)

    ## (month, hour, pixel, feature, metric)
    slices = sector_slices(
            #sector_shape=(None,None,16384,None,None),
            sector_shape=(3,None,sector_size,None,None),
            bounding_box=gridstats.shape,
            iteration_order=None,
            separate_sparse=True,
            )

    ix_valid = np.where(m_valid)


def plot_gridstats_spatial(gridstat_path, fig_dir:Path, plot_spec_per_feat={},
        sector_size=32768, debug=False):
    F = h5py.File(gs_path, "r")
    dlabels  = json.loads(F["data"].attrs["gridstats"])["flabels"]
    mlabels  = json.loads(F["data"].attrs["gridstats"])["mlabels"]
    slabels  = json.loads(F["data"].attrs["static"])["flabels"]
    counts = F["/data/counts"][...]
    gridstats = F["/data/gridstats"]
    static = F["/data/static"][...]
    m_valid = F["/data/mask"][...]
    latlon = F["/data/latlon"][...]
    valid_ixs = np.where(m_valid)

    ## generate gridded wrt (month, tod, pixel, feat, metric)
    slices = sector_slices(
            #sector_shape=(None,None,16384,None,None),
            sector_shape=(None,None,sector_size,None,None),
            bounding_box=gridstats.shape,
            iteration_order=None,
            separate_sparse=True,
            )

    ix_valid = np.where(m_valid)

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
        if debug:
            print(f"{tmp_gs.shape[2]} {t1-t0} {t2-t1}")

    stats_2d = np.full((*m_valid.shape, *spatial_stats.shape[-2:]), np.nan)
    stats_2d[*ix_valid,...] = spatial_stats

    for i,dl in enumerate(dlabels):
        print(f"{dl} min:{np.nanmin(stats_2d[:,:,i,0]):.2f} " + \
                f"max:{np.nanmax(stats_2d[:,:,i,1]):.2f}")
        fig_path = fig_dir.joinpath(f"{gs_path.stem}_spatial_{dl}.png")
        print(f"plotting {dl}")
        geo_quad_plot(
                data=[stats_2d[:,:,i,j] for j in range(len(mlabels))],
                flabels=["minimum","maximum","average","standard deviation"],
                latitude=latlon[...,0],
                longitude=latlon[...,1],
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
                    **plot_spec_per_feat[dl],
                    },
                show=False,
                fig_path=fig_path,
                )
if __name__=="__main__":
    data_dir = Path("/rstor/mdodson/era5")
    tg_dir = data_dir.joinpath("timegrids-new")
    gridstat_dir = data_dir.joinpath("gridstats")
    #fig_dir = Path("/rhome/mdodson/emulate-era5-land/figures/gridstats")
    fig_dir = Path(
            "/rhome/mdodson/emulate-era5-land/figures/gridstats-unbounded")
    '''
    era5_info = json.load(Path("data/list_feats_era5.json").open("r"))
    gs_path = gridstat_dir.joinpath("gridstats_era5_2012-2023.h5")
    gh_path = gridstat_dir.joinpath("gridhists_era5_2012-2023.h5")
    '''
    #'''
    #era5_info = json.load(Path("data/list_feats_nldas2.json").open("r"))
    gs_path = gridstat_dir.joinpath("gridstats_nldas2_2012-2023.h5")
    gh_path = gridstat_dir.joinpath("gridhists_nldas2_2012-2023.h5")
    #'''

    ## select timegrids for the gridstat/gridhist files
    #substr = "timegrid_era5"
    substr = "timegrid_nldas2"
    timegrids = sorted([p for p in tg_dir.iterdir() if substr in p.name])
    print(timegrids)

    ## generate the gridstat hdf5 from timegrid hdf5s
    '''
    make_gridstat_hdf5(
            timegrids=timegrids,
            out_file=gs_path,
            depermute=True,
            time_sector_size=24*14,
            space_sector_chunks=16,
            nworkers=16,
            debug=True,
            )
    '''

    #exit(0)

    ## generate the gridhist hdf5 from timegrid hdf5s
    '''
    hist_bounds = era5_info["hist-bounds"]
    make_gridhist_hdf5(
            timegrids=timegrids,
            out_file=gh_path,
            depermute=True,
            time_sector_size=24*14,
            space_sector_chunks=16,
            hist_resolution={
                **{fl:512 for fl in hist_bounds.keys()},
                "weasd":2048,
                "shtfl":1024,
                "lhtfl":1024,
                },
            hist_bounds=hist_bounds,
            nworkers=16,
            debug=True,
            )
    '''

    #exit(0)

    ## grab the labels for the plotting method calls below
    with h5py.File(gs_path, "r") as F:
        dattrs = json.loads(F["data"].attrs["gridstats"])
        dlabels = dattrs["flabels"]
        sattrs = json.loads(F["data"].attrs["static"])
        slabels = sattrs["flabels"]

    ## plot pixel-wise min/max/mean/stddev per feature
    '''
    logscale = ["weasd", "apcp"]
    exclude_bounds = ["weasd", "apcp"]
    plot_gridstats_spatial(
            gridstat_path=gs_path,
            fig_dir=fig_dir,
            plot_spec_per_feat={l:{
                #"vmin":[*[era5_info["hist-bounds"][l][0]]*3,None
                #    ] if l not in exclude_bounds else [None]*4,
                #"vmax":[*[era5_info["hist-bounds"][l][1]]*3,None
                #    ] if l not in exclude_bounds else [None]*4,
                #"vmax":[era5_info["hist-bounds"][l][1]]*3+[None],
                #"title":era5_info["desc-mapping"][l],
                "title":l,
                "cbar_orient":"horizontal",
                "cmap":"nipy_spectral",
                "norm":["linear","symlog"][l in logscale],
                } for l in dlabels
                },
            sector_size=32768,
            debug=True,
            )
    '''

    #exit(0)

    hist_plot_specs = {
            "apcp":{"yscale":"log"},
            #"weasd":{"ylim":(0,2e8)},
            #"weasd":{"ylim":(0,.05)},
            "weasd":{"yscale":"log"},
            #"dswrf":{"ylim":(0,2e8)},
            "dswrf":{"ylim":(0,.01)},
            "evp":{"yscale":"log"},
            #"lai-high":{"ylim":(0,7e8)},
            "lai-high":{"ylim":(0,.01)},
            #"lai-low":{"ylim":(0,7e8)},
            "lai-low":{"ylim":(0,.1)},
            "lhtfl":{"yscale":"log"},
            "pevap":{"yscale":"log", "ylim":(1e-11,1)},
            "shtfl":{"yscale":"log"},
            #"swnet":{"ylim":(0,2e8)},
            "swnet":{"ylim":(0,.01)},
            "pres":{"xlim":(60000,120000)},
            "dwpt":{"xlim":(220,320)},
            "tmp":{"xlim":(220,320)},
            "tskin":{"xlim":(220,350)},
            "tsoil-07":{"xlim":(240,340)},
            "tsoil-28":{"xlim":(240,340)},
            "tsoil-100":{"xlim":(240,340)},
            "tsoil-289":{"xlim":(240,340)},
            }
    normalize = True
    ## plot the global histograms
    #'''
    with h5py.File(gh_path, "r") as F:
        for fk in F["/data/hists"].keys():
            tmp_coords = F[f"/data/hcoords/{fk}"][...]
            tmp_hist = np.sum(F[f"/data/hists/{fk}"][...], axis=0)
            print(f"{fk}: {np.median(tmp_hist)} {np.amax(tmp_hist)}")
            if normalize:
                tmp_hist = tmp_hist.astype(np.float64) / np.sum(tmp_hist)
                tmp_hist[tmp_hist==0] = np.nan
            lname = era5_info["desc-mapping"][fk]
            units = era5_info["units"]["dynamic"][fk]
            plot_hists(
                    counts=[tmp_hist],
                    labels=[f"{fk} {units}"],
                    bin_coords=[tmp_coords],
                    plot_spec={
                        "title":f"{lname} (2012-2023)",
                        "ylabel":"Percent Density",
                        "xlabel":units,
                        "linewidth":3,
                        "cmap":"tab20",
                        "title_fontsize":30,
                        "label_fontsize":22,
                        "legend_fontsie":26,
                        "tick_fontsize":24,
                        "hlines":[(0,{"color":"lightgrey","linewidth":2})],
                        "vlines":[(0,{"color":"lightgrey","linewidth":2})],
                        **hist_plot_specs.get(fk, {}),
                        },
                    show=False,
                    fig_path=fig_dir.joinpath(
                        f"gridhist_full_2012-2023_{fk}.png"),
                    )
    #'''

    ## plot the histograms stratified by static parameter
    #'''
    #strat_param = "soilt"
    #strat_param = "vt-low"
    strat_param = "vt-high"
    normalize = True
    norm_by_global = False
    with h5py.File(gh_path, "r") as F:
        for fk in F["/data/hists"].keys():
            tmp_coords = F[f"/data/hcoords/{fk}"][...]
            slabels = json.loads(F["data"].attrs["static"])["flabels"]
            sdata = F["/data/static"][...,slabels.index(strat_param)]
            full_fhist = F[f"/data/hists/{fk}"][...]
            hists = []
            labels = []
            for st in np.unique(sdata.astype(int)):
                tmp_hist = np.sum(full_fhist[sdata==st], axis=0)
                print(f"{fk} {st}: {np.median(tmp_hist)} {np.amax(tmp_hist)}")
                if normalize:
                    tmp_hist = tmp_hist.astype(np.float64)
                    if norm_by_global:
                        tmp_hist = tmp_hist / np.sum(full_fhist)
                    else:
                        tmp_hist = tmp_hist / np.sum(tmp_hist)
                    tmp_hist[tmp_hist==0] = np.nan
                hists.append(tmp_hist)
                labels.append(
                        era5_info["static-classes"][strat_param][str(st)])
            lname = era5_info["desc-mapping"][fk]
            units = era5_info["units"]["dynamic"][fk]
            plot_hists(
                    counts=hists,
                    labels=labels,
                    bin_coords=[tmp_coords for i in range(len(hists))],
                    plot_spec={
                        "title":f"{lname} by {strat_param} (2012-2023)",
                        "ylabel":"Percent Density",
                        "xlabel":units,
                        "linewidth":3,
                        "cmap":"tab20",
                        "title_fontsize":24,
                        "label_fontsize":22,
                        "legend_fontsie":24,
                        "tick_fontsize":20,
                        "hlines":[(0,{"color":"lightgrey","linewidth":2})],
                        "vlines":[(0,{"color":"lightgrey","linewidth":2})],
                        **hist_plot_specs.get(fk, {}),
                        },
                    show=False,
                    fig_path=fig_dir.joinpath(
                        f"gridhist_{strat_param}_2012-2023_{fk}.png"),
                    )
    #'''
