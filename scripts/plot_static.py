"""
Generate masks of each unique class combination of vegetation and soil pixels,
and plot a grid showing the number of pixels in each combination category.
"""
import numpy as np
import pickle as pkl
import h5py
from netCDF4 import Dataset
import json
from pathlib import Path
import matplotlib.pyplot as plt

from emulate_era5_land.plotting import plot_geo_ints,plot_geo_scalar,plot_lines
from emulate_era5_land.plotting import plot_lines,plot_combo_matrix

if __name__=="__main__":

    #'''
    proj_root_dir = Path("/rhome/mdodson/emulate-era5-land")
    static_pkl_path = proj_root_dir.joinpath("data/static/era5_static.pkl")
    info_era5 = json.load(proj_root_dir.joinpath(
        "data/list_feats_era5.json").open("r"))

    ## set pixel bounds wrt top left of full grid domain
    grid_bounds,locale = (slice(None,None), slice(None,None)),"full"
    #grid_bounds,locale = (slice(80,108), slice(35,55)),"lt-high-sierra"
    #grid_bounds,locale = (slice(25,50), slice(308,333)),"lt-north-michigan"
    #grid_bounds,locale = (slice(40,65), slice(184,209)),"lt-high-plains"
    #grid_bounds,locale = (slice(123,168), slice(259,274)),"lt-miss-alluvial"
    #grid_bounds,locale = (slice(98,188), slice(144,254)),"central-tx"

    ## optionally use latlon bounds instead of pixel bounds
    ## if you want to use pixel bounds, make sure bbox=None
    bbox = None
    #bbox,locale = ((33.,37.), (-89.,-85)),"bobcat-cave-watershed"

    soil_ints_fig_path = proj_root_dir.joinpath(
            f"figures/static/static_gridded_soilt_{locale}.png")
    veg_ints_high_fig_path = proj_root_dir.joinpath(
            f"figures/static/static_gridded_vt-high_{locale}.png")
    veg_ints_low_fig_path = proj_root_dir.joinpath(
            f"figures/static/static_gridded_vt-low_{locale}.png")
    elev_fig_path = proj_root_dir.joinpath(
            f"figures/static/static_gridded_elev_{locale}.png")


    """ Generate pixel masks for each veg/soil class combination """
    ## Load the full-CONUS static pixel grid
    slabels,sdata = pkl.load(static_pkl_path.open("rb"))
    ## Get the integer-identified soil texture and vegetation class grids
    lat = sdata[slabels.index("lat")]
    lon = sdata[slabels.index("lon")]
    int_veg_low = sdata[slabels.index("vt-low")].astype(int)
    int_veg_high = sdata[slabels.index("vt-high")].astype(int)
    int_soil = sdata[slabels.index("soilt")].astype(int)

    ## custom thresholds for determining valid land pixel.
    ## These should be the same as those selected for timegrid extraction.
    m_valid_base = sdata[slabels.index("m_valid")].astype(bool)
    m_lakec = sdata[slabels.index("lakec")] < .15
    m_land = sdata[slabels.index("landmask")] >= .8
    m_valid = m_valid_base & m_lakec & m_land
    '''
    slope = sdata[slabels.index("slope")]
    elev = sdata[slabels.index("elev")]
    elev_std = sdata[slabels.index("elev_std")]
    m_valid = sdata[slabels.index("m_valid")].astype(bool)
    '''

    if not bbox is None:
        m_lat = (lat >= bbox[0][0]) & (lat <= bbox[0][1])
        m_lon = (lon >= bbox[1][0]) & (lon <= bbox[1][1])
        latrange,lonrange = map(np.unique,np.where(m_lat & m_lon))
        grid_bounds = (slice(np.amin(latrange), np.amax(latrange)+1),
                slice(np.amin(lonrange), np.amax(lonrange)+1))

    ## Plot combination matrix of soil textures and vegetation
    #'''
    unq_vh = np.unique(int_veg_high[m_valid])
    unq_vl = np.unique(int_veg_low[m_valid])
    unq_st = np.unique(int_soil[m_valid])
    combos_st_vh = np.full((unq_st.size, unq_vh.size), np.nan)
    combos_st_vl = np.full((unq_st.size, unq_vl.size), np.nan)
    combos_vh_vl = np.full((unq_vh.size, unq_vl.size), np.nan)
    for i,stv in enumerate(unq_st):
        m_stv = (int_soil==stv)
        for j,vhv in enumerate(unq_vh):
            m_vhv = (int_veg_high==vhv)
            combos_st_vh[i,j] = np.count_nonzero(m_stv & m_vhv)
        for j,vlv in enumerate(unq_vl):
            m_vlv = (int_veg_low==vlv)
            combos_st_vl[i,j] = np.count_nonzero(m_stv & m_vlv)
    for i,vhv in enumerate(unq_vh):
        m_vhv = (int_veg_high==vhv)
        for j,vlv in enumerate(unq_vl):
            m_vlv = (int_veg_low==vlv)
            combos_vh_vl[i,j] = np.count_nonzero(m_vhv & m_vlv)
    combos_st_vh[combos_st_vh==0] = np.nan
    combos_st_vl[combos_st_vl==0] = np.nan
    combos_vh_vl[combos_vh_vl==0] = np.nan

    ## Make a grid plot of the number of samples within each combination.
    plot_combo_matrix(
            matrix=combos_st_vh,
            fig_path=proj_root_dir.joinpath(
                "figures/static/static_combos_soilt_vt-high.png"),
            plot_spec={
                "title":"Pixels per Soil Type and High Vegetation\n" + \
                        "Class Combination (Full Domain)",
                "xticks":[info_era5["static-classes"]["vt-high"][str(v)]
                    for v in unq_vh],
                "yticks":[info_era5["static-classes"]["soilt"][str(v)]
                    for v in unq_st],
                "xlabel":"High Vegetation Category",
                "ylabel":"Soil Texture Category",
                "cmap":"inferno",
                "norm":"log"
                #"vmax":50000,
                },
            )
    plot_combo_matrix(
            matrix=combos_st_vl,
            fig_path=proj_root_dir.joinpath(
                "figures/static/static_combos_soilt_vt-low.png"),
            plot_spec={
                "title":"Pixels per Soil Type and Low Vegetation\n" + \
                        "Class Combination (Full Domain)",
                "xticks":[info_era5["static-classes"]["vt-low"][str(v)]
                    for v in unq_vl],
                "yticks":[info_era5["static-classes"]["soilt"][str(v)]
                    for v in unq_st],
                "xlabel":"Low Vegetation Category",
                "ylabel":"Soil Texture Category",
                "cmap":"inferno",
                "norm":"log"
                #"vmax":50000,
                },
            )
    plot_combo_matrix(
            matrix=combos_vh_vl,
            fig_path=proj_root_dir.joinpath(
                "figures/static/static_combos_vt-high_vt-low.png"),
            plot_spec={
                "title":"Pixels per High and Low Vegetation\n" + \
                        "Class Combination (Full Domain)",
                "xticks":[info_era5["static-classes"]["vt-low"][str(v)]
                    for v in unq_vl],
                "yticks":[info_era5["static-classes"]["vt-high"][str(v)]
                    for v in unq_vh],
                "xlabel":"Low Vegetation Category",
                "ylabel":"High Vegetation Category",
                "cmap":"inferno",
                "norm":"log"
                #"vmax":50000,
                },
            )
    #exit(0)
    #'''

    ## Plot integer high vegetation map
    #'''
    plot_geo_ints(
            int_data=np.where(m_valid, int_veg_high, np.nan)[*grid_bounds],
            lat=lat[*grid_bounds],
            lon=lon[*grid_bounds],
            geo_bounds=None,
            #int_ticks=np.array(list(range(14)))*(13/14)+.5,
            int_labels={ix:info_era5["static-classes"]["vt-high"][str(ix)]
                for ix in np.unique(int_veg_high)},
            fig_path=veg_ints_high_fig_path,
            latlon_ticks=True,
            show=False,
            plot_spec={
                "cmap":"tab20b",
                "cbar_pad":0.1,
                "cbar_orient":"horizontal",
                "cbar_shrink":.9,
                "tick_rotation":45,
                "tick_frequency":18,
                "cbar_tick_rotation":-45,
                "cbar_fontsize":14,
                "title":f"High Vegetation Classes ({locale})",
                "title_fontsize":18,
                "interpolation":"none",
                },
            colors={int(k):v
                for k,v in info_era5["static-colors"]["vt-high"].items()},
            )
    print(f"Generated {veg_ints_high_fig_path.as_posix()}")
    plt.clf()
    #'''

    ## Plot integer low vegetation map
    #'''
    plot_geo_ints(
            int_data=np.where(m_valid, int_veg_low, np.nan)[*grid_bounds],
            lat=lat[*grid_bounds],
            lon=lon[*grid_bounds],
            geo_bounds=None,
            int_labels={ix:info_era5["static-classes"]["vt-low"][str(ix)]
                for ix in np.unique(int_veg_low)},
            fig_path=veg_ints_low_fig_path,
            latlon_ticks=True,
            show=False,
            plot_spec={
                "cmap":"tab20b",
                "cbar_pad":0.1,
                "cbar_orient":"horizontal",
                "cbar_shrink":.9,
                "tick_rotation":45,
                "tick_frequency":18,
                "cbar_tick_rotation":-45,
                "cbar_fontsize":14,
                "title":f"Low Vegetation Classes ({locale})",
                "title_fontsize":18,
                "interpolation":"none",
                },
            colors={int(k):v
                for k,v in info_era5["static-colors"]["vt-low"].items()},
            )
    print(f"Generated {veg_ints_low_fig_path.as_posix()}")
    plt.clf()
    #'''

    ## Plot integer soil texture map
    #'''
    plot_geo_ints(
            int_data=np.where(m_valid, int_soil, np.nan)[*grid_bounds],
            lat=lat[*grid_bounds],
            lon=lon[*grid_bounds],
            geo_bounds=None,
            int_labels={ix:info_era5["static-classes"]["soilt"][str(ix)]
                for ix in np.unique(int_soil)},
            fig_path=soil_ints_fig_path,
            latlon_ticks=True,
            show=False,
            plot_spec={
                "cmap":"gist_ncar",
                "cbar_pad":0.1,
                "cbar_orient":"horizontal",
                "cbar_shrink":.9,
                "tick_rotation":45,
                "tick_frequency":18,
                "cbar_tick_rotation":-45,
                "cbar_fontsize":14,
                "title":f"Soil Texture Classes ({locale})",
                "title_fontsize":18,
                "interpolation":"none",
                },
            colors={int(k):v
                for k,v in info_era5["static-colors"]["soilt"].items()},
            )
    print(f"Generated {soil_ints_fig_path.as_posix()}")
    #exit(0)
    #'''

    ## Plot scalar elevation
    '''
    plot_geo_scalar(
            data=np.where(m_valid, elev, np.nan)[*grid_bounds],
            latitude=lat[*grid_bounds],
            longitude=lon[*grid_bounds],
            latlon_ticks=True,
            bounds=None,
            plot_spec={
                "title":f"GTOPO30 Elevation in meters ({locale})",
                "cmap":"gnuplot",
                "cbar_label":"Elevation (meters)",
                #"cbar_orient":"horizontal",
                "cbar_orient":"vertical",
                "cbar_shrink":1.,
                "tick_rotation":45,
                "tick_frequency":18,
                "tick_frequency":6,
                "cbar_pad":.02,
                "fontsize_title":18,
                "fontsize_labels":14,
                },
            fig_path=elev_fig_path,
            )
    print(f"Generated {elev_fig_path.as_posix()}")
    '''

    ## plot all real-valued static datasets
    '''
    slabels_to_plot = [
            "pct_sand", "pct_silt", "pct_clay", "porosity", "fieldcap",
            "wiltingp", "bparam", "matricp", "hydcond", "elev", "elev_std",
            "slope", "aspect", "vidx", "hidx"
            ]
    for l in slabels_to_plot:
        plot_geo_scalar(
                data=np.where(m_valid, sdata[slabels.index(l)], np.nan),
                latitude=lat,
                longitude=lon,
                bounds=None,
                plot_spec={
                    "title":f"{l.capitalize()} (Full Domain)",
                    "cmap":"gnuplot",
                    "cbar_label":"",
                    "cbar_orient":"horizontal",
                    "cbar_pad":.02,
                    "fontsize_title":18,
                    "fontsize_labels":14,
                    "norm":"linear",
                    },
                fig_path=proj_root_dir.joinpath(
                    f"figures/static/static_{l}.png"),
                )
    '''
