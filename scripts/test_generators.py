
#import torch
import numpy as np
from pathlib import Path
from datetime import datetime,timedelta
import json
import pickle as pkl

from emulate_era5_land.extract_era5 import get_grib_extract_gen
from emulate_era5_land.plotting import plot_time_lines_multiy

'''
class TimegridSequenceDataset(torch.utils.data.IterableDataset):
    def __init__(self):
        pass
'''

if __name__=="__main__":
    proj_root_dir = Path("/rhome/mdodson/emulate-era5-land")
    data_dir = proj_root_dir.joinpath("data/era5/2022")

    ## get the 2d index of the target location
    target_latlon = (33.31, -80.19) ## ian landfall
    static_pkl = Path("data/static/era5_static.pkl")
    slabels,sdata = pkl.load(static_pkl.open("rb"))
    m_valid = sdata[slabels.index("m_valid")].astype(bool)
    lat = sdata[slabels.index("lat")][m_valid]
    lon = sdata[slabels.index("lon")][m_valid]
    #tix = np.unravel_index(
    #        np.argmin((lat-target_latlon[0])**2+(lon-target_latlon[1])**2),
    #        lat.shape
    #        )
    tix = np.argmin((lat-target_latlon[0])**2 + (lon-target_latlon[1])**2)
    print(tix)

    get_gen = get_grib_extract_gen(
            rec_labels=["skt","stl1","stl2","stl3","stl4","swvl1","swvl2",
                "swvl3","swvl4","var251","e","tp"],
            accumulation_vars=["var251","e","tp"],
            conversions={
                "tp":lambda w:np.clip(w*1000,0,None),
                "e":lambda w:w*1000,
                "var251":lambda w:w*1000,
                }
            )

    gen  = get_gen(
            data_dir.joinpath("era5land_soil_vars_202210.grib"),
            data_dir.joinpath("era5land_soil_vars_202209.grib"),
            m_valid,
            )

    rlabels = None
    daily_data = []
    for n,(rl,ra) in enumerate(gen):
        if rlabels is None:
            rlabels = rl
        daily_data.append(ra)
        if n>5:
            break

    x = np.concatenate(daily_data, axis=0)
    print(x.shape)

    plot_feats = ["tp", "e", "var251"]
    fig_path = Path("figures/alignment/time_series_alignment_hrc-ian.png")
    times = [datetime(2022,10,1)+timedelta(hours=1)*nh
            for nh in range(x.shape[0])]
    plot_time_lines_multiy(
            time_series=[x[:, tix, fix] for fix in
                [rlabels.index(pf) for pf in plot_feats]],
            times=times,
            plot_spec={
                "y_labels":plot_feats,
                "fig_size":(14,6),
                "dpi":120,
                "spine_increment":.06,
                "date_format":"%Y-%m-%d %H",
                "grid":True,
                "zero_axis":True,
                },
            show=False,
            fig_path=fig_path,
            )
