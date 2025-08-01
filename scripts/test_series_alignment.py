import h5py
from pathlib import Path
import numpy as np
import json
from datetime import datetime

from emulate_era5_land.plotting import plot_time_lines_multiy

if __name__=="__main__":
    target_latlon = (34.729, -86.585)
    ## rain event focus
    #time_range = (datetime(2012, 3, 8, 4), datetime(2012,3,8,12))
    #time_range = (datetime(2012, 3, 8), datetime(2012,3,12))
    ## hour zero
    time_range = (datetime(2011, 12, 31), datetime(2012,1,2))
    timegrid_path = Path("data/timegrids/timegrid_era5_2012.h5")

    F = h5py.File(timegrid_path, "r")
    dattrs = json.loads(F["data"].attrs["dynamic"])
    sattrs = json.loads(F["data"].attrs["static"])
    sdata = F["/data/static"][...]
    lat = sdata[...,sattrs["flabels"].index("lat")]
    lon = sdata[...,sattrs["flabels"].index("lon")]
    target_ix = np.argmin((lat-target_latlon[0])**2+(lon-target_latlon[1])**2)
    etimes = F["/data/time"][...]
    dtimes = [datetime.fromtimestamp(int(t)) for t in etimes]
    m_time = np.asarray([
        t >= time_range[0] and t < time_range[1] for t in dtimes
        ])
    extract_vars = [
            #"apcp", "tsoil-07", "tsoil-28", ## check rain
            #"lhtfl", "shtfl",
            #"lwnet", "swnet",
            #"evp", "pevap",
            "dlwrf", "dswrf"
            #"apcp",
            ]

    data = []
    for fl in extract_vars:
        x = F["/data/dynamic"][:,target_ix,dattrs["flabels"].index(fl)][m_time]
        data.append(x)

    fig_path = Path("figures/alignment/time_series_alignment_crossfile.png")
    plot_time_lines_multiy(
            time_series=data,
            times=[t for i,t in enumerate(dtimes) if m_time[i]],
            plot_spec={
                "y_labels":extract_vars,
                "fig_size":(14,6),
                "dpi":120,
                "spine_increment":.03,
                "date_format":"%Y-%m-%d %H",
                "grid":True,
                "zero_axis":True,
                },
            show=False,
            fig_path=fig_path,
            )
    print(f"Generated {fig_path.as_posix()}")
