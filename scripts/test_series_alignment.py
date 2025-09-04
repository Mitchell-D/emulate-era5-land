import h5py
from pathlib import Path
import numpy as np
import json
from datetime import datetime as dttime
from datetime import timezone as tz

from emulate_era5_land.plotting import plot_time_lines_multiy

if __name__=="__main__":
    ## extract and plot time series from a single file
    '''
    #target_latlon = (34.729, -86.585) ## huntsville
    target_latlon = (33.31, -80.19) ## ian landfall

    ## rain event focus
    #time_range = (dttime(2012, 3, 8, 4), dttime(2012,3,8,12))
    ## multi day time period
    #time_range = (dttime(2012,3,8,tzinfo=tz.utc),
    #        dttime(2012,3,12,tzinfo=tz.utc))
    ##
    ## hour zero
    #time_range = (dttime(2011,12,31,tzinfo=tz.utc),
    #        dttime(2012,1,2,tzinfo=tz.utc))
    ## hurricane ian
    time_range = (dttime(2022,9,29,tzinfo=tz.utc),
            dttime(2022,10,2,tzinfo=tz.utc))
    #timegrid_path = Path("data/timegrids/timegrid_era5_2012.h5")
    timegrid_path = Path("/rstor/mdodson/era5/timegrids-test/" + \
            "timegrid_era5_2022.h5")

    F = h5py.File(timegrid_path, "r")
    dattrs = json.loads(F["data"].attrs["dynamic"])
    sattrs = json.loads(F["data"].attrs["static"])
    sdata = F["/data/static"][...]
    lat = sdata[...,sattrs["flabels"].index("lat")]
    lon = sdata[...,sattrs["flabels"].index("lon")]
    target_ix = np.argmin((lat-target_latlon[0])**2+(lon-target_latlon[1])**2)
    etimes = F["/data/time"][...]
    dtimes = [dttime.fromtimestamp(int(t),tz=tz.utc) for t in etimes]
    m_time = np.asarray([
        t >= time_range[0] and t < time_range[1] for t in dtimes
        ])
    extract_vars = [
            "apcp", #"tsoil-07", "tsoil-28", ## check rain
            #"lhtfl", "shtfl",
            #"lwnet", "swnet",
            "evp", "pevap",
            #"dlwrf", "dswrf",
            #"apcp",
            ]

    data = []
    for fl in extract_vars:
        x = F["/data/dynamic"][:,target_ix,dattrs["flabels"].index(fl)][m_time]
        data.append(x)

    fig_path = Path("figures/alignment/time_series_alignment_hrc-ian-new.png")
    plot_time_lines_multiy(
            time_series=data,
            times=[t for i,t in enumerate(dtimes) if m_time[i]],
            plot_spec={
                "y_labels":extract_vars,
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
    print(f"Generated {fig_path.as_posix()}")
    '''

    ## extract a time range for specific pixels
    '''
    trange_coord = [
            #((dttime(2015,5,15),dttime(2015,5,17)),(28.8,-98.1)), ## 92776
            #((dttime(2013,5,20),dttime(2013,5,22)),(33.3,-96.0)), ## 79104
            #((dttime(2016,5,25),dttime(2016,5,27)),(30.0,-96.5)), ## 90220
            #((dttime(2014,6,22),dttime(2014,6,24)),(25.2,-100.5)), ## 98327
            #((dttime(2015,6,22),dttime(2015,6,24)),(39.5,-82.6)), ## 51951
            #((dttime(2023,7,12),dttime(2023,7,14)),(37.0,-94.6)), ## 63422
            ((dttime(2021,3,15),dttime(2021,3,15,12)),(34.56,-87.06))
            ]
    extract_vars = [
            "apcp", #"tsoil-07", "tsoil-28", ## check rain
            "vsm-07","vsm-28",
            #"evp", #"pevap",
            "lhtfl", "shtfl",
            #"lwnet", "swnet",
            #"dlwrf", "dswrf",
            #"apcp",
            ]
    plot_diff = ["vsm-07", "vsm-28"]
    tg_dir = Path("/rstor/mdodson/era5/timegrids")
    fig_dir = Path("figures/alignment")

    fig_path = "alignment_apcp-vsm_{stime}.png"
    for (tt0,ttf),(tlat,tlon) in trange_coord:
        tt0 = tt0.replace(tzinfo=tz.utc)
        ttf = ttf.replace(tzinfo=tz.utc)
        timegrid_path = tg_dir.joinpath(f"timegrid_era5_{tt0.year}.h5")
        F = h5py.File(timegrid_path, "r")
        dattrs = json.loads(F["data"].attrs["dynamic"])
        sattrs = json.loads(F["data"].attrs["static"])
        etimes = F["/data/time"][...]
        dtimes = [dttime.fromtimestamp(int(t),tz=tz.utc) for t in etimes]
        sdata = F["/data/static"][...]
        lat = sdata[...,sattrs["flabels"].index("lat")]
        lon = sdata[...,sattrs["flabels"].index("lon")]
        six = np.argmin((lat-tlat)**2+(lon-tlon)**2)
        m_time = np.asarray([t >= tt0 and t < ttf for t in dtimes])
        data = []
        for fl in extract_vars:
            fix = dattrs["flabels"].index(fl)
            x = F["/data/dynamic"][m_time,six,fix]
            if fl in plot_diff:
                x = np.diff(x, axis=0)
                x = np.concatenate([np.full(x.shape, np.nan)[:1], x], axis=0)
            data.append(x)

        plot_times = [t for i,t in enumerate(dtimes) if m_time[i]]

        fig_path = fig_dir.joinpath(
                fig_path.format(stime=tt0.strftime('%Y%m%d')))
        plot_time_lines_multiy(
                time_series=data,
                times=plot_times,
                plot_spec={
                    "y_labels":extract_vars,
                    "fig_size":(14,6),
                    "dpi":120,
                    "spine_increment":.06,
                    "date_format":"%Y-%m-%d %H",
                    "grid":True,
                    "grid_kwargs":{"which":"major"},
                    "zero_axis":True,
                    },
                show=False,
                fig_path=fig_path,
                )
        print(f"Generated {fig_path.as_posix()}")
    '''
