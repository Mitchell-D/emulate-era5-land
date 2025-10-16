
#import torch
import numpy as np
from pathlib import Path
from datetime import datetime,timedelta,timezone
import json
import pickle as pkl
import torch

from emulate_era5_land.generators import stsd_worker_init_fn
from emulate_era5_land.generators import SparseTimegridSampleDataset
from emulate_era5_land.plotting import plot_geo_scalar,plot_scatter,plot_lines

if __name__=="__main__":
    proj_root = Path("/rhome/mdodson/emulate-era5-land")
    data_dir = proj_root.joinpath("data/timegrids")
    fig_dir = proj_root.joinpath("figures/test-gen")
    info_era5 = json.load(proj_root.joinpath(
                "data/list_feats_era5.json").open("r"))
    gen_loc_path = proj_root.joinpath("data/test-gen/test-gen_10.pkl")
    gen_stats_path = proj_root.joinpath("data/test-gen/test-gen_stats_10.pkl")
    gen_span_path = proj_root.joinpath("data/test-gen/test-gen_span_10.pkl")
    span_plot_labels = ["tmp", "dswrf", "swm-7", "diff swm-7", "diff swm-28"]
    max_plotted_samples = 12

    ## Run the generator and store the time/location of its outputs
    #'''
    calculate_stats = True
    batches_per_stat = 16
    collect_sample_locations = True
    collect_spanning_samples = True
    max_batches = 1024
    ## perilously assume sorting path strings will return files chronologically
    tgs = sorted([
        tg for tg in data_dir.iterdir()
        if int(tg.stem.split("_")[-1]) in range(2012,2018)
        ])
    print("Extracting from:")
    for tg in tgs:
        print(tg.as_posix())
    ds_train = SparseTimegridSampleDataset(
            timegrids=tgs,
            window_feats=[
                #"lai-low", "lai-high", "alb", "weasd", "pres", "windmag",
                #"tmp", "dwpt", "apcp", "dlwrf", "dswrf", "alb"
                #"swm-07", "swm-28", "swm-100", "swm-289",
                #"ugrd","vgrd","evp","pevap","shtfl","lhtfl","swnet","lwnet",
                #"wm-skin","roff-all","evp-snow","evp-trsp","evp-bare",
                #"evp-cnpy","roff-bg","roff-sf",
                #"tsoil-07", "tsoil-28", "tsoil-100", "tsoil-289",
                #"evp-snow", "evp-trsp", "evp-bare", "evp-cnpy",
                #"roff-bg", "roff-sf",
                #"tskin", "wm-skin", "wm-snow",
                "tmp", "dswrf", "swm-7", "diff swm-7", "swm-28",
                ],
            horizon_feats=[
                #"lai-low", "lai-high", "alb", "weasd", "pres", "windmag",
                #"tmp", "dwpt", "apcp", "dlwrf", "dswrf", "alb"
                #"ugrd","vgrd","evp","pevap","shtfl","lhtfl","swnet","lwnet",
                #"wm-skin","roff-all","evp-snow","evp-trsp","evp-bare",
                #"evp-cnpy","roff-bg","roff-sf",
                #"evp-snow", "evp-trsp", "evp-bare", "evp-cnpy",
                #"roff-bg", "roff-sf",
                "tmp", "dswrf", "swm-7", "diff swm-7"
                ],
            target_feats=[
                #"diff swm-07", "diff swm-28", "diff swm-100", "diff swm-289",
                #"diff tsoil-07", "diff tsoil-28",
                #"diff tsoil-100", "diff tsoil-289",
                #"diff tskin", "diff wm-skin", "diff wm-snow",
                "diff swm-28"
                ],
            static_feats=["vc-high", "vc-low", "geopot", "lakec"],
            static_int_feats=["soilt", "vt-low", "vt-high"],
            aux_dynamic_feats=[],
            aux_static_feats=["vidxs", "hidxs"],
            derived_feats={
                "windmag":(("ugrd", "vgrd"), tuple(),
                    "lambda d,s:(d[0]**2+d[1]**2)**(1/2)")
                },
            static_embed_maps={
                "soilt":[0, 1, 2, 3, 4, 5, 6, 7],
                "vt-low":[0, 1, 2, 7, 9, 10, 11, 13, 16, 17],
                "vt-high":[0, 3, 5, 6, 18, 19],
                },
            #window_size=24,
            window_size=72,
            horizon_size=72,
            norm_coeffs=info_era5["norm-coeffs"],
            shuffle=True,
            seed=200007221751,
            sample_cutoff=.7,
            sample_across_files=True,
            sample_under_cutoff=True,
            #sample_separation=197, ## number coprime with 24 is best
            #sample_separation=157, ## number coprime with 24 is best
            sample_separation=409, ## Only want a fraction of each chunk.
            random_offset=True,
            chunk_pool_count=48,
            buf_size_mb=4096,
            buf_slots=48,
            buf_policy=0,
            debug=True
            )
    gen = torch.utils.data.DataLoader(
            dataset=ds_train,
            batch_size=64,
            num_workers=8,
            prefetch_factor=4,
            worker_init_fn=stsd_worker_init_fn,
            )

    sample_locations = []
    stats_buffer = []
    all_stats = {k:[] for k in "whsy"}
    spanning_samples = []
    for bix,(inputs,outputs,auxiliary) in enumerate(gen, 1):
        w,h,s,si,init = inputs
        y, = outputs
        aux_d,aux_s,t = auxiliary
        tmp_months,tmp_days = map(np.asarray, zip(*[
            (tmpt.month,tmpt.day)
            for tmpt in map(datetime.fromtimestamp, t[:,0].numpy().astype(int))
            ]))
        if collect_spanning_samples:
            m_span = (tmp_months == 12) & (tmp_days >= 26)
            if not np.any(m_span):
                continue
            w = w[m_span]
            h = h[m_span]
            s = s[m_span]
            init = init[m_span]
            y = y[m_span]
            aux_d = aux_d[m_span]
            aux_s = aux_s[m_span]
            t = t[m_span]
            for j in range(w.shape[0]):
                spanning_samples.append((
                    (w[j],h[j],s[j],init[j]),
                    (y[j],),
                    (aux_d[j],aux_s[j],t[j])
                    ))

        if collect_sample_locations:
            sample_locations.append((
                aux_s[:,0].numpy().astype(np.uint16),
                aux_s[:,1].numpy().astype(np.uint16),
                t[:,0].numpy().astype(int),
                t[:,-1].numpy().astype(int)
                ))

        ## every batches_per_stat batches, collect the statistics
        if calculate_stats:
            if (bix+1) % batches_per_stat == 0:
                tmp_stats = {
                    k:np.concatenate(
                        [sb[k] for sb in stats_buffer],
                        axis=0
                        ).reshape((-1, stats_buffer[0][k].shape[-1]))
                    for k in "whsy"
                    }
                for k in "whsy":
                    all_stats[k].append(
                            np.stack([
                                np.amin(tmp_stats[k], axis=0),
                                np.amax(tmp_stats[k], axis=0),
                                np.average(tmp_stats[k], axis=0),
                                np.std(tmp_stats[k], axis=0),
                                ], axis=-1)
                            )
                stats_buffer = []
            else:
                stats_buffer.append({"w":w,"h":h,"s":s,"y":y})
        print(f"Got sample: {[tuple(v.shape) for v in (w,h,y,s)]}")
        if bix >= max_batches:
            break

    if collect_spanning_samples:
        pkl.dump((ds_train.signature, spanning_samples),
                gen_span_path.open("wb"))
    if collect_sample_locations:
        pkl.dump((ds_train.signature, sample_locations),
                gen_loc_path.open("wb"))
    if calculate_stats:
        pkl.dump((ds_train.signature, all_stats),
                gen_stats_path.open("wb"))
    #'''

    ## Print out the stat metrics per feature / data catagory / batch group
    '''
    sig,stats = pkl.load(gen_stats_path.open("rb"))
    fmap = {"w":"window","h":"horizon","s":"static","y":"target"}
    for k in "whsy":
        for j,fl in enumerate(sig[f"{fmap[k]}_feats"]):
            print(f"{fmap[k]} {fl}")
            for i,m in enumerate(["   min", "   max", "  mean", "stddev"]):
                vals = np.stack([v[j,i] for v in stats[k]], axis=0)
                vmets = [np.amin(vals, axis=0), np.amax(vals, axis=0),
                        np.average(vals, axis=0), np.std(vals, axis=0)]
                print(f"{m}:",' '.join([f'{v:.5f}' for v in vmets]))
            print()
    '''

    ## plot the spatial and temporal distribution of generator outputs
    '''
    max_batches = 256
    desc = "sample sep 157, pool chunks 48, batch size 256"
    sgn,locs = pkl.load(gen_loc_path.open("rb"))
    slabels,sdata = pkl.load(Path("data/static/era5_static.pkl").open("rb"))
    m_valid_base = sdata[slabels.index("m_valid")].astype(bool)
    m_lakec = sdata[slabels.index("lakec")] < .15
    m_land = sdata[slabels.index("landmask")] >= .8
    m_valid = m_valid_base & m_lakec & m_land
    lat = sdata[slabels.index("lat")]
    lon = sdata[slabels.index("lon")]
    spatial_counts = np.zeros(lat.shape)
    spatial_doys = np.full(lat.shape, np.nan)
    batch_time = []
    batch_doy = []
    for bix,(ixv,ixh,t0,tf) in enumerate(locs):
        if bix==max_batches:
            break
        ixv = ixv.astype(int)
        ixh = ixh.astype(int)
        doy = []
        for t in t0:
            st = datetime.fromtimestamp(int(t), tz=timezone.utc)
            cur_doy = int(st.strftime("%j"))
            doy.append(cur_doy)
            batch_time.append((bix,st))
            batch_doy.append((bix,cur_doy))
        spatial_counts[ixv,ixh] += 1
        spatial_doys[ixv,ixh] = np.asarray(doy)

    spatial_counts[~m_valid] = np.nan
    #spatial_doys[~m_valid] = np.nan

    fpath = fig_dir.joinpath(f"{gen_loc_path.stem}_b{bix}_spatial-counts.png")
    plot_geo_scalar(
            data=spatial_counts,
            latitude=lat,
            longitude=lon,
            plot_spec={
                "title":f"Spatial Sample Counts ({gen_loc_path.name}); " + \
                        f"batches: {bix}\n{desc}",
                "cmap":"gnuplot",
                "cbar_label":"Number of samples",
                "cbar_orient":"horizontal",
                "cbar_shrink":.9,
                "cbar_pad":.02,
                "fontsize_title":18,
                "fontsize_labels":14,
                },
            fig_path=fpath,
            )
    print(f"Generated {fpath.as_posix()}")

    fpath = fig_dir.joinpath(f"{gen_loc_path.stem}_b{bix}_spatial-doys.png")
    plot_geo_scalar(
            data=spatial_doys,
            latitude=lat,
            longitude=lon,
            plot_spec={
                "title":f"Spatial Sample DoYs ({gen_loc_path.name}); " + \
                        f"batches: {bix}\n{desc}",
                "cmap":"hsv",
                "cbar_label":"Most Recent Sample Day of Year",
                "cbar_orient":"horizontal",
                "cbar_shrink":.9,
                "cbar_pad":.02,
                "fontsize_title":18,
                "fontsize_labels":14,
                "vmin":1,
                "vmax":365,
                },
            fig_path=fpath,
            )
    print(f"Generated {fpath.as_posix()}")

    batch,time = zip(*batch_time)
    fpath = fig_dir.joinpath(f"{gen_loc_path.stem}_b{bix}_batch-time.png")
    plot_scatter(
            x=time,
            y=batch,
            size=1,
            color="black",
            plot_spec={
                "title":f"sample times vs batch ({gen_loc_path.name}); " + \
                        f"batches: {bix}\n{desc}",
                "marker":",",
                "xlabel":"sample initialization time",
                "ylabel":"batch number",
                "linewidths":0,
                },
            fig_path=fpath,
            )
    print(f"Generated {fpath.as_posix()}")

    batch,doy = zip(*batch_doy)
    fpath = fig_dir.joinpath(f"{gen_loc_path.stem}_b{bix}_batch-doy.png")
    plot_scatter(
            x=doy,
            y=batch,
            size=1,
            color="black",
            plot_spec={
                "title":f"sample DoY vs batch ({gen_loc_path.name}); " + \
                        f"batches: {bix}\n{desc}",
                "marker":",",
                "xlabel":"sample initialization day of year",
                "ylabel":"batch number",
                "linewidths":0,
                },
            fig_path=fpath,
            )
    print(f"Generated {fpath.as_posix()}")
    '''

    sig,span_samples = pkl.load(gen_span_path.open("rb"))
    w_fixs,h_fixs,y_fixs = [],[],[]
    w_fl,h_fl,y_fl = [],[],[]
    span_feats = {k:{"fl":[],"ix":[]} for k in "why"}
    sig_map = {"w":"window_feats","h":"horizon_feats","y":"target_feats"}

    ## collect the indeces and labels per data category for plotted series
    import matplotlib.colors as cm
    import matplotlib.pyplot as plt
    cmap = {}
    for i,spl in enumerate(span_plot_labels):
        for k in "why":
            if spl in sig[sig_map[k]]:
                span_feats[k]["ix"].append(sig[sig_map[k]].index(spl))
                span_feats[k]["fl"].append(spl)
                if spl not in cmap.keys():
                    cmap[spl] = cm.to_hex(plt.cm.Dark2(i))

    for six,((w,h,_,_),(y,),(_,_,t)) in enumerate(
            span_samples[:max_plotted_samples]):
        plot_samples = []
        w_time = list(map(
            datetime.fromtimestamp,
            map(int,list(t[1:w.shape[0]+1]))
            ))
        h_time = list(map(
            datetime.fromtimestamp,
            map(int,list(t[-h.shape[0]:]))
            ))
        for ix,fl in zip(span_feats["w"]["ix"],span_feats["w"]["fl"]):
            plot_samples.append([w_time,w[...,ix],cmap[fl],"-",f"w {fl}"])
        for ix,fl in zip(span_feats["h"]["ix"],span_feats["h"]["fl"]):
            plot_samples.append([h_time,h[...,ix],cmap[fl],"--",f"h {fl}"])
        for ix,fl in zip(span_feats["y"]["ix"],span_feats["y"]["fl"]):
            plot_samples.append([h_time,y[...,ix],cmap[fl],"-.",f"y {fl}"])

        domains,ylines,colors,linestyles,labels = zip(*plot_samples)

        fpath = fig_dir.joinpath(gen_span_path.stem+f"_{six}.png")
        plot_lines(
                domain=domains,
                ylines=ylines,
                fig_path=fpath,
                labels=labels,
                multi_domain=True,
                plot_spec={
                    "colors":colors,
                    "linestyle":linestyles,
                    "xtick_rotation":45,
                    "zero_axis":True,
                    },
                show=False
                )
        print(f"Generated {fpath.as_posix()}")
