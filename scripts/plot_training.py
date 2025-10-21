"""
Plotting methods for visualizing training processes, including:

    1. individual sequence samples retained by the model
    2. spatiotemporal distribution of t/v samples
    3. learning rate plotted against training and validation metrics
"""
#import torch
import numpy as np
from pathlib import Path
from datetime import datetime,timedelta,timezone
import json
import pickle as pkl
import torch
import matplotlib.colors as cm
import matplotlib.pyplot as plt

from emulate_era5_land.helpers import np_collate_fn
from emulate_era5_land.evaluators import Evaluator
from emulate_era5_land.ModelDir import ModelDir
from emulate_era5_land.plotting import plot_heatmap,plot_lines,plot_geo_scalar

if __name__=="__main__":
    proj_root = Path("/rhome/mdodson/emulate-era5-land")
    model_parent_dir = proj_root.joinpath("data/models")
    static_pkl_path = proj_root.joinpath("data/static/era5_static.pkl")
    fig_dir = proj_root.joinpath("figures/training")
    model_name = "acclstm-era5-swm-23"
    md = ModelDir(model_parent_dir.joinpath(model_name))

    grid_domain_shape = (261,586)
    plot_retained_samples = 16

    ## plot learning curves
    #'''
    lc_metrics_simple = json.load(md.dir.joinpath(
        f"{md.name}_metrics_simple.json").open("r"))
    lc_simple = {}
    for i in range(len(lc_metrics_simple["train_epochs"])):
        for k in lc_metrics_simple["train"].keys():
            if k not in lc_simple.keys():
                lc_simple[k] = []
            lc_simple[k].append([
                lc_metrics_simple["train"][k][i],
                lc_metrics_simple["val"][k][i],
                ])

    ## shaped (epoch,)
    lr = np.squeeze(np.array(lc_metrics_simple["lr"]))

    ## each shaped (epoch, t/v, mean/stddev)
    for k in lc_simple.keys():
        fig_path = fig_dir.joinpath(f"{md.name}_lc_epoch_{k}.png")
        lc_simple[k] = np.asarray(lc_simple[k])
        plot_lines_multiy(
                domain=np.arange(lr.shape[0]),
                ylines=[
                    [lc_simple[k][:,0,0],lc_simple[k][:,1,0], ## t/v mean
                        lc_simple[k][:,0,1],lc_simple[k][:,1,1]], ## t/v stddev
                    [lr]
                    ],
                fig_path=fig_path,
                plot_spec={
                    "title":f"{md.name}",
                    "xlabel":"Epoch",
                    "ylabel":k,
                    "y_labels":["Loss", "Learning Rate"]
                    "y_ranges":[None,(1e-5,0.1)],
                    "y_scales":["linear", "log"],
                    "linestyle":["-","-","-.","-.","-"],
                    "line_labels":["train mean", "val mean",
                        "train stddev", "val stddev", "lr"]
                    "line_colors":[
                        "blue", "orange", "blue", "orange", "black"],
                    }
                )
        print(f"Generated {fig_path.name}")

    ## load and sort pkls containing batch-wise mean error values.
    lc_pkls_t = list(sorted([
        (p,p.stem.split("_")) for p in md.dir.iterdir()
        if f"{md.name}_metrics_train" in p.stem
        ], key=lambda p:int(p[1][-1])))
    lc_pkls_v = list(sorted([
        (p,p.stem.split("_")) for p in md.dir.iterdir()
        if f"{md.name}_metrics_val" in p.stem
        ], key=lambda p:int(p[1][-1])))

    ## make dicts mapping the metrics to simple arrays
    lc_all = {}
    nbatches = None
    for (lctp,_),(lcvp,_) in zip(lc_pkls_t,lc_pkls_v):
        lct,lcv = pkl.load(lctp.open("rb")),pkl.load(lcvp.open("rb"))
        for k in lct.keys():
            if k not in lc_all.keys():
                lc_all[k] = []
            if nbatches is None:
                nbatches = len(lct[k])
            lc_all[k].append([lct[k],lcv[k]])

    ## each shaped (epoch, batch, t/v)
    lr = np.stack(
        [lr for i in range(nbatches)],
        axis=1,
        ).reshape(-1)
    for k in lc_all.keys():
        fig_path = fig_dir.joinpath(f"{md.name}_lc_batch_{k}.png")
        lc_all[k] = np.asarray(lc_all[k]).transpose(0,2,1).reshape(-1,2)
        plot_lines(
                domain=np.arange(lr.shape[0]),
                ylines=[lc_all[k][:,0],lc_all[k][:,1],lr],
                labels=["training", "validation", "lr"],
                fig_path=fig_path,
                plot_spec={
                    "title":f"{md.name}",
                    "xlabel":"Batch",
                    "ylabel":k,
                    "colors":["blue", "orange", "black"],
                    }
                )
        print(f"Generated {fig_path.name}")
    #'''

    ## plot a subset of samples retained by the model during training.
    '''
    inputs,outputs= np_collate_fn(pkl.load(
            md.dir.joinpath(f"{model_name}_samples.pkl").open("rb"),
            #map_location=torch.device("cpu"),
            ))
    window,horizon,static,static_int = inputs
    target,pred,cycle = outputs
    ## randomly decide which retained samples to extract
    sixs = np.random.choice(
            a=np.arange(window.shape[0]),
            size=plot_retained_samples,
            replace=False
            )
    dom_w = np.arange(0,window.shape[1])
    dom_h = np.arange(window.shape[1],window.shape[1]+horizon.shape[1])

    feats = ["pres", "tmp", "dwpt", "dlwrf", "dswrf"]
    for six in sixs:
        domains,ylines,colors,labels = [],[],[],[]
        for i,fl in enumerate(feats):
            c = cm.to_hex(plt.cm.Dark2(i))
            fix_w = md.config["feats"]["window_feats"].index(fl)
            fix_h = md.config["feats"]["horizon_feats"].index(fl)
            domains += [dom_w, dom_h]
            ylines += [window[six,:,fix_w], horizon[six,:,fix_h]]
            colors += [c, c]
            labels += [fl, fl]
        name_fields = [md.name, "samples", "forcings-1", f"{six:04}"]
        fig_path = fig_dir.joinpath("_".join(name_fields)+".png")
        plot_lines(
            domain=domains, ylines=ylines, labels=labels, multi_domain=True,
            plot_spec={"colors":colors, "xtick_rotation":45, "zero_axis":True,
                "title":" ".join(name_fields), },
            fig_path=fig_path,
            )
        print(f"Generated {fig_path.name}")

    feats = ["apcp", "alb", "wm-snow", "lai-low", "lai-high"]
    colors = [cm.to_hex(plt.cm.Dark2(i)) for i in range(len(feats))]
    for six in sixs:
        domains,ylines,colors,labels = [],[],[],[]
        for i,fl in enumerate(feats):
            c = cm.to_hex(plt.cm.Dark2(i))
            fix_w = md.config["feats"]["window_feats"].index(fl)
            fix_h = md.config["feats"]["horizon_feats"].index(fl)
            domains += [dom_w, dom_h]
            ylines += [window[six,:,fix_w], horizon[six,:,fix_h]]
            colors += [c, c]
            labels += [fl, fl]
        name_fields = [md.name, "samples", "forcings-2", f"{six:04}"]
        fig_path = fig_dir.joinpath("_".join(name_fields)+".png")
        plot_lines(
            domain=domains, ylines=ylines, labels=labels, multi_domain=True,
            plot_spec={"colors":colors, "xtick_rotation":45, "zero_axis":True,
                "title":" ".join(name_fields), },
            fig_path=fig_path,
            )
        print(f"Generated {fig_path.name}")

    feats = ["swm-7", "swm-28", "swm-100", "swm-289"]
    colors = [cm.to_hex(plt.cm.Dark2(i)) for i in range(len(feats))]
    for six in sixs:
        domains,ylines,colors,labels,linestyle = [],[],[],[],[]
        for i,fl in enumerate(feats):
            c = cm.to_hex(plt.cm.Dark2(i))
            fix_w = md.config["feats"]["window_feats"].index(fl)
            fix_h = md.config["feats"]["target_feats"].index(f"diff {fl}")
            ## note shifting the cycled feats back to their relevant timestep
            domains += [dom_w, dom_h, dom_h, dom_h-1]
            ylines += [window[six,:,fix_w], target[six,:,fix_h],
                    pred[six,:,fix_h], cycle[six,:,fix_h]]
            colors += [c, c, c, c]
            labels += [f"w {fl}", f"t diff {fl}", f"p diff {fl}", f"c {fl}"]
            linestyle += ["-","-","--","-."]
        name_fields = [md.name, "samples", "outputs-1", f"{six:04}"]
        fig_path = fig_dir.joinpath("_".join(name_fields)+".png")
        plot_lines(
            domain=domains, ylines=ylines, labels=labels, multi_domain=True,
            plot_spec={"colors":colors, "linestyle":linestyle,
                "xtick_rotation":45, "zero_axis":True,
                "title":" ".join(name_fields) },
            fig_path=fig_path,
            )
        print(f"Generated {fig_path.name}")

    '''

    ## load+plot training and validation sample source evaluators
    '''
    ss_evs = [
        Evaluator.from_pkl(md.dir.joinpath(p)) for p in [
            f"{md.name}_sample-sources_train.pkl",
            f"{md.name}_sample-sources_val.pkl",
            ]]
    ## load latitude and longitude from static data
    slabel,sdata = pkl.load(static_pkl_path.open("rb"))
    lat = sdata[slabel.index("lat")]
    lon = sdata[slabel.index("lon")]
    m_valid = sdata[slabel.index("m_valid")].astype(bool)
    for ev in ss_evs:
        gd_count = np.full(grid_domain_shape, 0, dtype=np.uint16)
        gd_batch = np.full(grid_domain_shape, 0, dtype=np.uint16)
        res = ev.final_results(time_format="ydh")
        doys_bixs = np.full((len(res["vidxs"]), 366), 0)
        doys = []
        bixs = []
        for bix in range(len(res["vidxs"])):
            for six in range(res["vidxs"][bix].shape[0]):
                gd_count[res["vidxs"][bix][six],res["hidxs"][bix][six]] += 1
                gd_batch[res["vidxs"][bix][six],res["hidxs"][bix][six]] = bix
            ## zero-indexing sequence axis shouldn't be needed for new evs
            doys_bixs[bix, res["times"][bix][...,1]] += 1

        fig_path = fig_dir.joinpath(ev.meta["name"]+"_doy-bixs.png")
        plot_heatmap(
            heatmap=np.where(doys_bixs==0,np.nan,doys_bixs),
            plot_spec={
                "title":f"Sample DoY vs Batch ({ev.meta['name']})",
                "cmap":"turbo",
                "xlabel":"Sample initialization DoY",
                "ylabel":"Batch number",
                "imshow_aspect":.1,
                },
            fig_path=fig_path,
            )
        print(f"Generated {fig_path.name}")
        fig_path = fig_dir.joinpath(ev.meta["name"]+"_spatial-counts.png")
        plot_geo_scalar(
            data=np.where(m_valid,gd_count,np.nan),
            latitude=lat,
            longitude=lon,
            plot_spec={
                "title":f"Spatial Sample Counts ({ev.meta['name']})",
                "cmap":"gnuplot",
                "cbar_label":"Number of samples",
                "cbar_orient":"horizontal",
                "cbar_shrink":.9,
                "cbar_pad":.02,
                "fontsize_title":18,
                "fontsize_labels":14,
                },
            fig_path=fig_path,
            )
        print(f"Generated {fig_path.name}")
        fig_path = fig_dir.joinpath(ev.meta["name"]+"_spatial-batch.png")
        plot_geo_scalar(
            data=np.where(m_valid,gd_batch,np.nan),
            latitude=lat,
            longitude=lon,
            plot_spec={
                "title":f"Most Recent Sample Batch ({ev.meta['name']})",
                "cmap":"gist_rainbow",
                "cbar_label":"Sample batch",
                "cbar_orient":"horizontal",
                "cbar_shrink":.9,
                "cbar_pad":.02,
                "fontsize_title":18,
                "fontsize_labels":14,
                },
            fig_path=fig_path,
            )
        print(f"Generated {fig_path.name}")
    '''

