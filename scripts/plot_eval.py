""" """
#import torch
import copy
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
from emulate_era5_land.plotting import plot_lines_multiy
from emulate_era5_land.plotting import plot_lines_and_heatmap_split
from emulate_era5_land.plotting import plot_joint_hist_and_cov

from config_eval_plot import config

if __name__=="__main__":
    proj_root = Path("/rhome/mdodson/emulate-era5-land")
    model_parent_dir = proj_root.joinpath("data/models")
    static_pkl_path = proj_root.joinpath("data/static/era5_static.pkl")
    fig_dir = proj_root.joinpath("figures/eval-new")
    eval_dir = proj_root.joinpath("data/eval-new")

    ## evaluators can be organized by:
    ## (subdomain, model, eval_type, eval_preset, eval_instance)
    ## since each evaluator type can have multiple presets, and presets
    ## can by applied to multiple different instance types.
    plot_models = [
        #"acclstm-era5-swm-9",
        #"acclstm-era5-swm-50",
        "acclstm-era5-swm-64",
        ]
    plot_subdomains = [
        "full"
        ]
    plot_instances = { ## (etype, data_primary, data_secondary):plots
        #("EvalSampleSources", "space-time", "none"):[],

        #("EvalTemporal", "doy-tod", "intg-all"):[],
        #("EvalTemporal", "doy-tod", "diff-all"):[],

        ("EvalJointHist", "state-diff-swm-7", "err-diff"):["bias", "abs"],
        ("EvalJointHist", "state-diff-swm-28", "err-diff"):["bias", "abs"],
        #("EvalJointHist", "state-diff-swm-100", "err-diff"):["bias", "abs"],
        #("EvalJointHist", "state-diff-swm-289", "err-diff"):["bias", "abs"],

        ("EvalJointHist", "swm-7", "counts"):["vc"],
        ("EvalJointHist", "swm-28", "counts"):["vc"],
        #("EvalJointHist", "swm-100", "counts"):["vc"],
        #("EvalJointHist", "swm-289", "counts"):["vc"],

        ("EvalJointHist", "diff-swm-7", "counts"):["vc"],
        ("EvalJointHist", "diff-swm-28", "counts"):["vc"],
        #("EvalJointHist", "diff-swm-100", "counts"):["vc"],
        #("EvalJointHist", "diff-swm-289", "counts"):["vc"],

        ("EvalJointHist", "tmp-dwpt", "err-diff"):[
            "swm-7-bias", "swm-7-abs", "swm-7-abs-stddev"],
        #("EvalJointHist", "tmp-snow", "err-diff"):["swm-7-bias", "swm-7-abs"],
        #("EvalJointHist", "trsp-evp", "err-diff"):["swm-7-bias", "swm-7-abs"],

        #("EvalStatic", "grid", "err-mean"):[],
        #("EvalStatic", "grid", "err-max"):[],
        #("EvalStatic", "grid", "err-min"):[],
        #("EvalStatic", "veg-soil-combo", "err-mean"):[],
        }

    ## make sure each allowed plot instance has a configuration
    for pinstance in plot_instances:
        etype,d1,d2 = pinstance
        assert etype in config.keys(),pinstance
        assert d1 in config[etype].keys(),pinstance
        assert d2 in config[etype][d1].keys(),pinstance
        for ptype in config[etype][d1][d2]:
            assert ptype in config[etype][d1][d2].keys(),(pinstance,ptype)

    ## restrict the paths to the
    ev_paths = [
        (p,pt) for p,pt in ((q,q.stem.split("_")) for q in eval_dir.iterdir())
        if pt[0]=="eval"
        and pt[1] in plot_subdomains
        and pt[2] in plot_models
        and (pt[3],pt[4],pt[5]) in plot_instances
        ]

    print(f"Found {len(ev_paths)} valid evaluators")

    ## make a list of plottable evaluator pkl paths and the plot arguments,
    ## which are drawn from scripts/config_eval_plot.py
    to_plot = []
    for p,(_,subd,model,etype,d1,d2) in ev_paths:
        if model not in plot_models:
            print(f"Skipping (not in plot_models): {p.name}")
            continue
        if subd not in plot_subdomains:
            print(f"Skipping (not in plot_subdomains): {p.name}")
            continue
        pinst = (etype,d1,d2)
        if pinst not in plot_instances.keys():
            print(f"Skipping (not in plot_instances): {p.name}")
            continue
        cfg = config.get(etype,{}).get(d1,{}).get(d2,{})
        for ptype in plot_instances[pinst]:
            assert ptype in cfg.keys(), \
                f"Plot type {ptype} bust be configured for {pinst}"
            args = copy.deepcopy(cfg[ptype])
            ## update the arguments with a newly-created file path,
            ## which assumes d2 is not needed (when info captured by ptype)
            out_path = fig_dir.joinpath(
                f"eval_{subd}_{model}_{etype}_{d1}_{ptype}.png")
            args["fig_path"] = out_path
            to_plot.append((p, args))

    ## Plot each of the evaluator/argument combinations.
    for p,a in to_plot:
        _,subd,model,etype,d1,d2 = a["fig_path"].stem.split("_")
        *_,ptype = p.stem.split("_")

        ev = Evaluator.from_pkl(p)
        if "plot" not in dir(ev):
            print(f"WARNING: `plot` is not implemented for {etype}")
            continue
        if "plot_spec" in a.keys():
            for k,v in a["plot_spec"].items():
                if not isinstance(v,str):
                    continue
                a["plot_spec"][k] = v.format(
                    file_path=a["fig_path"].name,
                    subdomain=subd,
                    model=model,
                    eval_type=etype,
                    data_label=d1,
                    plot_type=ptype,
                    )
        ev.plot(**a)
