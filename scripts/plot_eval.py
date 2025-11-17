""" """
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
from emulate_era5_land.plotting import plot_lines_multiy
from emulate_era5_land.plotting import plot_lines_and_heatmap_split
from emulate_era5_land.plotting import plot_joint_hist_and_cov

plot_config = {
    "EvalJointHist":{
        ## temperature/dewpoint curve
        "tmp-dwpt":{"err-all":[
            ## swm-7 absolute error
            {"plot_type":"hist-cov",
                "cov_feats":[("err-abs","swm-7")],
                "plot_spec":{
                    "hist_title":"swm-7 MAE wrt temperature and dewpoint",
                    "xlabel":"Dewpoint (K)", "ylabel":"Temperature (K)",
                    "cov_title":"Mean Absolute Error in 0-7cm Layer"
                    }},

            ## swm-7 error bias
            {"plot_type":"hist-cov",
                "cov_feats":[("err-bias","swm-7")],
                "plot_spec":{}},

            ## swm-28 absolute error
            {"plot_type":"hist-cov",
                "cov_feats":[("err-abs","swm-28")],
                "plot_spec":{}},

            ## swm-28 error bias
            {"plot_type":"hist-cov",
                "cov_feats":[("err-bias","swm-28")],
                "plot_spec":{}},
            ]},

        ## swm-7 differential validation curve
        "diff-swm-7":{"counts":[
            {"plot_type":"hist", "plot_spec":{}},
            ]},

        ## true state and increment change histogram w error covariance
        "state-diff-swm-7":{"counts":[
            ## error bias
            {"plot_type":"hist-cov",
                "cov_feats":[("err-bias","swm-7")],
                "plot_spec":{}},

            ## absolute error
            {"plot_type":"hist-cov",
                "cov_feats":[("err-abs","swm-7")],
                "plot_spec":{}},
            ]},
        }
    }

if __name__=="__main__":
    proj_root = Path("/rhome/mdodson/emulate-era5-land")
    model_parent_dir = proj_root.joinpath("data/models")
    static_pkl_path = proj_root.joinpath("data/static/era5_static.pkl")
    fig_dir = proj_root.joinpath("figures/eval")
    eval_dir = proj_root.joinpath("data/eval")

    ## evaluators can be organized by:
    ## (subdomain, model, eval_type, eval_preset, eval_instance)
    ## since each evaluator type can have multiple presets, and presets
    ## can by applied to multiple different instance types.
    plot_models = [
        "acclstm-era5-swm-9",
        "acclstm-era5-swm-50",
        "acclstm-era5-swm-64",
        ]
    plot_subdomains = [
        "full"
        ]
    plot_instances = [ ## (preset, instance)
        ("EvalSampleSources", "space-time", "none"),
        ("EvalTemporal", "doy-tod", "intg-all"),
        ("EvalTemporal", "doy-tod", "diff-all"),
        ("EvalJointHist", "state-diff-swm-7", "err-diff"),
        ("EvalJointHist", "state-diff-swm-28", "err-diff"),
        ("EvalJointHist", "state-diff-swm-100", "err-diff"),
        ("EvalJointHist", "state-diff-swm-289", "err-diff"),
        ("EvalJointHist", "swm-7", "counts"),
        ("EvalJointHist", "swm-28", "counts"),
        ("EvalJointHist", "swm-100", "counts"),
        ("EvalJointHist", "swm-289", "counts"),
        ("EvalJointHist", "tmp-dwpt", "err-diff"),
        ("EvalJointHist", "tmp-snow", "err-diff"),
        ("EvalJointHist", "trsp-evp", "err-diff"),
        ("EvalStatic", "grid", "err-mean"),
        ("EvalStatic", "grid", "err-max"),
        ("EvalStatic", "grid", "err-min"),
        ("EvalStatic", "veg-soil-combo", "err-mean"),
        ]


    ev_paths = [
        (p,pt) for p,pt in ((q,q.stem.split("_")) for q in eval_dir.iterdir())
        if pt[0]=="eval"
        and pt[1] in plot_subdomains
        and pt[2] in plot_models
        and (pt[3],pt[4],pt[5]) in plot_instances
        ]

    to_plot = []
    for p in ev_paths:
        _,subd,model,etype,d1,d2 = p.stem.split("_")
        if model not in plot_models:
            print(f"Skipping not in plot_models: {model}")
            continue
        if subd not in plot_subdomains:
            print(f"Skipping not in plot_subdomains: {subd}")
            continue
        if (etype,d1,d2) not in plot_instances:
            print(f"Skipping not in plot_instances: {(etype,d1,d2)}")
            continue
        args = plot_config.get(etype,{}).get(d1,{}).get(d2,{})
        to_plot.append((p, args))

    for p,a in to_plot:
        ev = evaluators.Evaluator.from_pkl(p)
        if "plot" not in dir("ev"):
            print(f"WARNING: `plot` is not implemented for {etype}")
            continue
        ev.plot(**a)

    exit(0)
    for p,_ in ev_paths:
        ev = Evaluator.from_pkl(p)
        if ev.evtype=="EvalJointHist":
            fr = ev.final_results()
            fig_path = fig_dir.joinpath(p.stem+".png")
            cov = None if not fr["cov_mean"] else fr["cov_mean"][0][...,0]
            plot_joint_hist_and_cov(
                    counts=fr["counts"][0],
                    ax1_params=ev.params["axis_params"][0],
                    ax2_params=ev.params["axis_params"][1],
                    covariate=cov,
                    plot_covariate=True,
                    fig_path=fig_path,
                    separate_covariate_axes=True,
                    )
            print(f"Generated {fig_path.name}")

    exit(0)

    for model_name in model_names:
        print(f"Plotting {model_name}")
        model_path = model_parent_dir.joinpath(model_name)
        if not model_path.exists():
            print(f"Skipping missing model directory: {model_path.name = }")
        md = ModelDir(model_path)
        fd = fig_dir.joinpath(model_name)
        if not fd.exists():
            fd.mkdir()


