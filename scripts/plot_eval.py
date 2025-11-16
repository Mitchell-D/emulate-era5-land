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

conf = {
    "EvalTemporal":{"temporal":{
        "diff-all":{"f":plot_lines_multiy, "ps":{}},
        "intg-all":{"f":plot_lines_multiy, "ps":{}},
        }},
    "EvalSampleSources":{
        "spatial":{
            "doy":{"f":plot_geo_scalar},
            "count":{"f":plot_geo_scalar},
            },
        "temporal":{
            "doy":{"f":plot_heatmap, "ps":{}},
            "all":{"f":plot_heatmap, "ps":{}},
            },
        },
    "EvalJointHist":{
        "hist-vc-swm-7":{"counts":{"f":plot_heatmap, "ps":{}}},
        "hist-vc-swm-28":{"counts":{"f":plot_heatmap, "ps":{}}},
        "hist-vc-swm-100":{"counts":{"f":plot_heatmap, "ps":{}}},
        "hist-vc-swm-289":{"counts":{"f":plot_heatmap, "ps":{}}},

        "hist-diff-swm-7":{"err-all":{"f":plot_heatmap, "ps":{}}},
        "hist-diff-swm-28":{"err-all":{"f":plot_heatmap, "ps":{}}},
        "hist-diff-swm-100":{"err-all":{"f":plot_heatmap, "ps":{}}},
        "hist-diff-swm-289":{"err-all":{"f":plot_heatmap, "ps":{}}},

        "hist-tmp-dwpt":{"err-all":{"f":plot_heatmap, "ps":{}}},
        "hist-tmp-snow":{"err-all":{"f":plot_heatmap, "ps":{}}},
        "hist-trsp-evp":{"err-all":{"f":plot_heatmap, "ps":{}}},
        },
    }

if __name__=="__main__":
    proj_root = Path("/rhome/mdodson/emulate-era5-land")
    model_parent_dir = proj_root.joinpath("data/models")
    static_pkl_path = proj_root.joinpath("data/static/era5_static.pkl")
    fig_dir = proj_root.joinpath("figures/eval")
    eval_dir = proj_root.joinpath("data/eval")

    #model_names = [f"acclstm-era5-swm-{mn}" for mn in range(37,66)
    #        if mn not in [46, 42, 43]] ## don't have layer-wise error
    #model_names = ["acclstm-era5-swm-50"]

    ## evaluators can be organized by:
    ## (subdomain, model, eval_type, eval_preset, eval_instance)
    ## since each evaluator type can have multiple presets, and presets
    ## can by applied to multiple different instance types.
    plot_models = [
        "acclstm-era5-swm-64",
        "acclstm-era5-swm-50",
        ]
    plot_subdomains = [
        "test-full"
        ]
    plot_instances = [ ## (preset, instance)
        ("temporal", "intg-all"),
        ("temporal", "err-all"),
        ("hist-diff-swm-7", "err-all"),
        ("hist-diff-swm-28", "err-all"),
        ("hist-diff-swm-100", "err-all"),
        ("hist-diff-swm-289", "err-all"),
        ("hist-tmp-twpt", "err-all"),
        ("hist-tmp-snow", "err-all"),
        ("hist-trsp-evp", "err-all"),
        ("hist-vc-swm-7", "counts"),
        ("hist-vc-swm-28", "counts"),
        ("hist-vc-swm-100", "counts"),
        ("hist-vc-swm-289", "counts"),
        ]
    plot_subcats = [
        "err-all", "counts", "diff-all", "intg-all",
        ]






    ev_paths = [
        (p,pt) for p,pt in ((q,q.stem.split("_")) for q in eval_dir.iterdir())
        if pt[0]=="eval"
        and pt[1] in plot_models
        and pt[2] in plot_instances
        and pt[3] in plot_subcats
        ]

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


