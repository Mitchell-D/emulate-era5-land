import numpy as np
import json
import torch
from pathlib import Path
from time import perf_counter

from emulate_era5_land.generators import PredictionDataset
import emulate_era5_land.evaluators
from emulate_era5_land.helpers import np_collate_fn

## configuration system for evaluator objects
eval_options = {
        "temporal":{
            "eval_type":evaluators.EvalTemporal,
            "default_args":["attrs", "time_feat", "time_axis", "time_slice"],
            "required_args":evaluators.EvalTemporal.required,
            "defaults":{
                "time_feat":("time", "epoch"), "time_axis":1,
                "time_slice":"horizon",
                },
            },
        }

'''
        "horizon":{
            "eval_type":evaluators.EvalHorizon,
            "default_args":["pred_coarseness", "attrs"],
            "required_args":[],
            "defaults":{},
            },
        "static-combos":{
            "eval_type":evaluators.EvalStatic,
            "default_args":["soil_idxs", "attrs"],
            "required_args":["use_absolute_error"],
            "defaults":{},
            },
        "efficiency":{
            "eval_type":evaluators.EvalEfficiency,
            "default_args":["pred_coarseness", "attrs"],
            "required_args":["pred_feat"],
            "defaults":{},
            },
        "hist-true-pred":{
            "eval_type":evaluators.EvalJointHist,
            "default_args":[
                "ax1_hbounds", "ax2_hbounds", "ax1_hres", "ax2_hres", "attrs"],
            "required_args":["ax1", "ax2"],
            "defaults":{"ax1_dataset":"target", "ax2_dataset":"pred"},
            },
        "hist-saturation-error":{
            "eval_type":evaluators.EvalJointHist,
            "default_args":[
                "ax1_hbounds", "ax2_hbounds", "ax1_hres", "ax2_hres",
                "cov_dataset", "cov_feat", "pred_coarseness",
                "use_absolute_error", "attrs"],
            "required_args":["ax1", "ax2"],
            "defaults":{
                "ax1_dataset":"target", "ax2_dataset":"pred",
                "use_absolute_error":False, "cov_dataset":"error"},
            },
        #"hist-infiltration":{
        #    "eval_type":evaluators.EvalJointHist,
        #    },
        "hist-state-increment":{
            "eval_type":evaluators.EvalJointHist,
            "default_args":[
                "ax1_dataset", "ax2_dataset", "ax1_feat", "ax2_feat",
                "ax1_hbounds", "ax2_hbounds", "ax1_hres", "ax2_hres",
                "cov_dataset", "pred_coarseness", "ignore_nan", "attrs"],
            "required_args":["cov_feat","use_absolute_error"],
            "defaults":{
                "ax1_dataset":"target", "ax2_dataset":"target",
                "cov_dataset":"error", "ignore_nan":True,
                },
            },
        "hist-humidity-temp":{
            "eval_type":evaluators.EvalJointHist,
            "default_args":[
                "ax1_dataset", "ax2_dataset", "ax1_feat", "ax2_feat",
                "ax1_hbounds", "ax2_hbounds", "ax1_hres", "ax2_hres",
                "cov_dataset", "pred_coarseness", "ignore_nan", "attrs"],
            "required_args":["cov_feat","use_absolute_error"],
            "defaults":{
                "ax1_dataset":"horizon", "ax1_feat":"dwpt",
                "ax2_dataset":"horizon", "ax2_feat":"tmp",
                "cov_dataset":"error", "coarse_reduce_func":"mean",
                "ignore_nan":True,
                },
            },
'''

def get_eval_from_config(model_config, dataset_feats, eval_tuple):
    """
    Given a model configuration dict, a dict enumerating the features in each
    category, and a single tuple providing a evaluator instance type and
    non-default arguments for initializing it, return an instantiated subclass
    of the Evaluator which is prepared to recieve batch data.

    This method wraps some stinky code that handles Evaluator parameters which
    are dependent on model configuration parameters in order to keep the
    Evaluator subclasses and instance configuration data and model agnostic.
    """
    eval_category,*eval_positionals = eval_tuple
    eval_cfg = eval_options[eval_category]

    ## Go ahead and build a dict of defaults that are commonly used so they
    candidate_defaults = {
            "pred_coarseness":model_config["feats"].get("pred_coarseness", 1),
            "attrs":{"model_config":model_config},
            **eval_cfg["defaults"]
            }

    ## iterate through the default args
    eval_args = {}
    '''
    for a in eval_cfg["default_args"]:
        ## if a default argument is provided,
        if a in defaults.keys():
            eval_args[a] = defaults[a]
        elif a in ("ax1_dataset", "ax2_dataset"):
    '''
    for a in eval_cfg["required_args"]:
        if a in eval_cfg["default_args"].keys():
            ## perform substitutions for shorthand arguments that depend on
            ## the particular model configuration
            if a=="time_slice":
                if eval_cfg["default_args"][a] == "horizon":
                    eval_cfg["default_args"][a] = (
                            model_config["feats"]["window_size"], None)
                elif eval_cfg["default_args"][a] == "window":
                    eval_cfg["default_args"][a] = (
                            0, model_config["feats"]["window_size"])
                elif eval_cfg["default_args"][a] == "full":
                    eval_cfg["default_args"][a] = (0, None)

if __name__=="__main__":
    proj_root = Path("/rhome/mdodson/emulate-era5-land")
    model_parent_dir = proj_root.joinpath("data/models")
    model_dir = model_parent_dir.joinpath("acclstm-era5-swm-9")
    model_name = "acclstm-era5-swm-9_state_0069.pwf"

    batch_size = 256

    eval_tgs = [
            proj_root.joinpath(f"data/timegrids/timegrid_era5_{year}.h5")
            for year in range(2018,2024)
            ]
    assert all([tg.exists() for tg in eval_tgs])

    base_feats = ["swm-7", "swm-28", "swm-100", "swm-289"]
    pred_feats = ["diff swm-7", "diff swm-28", "diff swm-100", "diff swm-289"]
    eval_config = [
        ("horizon",),
        ("temporal", True),
        ("static-combos", True),
        ("static-combos", False),
        *[("efficiency", k) for k in pred_feats]
        *[("hist-true-pred", ("target", k), ("pred", k)) for k in pred_feats]
        *[("hist-saturation-error", ("target",bk), ("error",k))
          for bk,k in zip(base_feats,pred_feats)],
        ("hist-state-increment", "")
        ]

    md = ModelDir(model_dir)
    pds = PredictionDataset(
            model_path=model_dir.joinpath(model_name),
            use_dataset="eval",
            normalized_outputs=False,
            config_override={
                "feats":{
                    "horizon_size":336,
                    },
                "data":{
                    "eval":{
                        "timegrids":eval_tgs,
                        "shuffle":True,
                        "sample_cutoff":1.,
                        "sample_across_files":True,
                        "sample_under_cutoff":True,
                        "sample_separation":409,
                        "random_offset":True,
                        "chunk_pool_count":48,
                        "buf_size_mb":4096,
                        "buf_slots":48,
                        "buf_policy":0,
                        "batch_size":batch_size,
                        "num_workers":8,
                        "prefetch_factor":6,
                        "out_dtype":"f4",
                        },
                    },
                },
            output_device="cpu",
            )

    pdl = torch.utils.data.DataLoader(
            dataset=pds,
            batch_size=batch_size,
            collate_fn=np_collate_fn,
            )

    ## get feat indeces of differentiated feats, which must be discretely
    ## integrated and concatented with the other data.
    diff_fixs = []
    integ_feats = []
    for ix,f in enumerate(model_config["feats"]["target_feats"]):
        if f.split(" ")[0]=="diff":
            diff_fixs.append(ix)
            integ_feats.append(" ".join(f.split(" ")[1:]))
    ## make an updated feature listing included the integrated output values
    dataset_feats = {
            "window":model_config["feats"]["window_feats"],
            "horizon":model_config["feats"]["horizon_feats"],
            "target":model_config["feats"]["target_feats"]+integ_feats,
            "pred":model_config["feats"]["target_feats"]+integ_feats,
            "error":model_config["feats"]["target_feats"]+integ_feats,
            "aux_dynamic":model_config["feats"]["aux_dynamic_feats"],
            "aux_static":model_config["feats"]["aux_static_feats"],
            "time":["epoch"],
            }
    for i in range(32):
        x,(y,),a,(p,) = next(pdl)
        w,h,s,si,init = x
        a_d,a_s,t = a

        ## concatenate integrated versions of only the differentiated features.
        y = np.concatenate([
            y, init[...,diff_fixs]+np.cumsum(y[...,diff_fixs], axis=1)
            ], axis=-1)
        p = np.concatenate([
            p, init[...,diff_fixs]+np.cumsum(p[...,diff_fixs], axis=1)
            ], axis=-1)
        e = y-p

        bdict = {"window":w, "horizon":h, "static":s, "static_int":si,
                 "aux_dynamic":a_d, "aux_static":a_s, "time":t, "error":e,
                 "target":y, "pred":p, "error":e,
                 }
        print(p[0].shape)
