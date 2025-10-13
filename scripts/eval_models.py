import numpy as np
import json
import torch
from pathlib import Path
from time import perf_counter

from emulate_era5_land.generators import PredictionDataset
from emulate_era5_land.ModelDir import ModelDir
from emulate_era5_land import evaluators
from emulate_era5_land.helpers import np_collate_fn

## configuration system for evaluator objects
eval_options = {
        "temporal":{
            "eval_type":"EvalTemporal",
            "manual_args":["eval_feats"],
            "defaults":{
                 "batch_axis":0, "reduce_func":None,
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

    :@param model_config: training configuration for the model being evaluated.
    :@param dataset_feats: Dict mapping dataset names to lists of string
        feature labels indicating the order of the batch array datasets.
    :@param eval_tuple: 3-tuple (category, label, args) where category is one
        of the evaluator type keys from the eval_options configuration, label
        is a unique string identifier describing this evaluator instance, and
        args is an iterable of positional values corresponding to the
        manual_args labels in the configuration.
    """
    eval_category,eval_label,manual_args = eval_tuple
    eval_cfg = eval_options[eval_category]

    ## Go ahead and build a dict of defaults that are commonly used so they
    defaults = {
            "pred_coarseness":model_config["feats"].get("pred_coarseness", 1),
            **eval_cfg["defaults"],
            }

    ## get the required arguments for this evaluator type
    required = evaluators.EVALUATORS[eval_cfg["eval_type"]].required()

    ## validate configuration structure
    assert all(k in required for k in eval_cfg["defaults"].keys()), \
        "All configured defaults must correspond to required arguments." \
        + f"\nrequired: {required}" \
        + f"\nprovided: {eval_cfg['defaults']}"
    assert len(manual_args) == len(eval_cfg["manual_args"]), \
        "A single positional argument must be provided for each of: " \
        + f"{eval_cfg['manual_args']}\nincompatible args: {manual_args}"

    ## make a dict of the positional manual arguments
    manual_args = dict(zip(eval_cfg["manual_args"], manual_args))

    ## iterate through the required args and make substitutions where needed
    eval_params = {}
    for a in required:
        if a in defaults.keys():
            assert a not in manual_args.keys(), a
            ## perform substitutions for shorthand arguments that depend on
            ## the particular model configuration
            if a=="time_slice":
                if defaults[a] == "horizon":
                    defaults[a] = (model_config["feats"]["window_size"], None)
                elif defaults[a] == "window":
                    defaults[a] = (0, model_config["feats"]["window_size"])
                elif defaults[a] == "full":
                    defaults[a] = (0, None)
            eval_params[a] = defaults[a]
        elif a in manual_args.keys():
            eval_params[a] = manual_args[a]
        else:
            raise ValueError(f"Required argument not provided: {a}")

    name_fields = ("eval", model_config["name"], eval_category, eval_label)
    ## declare and return the Evaluator based on its configured type.
    return evaluators.EVALUATORS[eval_cfg["eval_type"]](
            params=eval_params,
            feats=dataset_feats,
            meta={"model_config":model_config, "name":"_".join(name_fields)},
            )

if __name__=="__main__":
    proj_root = Path("/rhome/mdodson/emulate-era5-land")
    model_parent_dir = proj_root.joinpath("data/models")
    model_dir = model_parent_dir.joinpath("acclstm-era5-swm-9")
    model_name = "acclstm-era5-swm-9_state_0069.pwf"

    batch_size = 256
    num_batches = 16

    eval_tgs = [
            proj_root.joinpath(f"data/timegrids/timegrid_era5_{year}.h5")
            for year in range(2018,2024)
            ]
    assert all([tg.exists() for tg in eval_tgs])

    ## declare shorthand variables for common (dataset, feature) groupings
    f_intg = [ "swm-7", "swm-28", "swm-100", "swm-289" ]
    f_diff = [f"diff {k}" for k in f_intg]
    df_diff_err_bias = [("err-bias",k) for k in f_diff]
    df_diff_err_abs = [("err-abs",k) for k in f_diff]
    df_intg_err_bias = [("err-bias",k) for k in f_intg]
    df_intg_err_abs = [("err-abs",k) for k in f_intg]
    df_diff_pred = [("pred",k) for k in f_diff]
    df_intg_pred = [("pred",k) for k in f_intg]
    df_diff_true = [("target",k) for k in f_diff]
    df_intg_true = [("target",k) for k in f_intg]

    df_intg_all = df_intg_true+df_intg_pred+df_intg_err_bias+df_intg_err_abs
    df_diff_all = df_diff_true+df_diff_pred+df_diff_err_bias+df_diff_err_abs

    ## eval_{model}_{eval-type}_{instance-str}.pkl
    eval_config = [
        ("temporal", "intg-all", [df_intg_all]),
        ("temporal", "diff-all", [df_diff_all]),
        ]

    ## declare a prediction dataset based on the already-trained model
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

    ## declare a data loader for the prediction dataset
    pdl = torch.utils.data.DataLoader(
            dataset=pds,
            batch_size=batch_size,
            collate_fn=np_collate_fn,
            )

    ## get feat indeces of differentiated feats, which must be discretely
    ## integrated and concatented with the other data.
    diff_fixs = []
    integ_feats = []
    for ix,f in enumerate(md.config["feats"]["target_feats"]):
        if f.split(" ")[0]=="diff":
            diff_fixs.append(ix)
            integ_feats.append(" ".join(f.split(" ")[1:]))

    ## make an updated feature listing included the integrated output values
    dataset_feats = {
            "window":md.config["feats"]["window_feats"],
            "horizon":md.config["feats"]["horizon_feats"],
            "static":md.config["feats"]["static_feats"],
            "static-int":md.config["feats"]["static_int_feats"],
            "aux-dynamic":md.config["feats"]["aux_dynamic_feats"],
            "aux-static":md.config["feats"]["aux_static_feats"],
            "time":["epoch"],
            "target":md.config["feats"]["target_feats"]+integ_feats,
            "pred":md.config["feats"]["target_feats"]+integ_feats,
            "err-bias":md.config["feats"]["target_feats"]+integ_feats,
            "err-abs":md.config["feats"]["target_feats"]+integ_feats,
            }

    ## declare the Evaluator subclass objects
    evals = [get_eval_from_config(md.config, dataset_feats, ctup)
            for ctup in eval_config]

    for i in range(num_batches):
        ## unpack the batch data
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

        ## construct a dictionary with all relevant data from this batch
        bdict = {"window":w, "horizon":h, "static":s, "static-int":si,
                 "aux-dynamic":a_d, "aux-static":a_s, "time":t, "target":y,
                 "pred":p, "err-bias":e, "err-abs":np.abs(e),
                 }

        ## update the evaluators
        for ev in evals:
            ev.add_batch(bdict)

    ## save the evaluators as pkls
    for ev in evals:
        ev.to_pkl(pkl_dir.joinpath(ev.meta["name"]+".pkl"))
