import numpy as np
import json
import torch
import pickle as pkl
from pathlib import Path
from time import perf_counter
from multiprocessing import Pool

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
        "sample-sources":{
            "eval_type":"EvalSampleSources",
            "manual_args":[],
            "defaults":{
                "vidx_feat":("auxs", "vidxs"),
                "hidx_feat":("auxs", "hidxs"),
                "time_feat":("time", "epoch"),
                "cov_feats":[],
                "cov_reduce_metric":None,
                "cov_reduce_axes":(1,),
                },
            },
        "hist-vc-swm-7":{ ## validation curve histograms
            "eval_type":"EvalJointHist",
            "manual_args":["cov_feats", "hist_conditions"],
            "defaults":{
                "axis_feats":[("target","swm-7"), ("pred","swm-7")],
                "axis_params":[(0,.8,256),(0,.8,256)],
                "round_oob":True,
                },
            },
        "hist-vc-swm-28":{ ## validation curve histograms
            "eval_type":"EvalJointHist",
            "manual_args":["cov_feats", "hist_conditions"],
            "defaults":{
                "axis_feats":[("target","swm-28"), ("pred","swm-28")],
                "axis_params":[(0,.8,256),(0,.8,256)],
                "round_oob":True,
                },
            },
        "hist-vc-swm-100":{ ## validation curve histograms
            "eval_type":"EvalJointHist",
            "manual_args":["cov_feats", "hist_conditions"],
            "defaults":{
                "axis_feats":[("target","swm-100"), ("pred","swm-100")],
                "axis_params":[(0,.8,256),(0,.8,256)],
                "round_oob":True,
                },
            },
        "hist-vc-swm-289":{ ## validation curve histograms
            "eval_type":"EvalJointHist",
            "manual_args":["cov_feats", "hist_conditions"],
            "defaults":{
                "axis_feats":[("target","swm-289"), ("pred","swm-289")],
                "axis_params":[(0,.8,256),(0,.8,256)],
                "round_oob":True,
                },
            },
        "hist-tmp-dwpt":{
            "eval_type":"EvalJointHist",
            "manual_args":["cov_feats", "hist_conditions"],
            "defaults":{
                "axis_feats":[("horizon","tmp"), ("horizon","dwpt")],
                "axis_params":[(220,320,256),(220,320,256)],
                "round_oob":True,
                },
            },
        "hist-diff-swm-7":{
            "eval_type":"EvalJointHist",
            "manual_args":["cov_feats", "hist_conditions"],
            "defaults":{
                "axis_feats":[("target","swm-7"), ("target","diff swm-7")],
                "axis_params":[(0,.8,256),(-.01,.01,256)],
                "round_oob":True,
                },
            },
        "hist-diff-swm-28":{
            "eval_type":"EvalJointHist",
            "manual_args":["cov_feats", "hist_conditions"],
            "defaults":{
                "axis_feats":[("target","swm-28"), ("target","diff swm-28")],
                "axis_params":[(0,.8,256),(-.005,.005,256)],
                "round_oob":True,
                },
            },
        "hist-diff-swm-100":{
            "eval_type":"EvalJointHist",
            "manual_args":["cov_feats", "hist_conditions"],
            "defaults":{
                "axis_feats":[("target","swm-100"), ("target","diff swm-100")],
                "axis_params":[(0,.8,256),(-.001,.001,256)],
                "round_oob":True,
                },
            },
        "hist-diff-swm-289":{
            "eval_type":"EvalJointHist",
            "manual_args":["cov_feats", "hist_conditions"],
            "defaults":{
                "axis_feats":[("target","swm-289"), ("target","diff swm-289")],
                "axis_params":[(0,.8,256),(-.0005,.0005,256)],
                "round_oob":True,
                },
            },
        "hist-tmp-snow":{
            "eval_type":"EvalJointHist",
            "manual_args":["cov_feats", "hist_conditions"],
            "defaults":{
                "axis_feats":[("horizon","tmp"), ("horizon","wm-snow")],
                "axis_params":[(220,320,256),(0,1,256)],
                "round_oob":True,
                },
            },
        "hist-trsp-evp":{
            "eval_type":"EvalJointHist",
            "manual_args":["cov_feats", "hist_conditions"],
            "defaults":{
                "axis_feats":[("auxd-h","evp-trsp"),
                    ("auxd-h","evp")],
                "axis_params":[(-.15,.05,256),(-.15,.05,256)],
                "round_oob":True,
                },
            },
        }

def mp_add_batch(args):
    """  """
    evt,bd = args
    return evaluators.Evaluator.from_tuple(evt).add_batch(bd).to_tuple()

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
    #model_name = "acclstm-era5-swm-9_state_0069.pwf"
    #model_name = "acclstm-era5-swm-50_state_0120.pwf"
    model_name = "acclstm-era5-swm-64_state_0024.pwf"
    model_dir = model_parent_dir.joinpath(model_name.split("_")[0])
    pkl_dir = proj_root.joinpath("data/eval")
    debug = True

    batch_size = 256
    prefetch_factor = 6
    #num_batches = 2048
    num_batches = 512
    save_every_nbatches = 32
    nworkers_dataset = 9
    nworkers_eval = 4

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
    df_intg_err = df_intg_err_bias + df_intg_err_abs
    df_diff_pred = [("pred",k) for k in f_diff]
    df_intg_pred = [("pred",k) for k in f_intg]
    df_diff_true = [("target",k) for k in f_diff]
    df_intg_true = [("target",k) for k in f_intg]
    df_err_all = [[
        ("err-bias", f), ("err-abs", f),
        ("err-bias", f"diff {f}"), ("err-abs", f"diff {f}"),
        ] for f in f_intg]

    df_intg_all = df_intg_true+df_intg_pred+df_intg_err_bias+df_intg_err_abs
    df_diff_all = df_diff_true+df_diff_pred+df_diff_err_bias+df_diff_err_abs

    ## eval_{model}_{eval-type}_{instance-str}.pkl
    eval_config = [
        ("temporal", "intg-all", [df_intg_all]),
        ("temporal", "diff-all", [df_diff_all]),
        ("hist-vc-swm-7", "counts", [[], []]),
        ("hist-vc-swm-28", "counts", [[], []]),
        ("hist-vc-swm-100", "counts", [[], []]),
        ("hist-vc-swm-289", "counts", [[], []]),
        ("hist-tmp-dwpt", "err-all", [df_intg_err, []]),
        ("hist-diff-swm-7", "err-all", [df_err_all[0], []]),
        ("hist-diff-swm-28", "err-all", [df_err_all[1], []]),
        ("hist-diff-swm-100", "err-all", [df_err_all[2], []]),
        ("hist-diff-swm-289", "err-all", [df_err_all[3], []]),
        ("hist-tmp-snow", "err-all", [df_err_all[0], []]),
        ("hist-trsp-evp", "err-all", [df_intg_err, []]),
        ]

    ## declare a prediction dataset based on the already-trained model
    md = ModelDir(model_dir)
    pds = PredictionDataset(
            model_path=model_dir.joinpath(model_name),
            use_dataset="eval",
            normalized_outputs=False,
            config_override={
                "feats":{
                    #"horizon_size":336,
                    "horizon_size":24*5,
                    "aux_dynamic_feats":list(set([
                        *md.config["feats"]["aux_dynamic_feats"],
                        "evp-trsp", "evp"])),
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
                        "num_workers":nworkers_dataset,
                        "prefetch_factor":prefetch_factor,
                        "out_dtype":"f4",
                        },
                    },
                },
            output_device="cpu",
            debug=False,
            )

    ## declare a data loader for the prediction dataset
    pdl = iter(torch.utils.data.DataLoader(
            dataset=pds,
            batch_size=batch_size,
            collate_fn=np_collate_fn,
            #num_workers=1,
            #prefetch_factor=1,
            ))

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
            "auxd-h":list(set([
                *md.config["feats"]["aux_dynamic_feats"], "evp-trsp", "evp"])),
            "auxd-w":list(set([
                *md.config["feats"]["aux_dynamic_feats"], "evp-trsp", "evp"])),
            "auxs":md.config["feats"]["aux_static_feats"],
            "time":["epoch"],
            "target":md.config["feats"]["target_feats"]+integ_feats,
            "pred":md.config["feats"]["target_feats"]+integ_feats,
            "err-bias":md.config["feats"]["target_feats"]+integ_feats,
            "err-abs":md.config["feats"]["target_feats"]+integ_feats,
            }

    ## declare the Evaluator subclass objects
    evals = [get_eval_from_config(md.config, dataset_feats, ctup)
            for ctup in eval_config]
    ev_names = [ev.meta["name"] for ev in evals]
    evals = [ev.to_tuple() for ev in evals]

    for i in range(num_batches):
        ## unpack the batch data
        if debug:
            _t0 = perf_counter()
        x,(y,),a,(p,) = next(pdl)
        if debug:
            _t1 = perf_counter()
            print(f"B{i+1:03} dataloader total: {_t1-_t0:.3f}")
        w,h,s,si,init = x
        a_d,a_s,t = a

        ## concatenate integrated versions of only the differentiated features.
        y = np.concatenate([
            y, init[...,diff_fixs]+np.cumsum(y[...,diff_fixs], axis=1)
            ], axis=-1)
        p = np.concatenate([
            p, init[...,diff_fixs]+np.cumsum(p[...,diff_fixs], axis=1)
            ], axis=-1)
        e = p-y

        wslice = slice(0,md.config["feats"]["window_size"])
        hslice = slice(-md.config["feats"]["horizon_size"],None)

        ## construct a dictionary with all relevant data from this batch
        bdict = {"window":w, "horizon":h, "static":s, "static-int":si,
                "auxd-w":a_d[:,wslice], "auxd-h":a_d[:,hslice],
                "auxs":a_s, "time":t, "target":y,
                "pred":p, "err-bias":e, "err-abs":np.abs(e), }

        ## update the evaluators
        if debug:
            _t0 = perf_counter()

        with Pool(nworkers_eval) as pool:
            evals = pool.map(mp_add_batch, [(ev,bdict) for ev in evals])
        #for ev in evals:
        #    ev.add_batch(bdict)
        if debug:
            _t1 = perf_counter()
            print(f"B{i+1:03} evaluator total: {_t1-_t0:.3f}")

        ## periodically save the evaluators as pkls
        if (i+1)%save_every_nbatches == 0:
            for en,ev in zip(ev_names,evals):
                pkl_path = pkl_dir.joinpath(en+".pkl")
                pkl.dump(ev, pkl_path.open("wb"))
                print(f"Wrote to {pkl_path.name}")

    ## Final evaluator save.
    for en,ev in zip(ev_names,evals):
        pkl_path = pkl_dir.joinpath(en+".pkl")
        pkl.dump(ev, pkl_path.open("wb"))
        print(f"Wrote to {pkl_path.name}")
