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
        ("EvalTemporal","doy-tod"):{
            "manual_args":["eval_feats"],
            "defaults":{
                 "batch_axis":0, "reduce_func":None,
                 "time_feat":("time", "epoch"), "time_axis":1,
                 "time_slice":"horizon",
                },
            },
        ("EvalSampleSources","space-time"):{
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
        ("EvalJointHist","diff-swm-7"):{
            "manual_args":["cov_feats", "hist_conditions"],
            "defaults":{
                "axis_feats":[
                    ("target","diff swm-7"), ("pred","diff swm-7")],
                "axis_params":[(-.001,.001,256),(-.1,.1,256)],
                "round_oob":True,
                },
            },
        ("EvalJointHist","diff-swm-28"):{
            "manual_args":["cov_feats", "hist_conditions"],
            "defaults":{
                "axis_feats":[
                    ("target","diff swm-28"), ("pred","diff swm-28")],
                "axis_params":[(-.0005,.0005,256),(-.05,.05,256)],
                "round_oob":True,
                },
            },
        ("EvalJointHist","diff-swm-100"):{
            "manual_args":["cov_feats", "hist_conditions"],
            "defaults":{
                "axis_feats":[
                    ("target","diff swm-100"), ("pred","diff swm-100")],
                "axis_params":[(-.0002,.0002,256),(-.02,.02,256)],
                "round_oob":True,
                },
            },
        ("EvalJointHist","diff-swm-289"):{
            "manual_args":["cov_feats", "hist_conditions"],
            "defaults":{
                "axis_feats":[
                    ("target","diff swm-289"), ("pred","diff swm-289")],
                "axis_params":[(-.0001,.0001,256),(-.01,.01,256)],
                "round_oob":True,
                },
            },
        ("EvalJointHist","swm-7"):{ ## validation curve histograms
            "manual_args":["cov_feats", "hist_conditions"],
            "defaults":{
                "axis_feats":[("target","swm-7"), ("pred","swm-7")],
                "axis_params":[(0,.8,256),(0,.8,256)],
                "round_oob":True,
                },
            },
        ("EvalJointHist","swm-28"):{ ## validation curve histograms
            "manual_args":["cov_feats", "hist_conditions"],
            "defaults":{
                "axis_feats":[("target","swm-28"), ("pred","swm-28")],
                "axis_params":[(0,.8,256),(0,.8,256)],
                "round_oob":True,
                },
            },
        ("EvalJointHist","swm-100"):{ ## validation curve histograms
            "manual_args":["cov_feats", "hist_conditions"],
            "defaults":{
                "axis_feats":[("target","swm-100"), ("pred","swm-100")],
                "axis_params":[(0,.8,256),(0,.8,256)],
                "round_oob":True,
                },
            },
        ("EvalJointHist","swm-289"):{ ## validation curve histograms
            "manual_args":["cov_feats", "hist_conditions"],
            "defaults":{
                "axis_feats":[("target","swm-289"), ("pred","swm-289")],
                "axis_params":[(0,.8,256),(0,.8,256)],
                "round_oob":True,
                },
            },
        ("EvalJointHist","tmp-dwpt"):{
            "manual_args":["cov_feats", "hist_conditions"],
            "defaults":{
                "axis_feats":[("horizon","tmp"), ("horizon","dwpt")],
                "axis_params":[(220,320,256),(220,320,256)],
                "round_oob":True,
                },
            },
        ("EvalJointHist","state-diff-swm-7"):{
            "manual_args":["cov_feats", "hist_conditions"],
            "defaults":{
                "axis_feats":[("target","swm-7"), ("target","diff swm-7")],
                "axis_params":[(0,.8,256),(-.001,.001,256)],
                "round_oob":True,
                },
            },
        ("EvalJointHist","state-diff-swm-28"):{
            "manual_args":["cov_feats", "hist_conditions"],
            "defaults":{
                "axis_feats":[("target","swm-28"), ("target","diff swm-28")],
                "axis_params":[(0,.8,256),(-.0005,.0005,256)],
                "round_oob":True,
                },
            },
        ("EvalJointHist","state-diff-swm-100"):{
            "manual_args":["cov_feats", "hist_conditions"],
            "defaults":{
                "axis_feats":[("target","swm-100"), ("target","diff swm-100")],
                "axis_params":[(0,.8,256),(-.0001,.0001,256)],
                "round_oob":True,
                },
            },
        ("EvalJointHist","state-diff-swm-289"):{
            "manual_args":["cov_feats", "hist_conditions"],
            "defaults":{
                "axis_feats":[("target","swm-289"), ("target","diff swm-289")],
                "axis_params":[(0,.8,256),(-.00005,.00005,256)],
                "round_oob":True,
                },
            },
        ("EvalJointHist","tmp-snow"):{
            "manual_args":["cov_feats", "hist_conditions"],
            "defaults":{
                "axis_feats":[("horizon","tmp"), ("horizon","wm-snow")],
                "axis_params":[(220,320,256),(0,1,256)],
                "round_oob":True,
                },
            },
        ("EvalJointHist","trsp-evp"):{
            "manual_args":["cov_feats", "hist_conditions"],
            "defaults":{
                "axis_feats":[("auxd-h","evp-trsp"),
                    ("auxd-h","evp")],
                "axis_params":[(-.15,.05,256),(-.15,.05,256)],
                "round_oob":True,
                },
            },

        ("EvalStatic","grid"):{
            "manual_args":["data_feats", "reduce_func",
                "collect_mean_var", "collect_min_max"],
            "defaults":{
                "static_feats":[("auxs","vidxs"), ("auxs","hidxs")],
                "static_values":["vidxs","hidxs"],
                },
            },
        ("EvalStatic","veg-soil-combos"):{
            "manual_args":["data_feats", "reduce_func",
                "collect_mean_var", "collect_min_max"],
            "defaults":{
                "static_feats":[("auxs","vt-low"),
                    ("auxs","vt-high"), ("auxs","soilt")],
                "static_values":["vt-low","vt-high", "soilt"],
                },
            },
        }

def mp_add_batch(args):
    """  """
    evt,bd = args
    return evaluators.Evaluator.from_tuple(evt).add_batch(bd).to_tuple()

def get_eval_from_config(subdomain:str, model_config, dataset_feats, eval_tuple,
        static_labels=None, static_data=None):
    """
    Given a model configuration dict, a dict enumerating the features in each
    category, and a single tuple providing a evaluator instance type and
    non-default arguments for initializing it, return an instantiated subclass
    of the Evaluator which is prepared to recieve batch data.

    This method wraps some stinky code that handles Evaluator parameters which
    are dependent on model configuration parameters in order to keep the
    Evaluator subclasses and instance configuration data and model agnostic.

    :@param subdomain: string describing the data source.
    :@param model_config: training configuration for the model being evaluated.
    :@param dataset_feats: Dict mapping dataset names to lists of string
        feature labels indicating the order of the batch array datasets.
    :@param eval_tuple: 3-tuple (category, label, args) where category is one
        of the evaluator type keys from the eval_options configuration, label
        is a unique string identifier describing this evaluator instance, and
        args is an iterable of positional values corresponding to the
        manual_args labels in the configuration.
    :@param static_labels: List of labels for each static data feature. These
        must be provided if using EvalStatic config substitutions.
    :@param static_data: Nx(Fs,) list of arrays of global static pixel data for
        reference when creating static coordinate arrays.
    """
    eval_class,eval_d1,eval_d2,manual_args = eval_tuple
    eval_cfg = eval_options[(eval_class,eval_d1)]

    ## Go ahead and build a dict of defaults that are commonly used so they
    defaults = {
            "pred_coarseness":model_config["feats"].get("pred_coarseness", 1),
            **eval_cfg["defaults"],
            }

    ## get the required arguments for this evaluator type
    required = evaluators.EVALUATORS[eval_class].required()

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
        if a=="static_values":
            assert isinstance(eval_params[a], (list,tuple))
            ## replace string references with the static data unique values
            for i,ep in enumerate(eval_params[a]):
                if isinstance(ep, str):
                    eval_params[a][i] = np.unique(
                            static_data[static_labels.index(ep)])

    ## eval_{subdomain}_{model}_{eval-type}_{data_primary}_{data_secondary}
    name_fields = ("eval", subdomain, model_config["name"],
            eval_class, eval_d1, eval_d2)
    ## declare and return the Evaluator based on its configured type.
    return evaluators.EVALUATORS[eval_class](
            params=eval_params,
            feats=dataset_feats,
            meta={"model_config":model_config, "name":"_".join(name_fields)},
            )

if __name__=="__main__":
    proj_root = Path("/rhome/mdodson/emulate-era5-land")
    pkl_dir = proj_root.joinpath("data/eval-new")
    model_parent_dir = proj_root.joinpath("data/models")
    ## static data for reference building coordinate arrays for EvalStatic
    slabels,sdata = pkl.load(proj_root.joinpath(
        "data/static/era5_static.pkl"
        ).open("rb"))

    model_name = "acclstm-era5-swm-9_state_0069.pwf"
    #model_name = "acclstm-era5-swm-50_state_0120.pwf"
    #model_name = "acclstm-era5-swm-64_state_0024.pwf"

    debug = True
    batch_size = 1024
    prefetch_factor = 6
    #num_batches = 2048
    #num_batches = 4096
    num_batches = 4096
    save_every_nbatches = 32
    horizon_hours = 24*5
    nworkers_dataset = 6
    nworkers_eval = 6

    subdomain = "full" ## indicate source data (full test dataset)

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
    df_err_abs = [[("err-abs", f), ("err-abs", f"diff {f}")] for f in f_intg]
    df_err_bias = [[("err-bias",f), ("err-bias",f"diff {f}")] for f in f_intg]
    df_err_all = df_err_abs + df_err_bias

    df_intg_all = df_intg_true+df_intg_pred+df_intg_err_bias+df_intg_err_abs
    df_diff_all = df_diff_true+df_diff_pred+df_diff_err_bias+df_diff_err_abs

    ## eval_{subdomain}_{model}_{eval-type}_{data_primary}_{data_secondary}
    ET,EJH,ES,ESS = ["EvalTemporal","EvalJointHist","EvalStatic",
            "EvalSampleSources"]
    eval_config = [
        (ESS,"space-time", "none", []), ## try covariate feats

        (ET,"doy-tod", "intg-all", [df_intg_all]),
        (ET, "doy-tod", "diff-all", [df_diff_all]),

        (EJH, "diff-swm-7", "counts", [[], []]),
        (EJH, "diff-swm-28", "counts", [[], []]),
        (EJH, "swm-7", "counts", [[], []]),
        (EJH, "swm-28", "counts", [[], []]),
        #(EJH, "swm-100", "counts", [[], []]),
        #(EJH, "swm-289", "counts", [[], []]),

        (EJH, "state-diff-swm-7", "err-diff",
            [[df_diff_err_bias[0],df_diff_err_abs[0]], []]),
        (EJH, "state-diff-swm-28", "err-diff",
            [[df_diff_err_bias[1],df_diff_err_abs[1]], []]),

        (EJH, "tmp-dwpt", "err-diff", [[
            df_diff_err_bias[0],df_diff_err_abs[0],
            df_intg_err_bias[0],df_intg_err_abs[0]], []]),
        (EJH, "tmp-snow", "err-diff",
            [[df_diff_err_bias[0],df_diff_err_abs[0]], []]),
        (EJH, "trsp-evp", "err-diff",
            [[df_diff_err_bias[0],df_diff_err_abs[0]], []]),

        (ES,"grid","err-mean",
            [df_intg_err_bias+df_intg_err_abs,"mean",True,False]),
        (ES,"grid","err-max", [df_intg_err_bias,"max",False,True]),
        (ES,"grid", "err-min", [df_intg_err_bias,"min",False,True]),

        (ES,"veg-soil-combos","err-mean",
            [df_intg_err_bias+df_intg_err_abs,"mean",True,False]),
        ]

    ## ----------------------------------------------------------------- ##

    aux_dynamic_addons = ["evp-trsp", "evp"]
    aux_static_addons = ["vt-high", "vt-low", "soilt", "vidxs", "hidxs"]

    ## declare a prediction dataset based on the already-trained model
    model_dir = model_parent_dir.joinpath(model_name.split("_")[0])
    md = ModelDir(model_dir)
    aux_dynamic_feats = md.config["feats"]["aux_dynamic_feats"] + \
        list(filter(lambda f:f not in md.config["feats"]["aux_dynamic_feats"],
                aux_dynamic_addons))
    aux_static_feats = md.config["feats"]["aux_static_feats"] + \
        list(filter(lambda f:f not in md.config["feats"]["aux_static_feats"],
                aux_static_addons))
    pds = PredictionDataset(
        model_path=model_dir.joinpath(model_name),
        use_dataset="eval",
        normalized_outputs=False,
        config_override={
            "feats":{
                #"horizon_size":336,
                "horizon_size":horizon_hours,
                "aux_dynamic_feats":aux_dynamic_feats,
                "aux_static_feats":aux_static_feats,
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
            "auxd-h":aux_dynamic_feats,
            "auxd-w":aux_dynamic_feats,
            "auxs":aux_static_feats,
            "time":["epoch"],
            "target":md.config["feats"]["target_feats"]+integ_feats,
            "pred":md.config["feats"]["target_feats"]+integ_feats,
            "err-bias":md.config["feats"]["target_feats"]+integ_feats,
            "err-abs":md.config["feats"]["target_feats"]+integ_feats,
            }

    ## declare the Evaluator subclass objects
    evals = [
        get_eval_from_config(
            subdomain, md.config, dataset_feats, ctup, slabels, sdata)
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
        hslice = slice(-horizon_hours,None)

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

    ## If the last batch didn't save...
    if not (i+1)%save_every_nbatches == 0:
        ## Final evaluator save.
        for en,ev in zip(ev_names,evals):
            pkl_path = pkl_dir.joinpath(en+".pkl")
            pkl.dump(ev, pkl_path.open("wb"))
            print(f"Wrote to {pkl_path.name}")
