"""
Script for dispatching single tracktrain-based training runs at a time,
each based on the configuration dict below. Each run will create a new
ModelDir directory and populate it with model info, the configuration,
and intermittent models saved duing training.
"""
from pathlib import Path
from emulate_era5_land.training import train_single
import json

import torch

## config template

proj_root = Path("/rhome/mdodson/emulate-era5-land/")
info_era5 = json.load(proj_root.joinpath(
    "data/list_feats_era5.json").open("r"))

timegrids = [
        proj_root.joinpath(
            f"data/timegrids/timegrid_era5_{y}.h5"
            ).as_posix()
        for y in range(2012,2019)
        ]
config = {
    "feats":{
        "window_feats":[
            "pres","tmp","dwpt","apcp","alb",
            "dlwrf","dswrf","wm-snow","ugrd","vgrd",
            "lai-low", "lai-high",
            "swm-7", "swm-28", "swm-100", "swm-289",
            ],
        "horizon_feats":[
            "pres","tmp","dwpt","apcp","alb",
            "dlwrf","dswrf","wm-snow","ugrd","vgrd",
            "lai-low", "lai-high",
            ],
        "target_feats":[
            "diff swm-7","diff swm-28","diff swm-100","diff swm-289",
            #"swm-7", "swm-28", "swm-100", "swm-289",
            ],
        "static_feats":["geopot","lakec","vc-high","vc-low",],
        "static_int_feats":["soilt","vt-high","vt-low"],
        "aux_dynamic_feats":["evp","lhtfl"],
        "aux_static_feats":["vidxs","hidxs"],
        "derived_feats":info_era5["derived-feats"],
        "static_embed_maps":{
            "soilt":[0, 1, 2, 3, 4, 5, 6, 7],
            "vt-low":[0,  1,  2,  7,  9, 10, 11, 13, 16, 17],
            "vt-high":[0,  3,  5,  6, 18, 19],
            },
        "window_size":48,
        "horizon_size":96,
        "norm_coeffs":info_era5["norm-coeffs"],
        },
    "data":{
        "train":{
            "timegrids":timegrids,
            "shuffle":True,
            "sample_cutoff":.66,
            "sample_across_files":True,
            "sample_under_cutoff":True,
            "sample_separation":409,
            "random_offset":True,
            "chunk_pool_count":48,
            "buf_size_mb":4096,
            "buf_slots":48,
            "buf_policy":0,
            "batch_size":32,
            "num_workers":6,
            "prefetch_factor":6,
            "out_dtype":"f4",
            },
        "val":{
            "timegrids":timegrids,
            "shuffle":True,
            "sample_cutoff":.66,
            "sample_across_files":True,
            "sample_under_cutoff":False,
            "sample_separation":409,
            "random_offset":True,
            "chunk_pool_count":48,
            "buf_size_mb":4096,
            "buf_slots":48,
            "buf_policy":0,
            "batch_size":32,
            "num_workers":6,
            "prefetch_factor":6,
            "out_dtype":"f4",
            },
        },
    "metrics":{
            "mse":{"reduction":"mean"},
            "mae":{"reduction":"mean"},
            "fwmae":{"feature_weights":[8,4,2,1]},
            },
    "model":{
        "type":"acclstm",
        "args":{
            "static_int_encoding_size":6,
            "num_hidden_feats":128,
            "num_hidden_layers":6,
            "lstm_kwargs":{},
            "normalized_inputs":True,
            "normalized_outputs":True,
            "cycle_targets":[
                "diff swm-7","diff swm-28","diff swm-100","diff swm-289",
                #"swm-7","swm-28","swm-100","swm-289",
                ],
            "teacher_forcing":True,
            },
        },
    "setup":{
        "loss_metric":"fwmae",
        #"loss_metric":"mae",
        "optimizer_type":"nadam",
        "optimizer_args":{},
        "schedule_type":"cyclic",
        "schedule_args":{
            "base_lr":1e-4,
            "max_lr":1e-2,
            "step_size_up":4,
            "step_size_down":16,
            "mode":"triangular2",
            "gamma":0.9,
            },
        "initial_lr":1e-2,
        "early_stop_patience":12.,
        "early_stop_delta":0.,
        "max_epochs":2048,
        "batches_per_epoch":256,
        "val_frequency":1,
        },
    "seed":200007221750,
    "name":"acclstm-era5-swm-19",
    "notes":"same as v18 but using weighted loss function with sfc heavier",
    }

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_single(
        config=config,
        model_parent_dir=proj_root.joinpath("data/models"),
        debug=True,
        device=device,
        )

