import numpy as np
import json
import torch
from pathlib import Path
from time import perf_counter

from emulate_era5_land.generators import PredictionDataset

if __name__=="__main__":
    proj_root = Path("/rhome/mdodson/emulate-era5-land")
    model_parent_dir = proj_root.joinpath("data/models")
    model_dir = model_parent_dir.joinpath("acclstm-era5-swm-9")
    model_name = "acclstm-era5-swm-9_state_0069.pwf"

    eval_tgs = [
            proj_root.joinpath(f"data/timegrids/timegrid_era5_{year}.h5")
            for year in range(2018,2024)
            ]
    assert all([tg.exists() for tg in eval_tgs])

    pds = PredictionDataset(
            model_path=model_dir.joinpath(model_name),
            use_dataset="eval",
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
                        "batch_size":32,
                        "num_workers":8,
                        "prefetch_factor":6,
                        "out_dtype":"f4",
                        },
                    },
                },
            )

    for i in range(32):
        x,y,a,p = next(pds)
        print(p[0].shape)
