
#import torch
import numpy as np
from pathlib import Path
from datetime import datetime,timedelta
import json
import pickle as pkl

from emulate_era5_land.generators import worker_init_fn
from emulate_era5_land.generators import SparseTimegridSampleDataset

if __name__=="__main__":
    proj_root = Path("/rhome/mdodson/emulate-era5-land")
    data_dir = proj_root.joinpath("data/timegrids")
    info_era5 = json.load(proj_root.joinpath(
                "data/list_feats_era5.json").open("r"))
    ds_train = SparseTimegridSampleDataset(
            timegrids=[tg for tg in data_dir.iterdir()
                if int(tg.stem.split("_")[-1]) in range(2012,2018)],
            window_feats=[
                "vsm-07", "vsm-28", "vsm-100", "vsm-289"
                "lai-low", "lai-high", "alb", "weasd", "pres", "windmag",
                "tmp", "dwpt", "apcp", "dlwrf", "dswrf", "alb"
                ],
            horizon_feats=[
                "lai-low", "lai-high", "alb", "weasd", "pres", "windmag",
                "tmp", "dwpt", "apcp", "dlwrf", "dswrf", "alb"
                ],
            target_feats=[
                "diff vsm-07", "diff vsm-28", "diff vsm-100", "diff vsm-289"
                ],
            static_int_feats=["soilt", "vt-low", "vt-high"],
            derived_feats={
                "windmag":(("ugrd", "vgrd"), tuple(),
                    "lambda d,s:(d[0]**2+d[1]**2)**(1/2)")
                },
            static_embed_maps={
                "soilt":[0, 1, 2, 3, 4, 5, 6, 7],
                "vt-low":[ 0,  1,  2,  7,  9, 10, 11, 13, 16, 17],
                "vt-high":[ 0,  3,  5,  6, 18, 19],
                },
            window_size=24,
            horizon_size=72,
            dynamic_norm_coeffs={},
            static_norm_coeffs={},
            shuffle=True,
            seed=200007221750,
            sample_across_files=True,
            sample_cutoff=.7,
            sample_under_cutoff=True,
            sample_separation=67,
            random_offset=True,
            chunk_pool_count=7,
            buf_size_mb=1024,
            buf_slots=128,
            buf_policy=0,
            )
    gen = torch.utils.data.DataLoader(
            dataset=ds_train,
            batch_size=32,
            num_workers=5,
            prefetch_factor=3,
            worker_init_fn=worker_init_fn,
            )

    for bix,sample in enumerate(gen):
        w,h,y,s,si,t = sample
        print(f"Gathered sample: {[v.shape for v in (w,h,y,s,t)]}")
        print(f"si: {[v.shape for v in si]}")
