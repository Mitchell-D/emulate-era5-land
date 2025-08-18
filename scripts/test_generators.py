
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
    ds_train = SparseTimegridSampleDataset(
            timegrids=[tg for tg in data_dir.iterdir()
                if int(tg.stem.split("_")[-1]) in range(2012,2018)],
            shuffle=True,
            seed=200007221750,
            sample_cutoff=.7,
            sample_separation=67,
            shuffle_offset=True,
            buf_size_mb=1024,
            buf_slots=128,
            buf_policy=0,
            )
    gen = torch.utils.data.DataLoader(
            dataset=ds_train,
            batch_size=32,
            num_workers=5,
            worker_init_fn=worker_init_fn,
            )

    for bix,sample in enumerate(gen):
