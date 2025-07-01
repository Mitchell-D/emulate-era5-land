# coding: utf-8
import numpy as np
import pickle as pkl
from pathlib import Path
from multiprocessing import Pool

from era5_testbed.helpers import get_permutation

config_labels = ["target_avg_dist", "roll_threshold", "threshold_diminish",
        "recycle_count", "seed", "dynamic_roll_threshold"]

configs = [
        (3, .33, .01, 10, 200007221750, False),
        (3, .2, .01, 10, 200007221750, False),
        (3, .5, .01, 10, 200007221750, False),
        (3, .7, .01, 10, 200007221750, False),
        (3, .85, .01, 10, 200007221750, False),
        (3, .33, .005, 10, 200007221750, False),
        (3, .2, .005, 10, 200007221750, False),
        (3, .5, .005, 10, 200007221750, False),
        (3, .7, .005, 10, 200007221750, False),
        (3, .85, .005, 10, 200007221750, False),

        (3, .33, .01, 50, 200007221750, False),
        (3, .2, .01, 50, 200007221750, False),
        (3, .5, .01, 50, 200007221750, False),
        (3, .7, .01, 50, 200007221750, False),
        (3, .85, .01, 50, 200007221750, False),
        (3, .33, .005, 50, 200007221750, False),
        (3, .2, .005, 50, 200007221750, False),
        (3, .5, .005, 50, 200007221750, False),
        (3, .7, .005, 50, 200007221750, False),
        (3, .85, .005, 50, 200007221750, False),

        (3, .33, .01, 3, 200007221750, False),
        (3, .2, .01, 3, 200007221750, True),
        (3, .5, .01, 3, 200007221750, False),
        (3, .7, .01, 3, 200007221750, True),
        (3, .85, .01, 3, 200007221750, False),
        (3, .33, .005, 3, 200007221750, True),
        (3, .2, .005, 3, 200007221750, False),
        (3, .5, .005, 3, 200007221750, True),
        (3, .7, .005, 3, 200007221750, False),
        (3, .85, .005, 3, 200007221750, True),

        (6, .33, .01, 10, 200007221750, False),
        (6, .2, .01, 10, 200007221750, False),
        (6, .5, .01, 10, 200007221750, False),
        (6, .7, .01, 10, 200007221750, False),
        (6, .85, .01, 10, 200007221750, False),
        (6, .33, .005, 10, 200007221750, False),
        (6, .2, .005, 10, 200007221750, False),
        (6, .5, .005, 10, 200007221750, False),
        (6, .7, .005, 10, 200007221750, False),
        (6, .85, .005, 10, 200007221750, False),

        (6, .33, .01, 50, 200007221750, False),
        (6, .2, .01, 50, 200007221750, False),
        (6, .5, .01, 50, 200007221750, False),
        (6, .7, .01, 50, 200007221750, False),
        (6, .85, .01, 50, 200007221750, False),
        (6, .33, .005, 50, 200007221750, False),
        (6, .2, .005, 50, 200007221750, False),
        (6, .5, .005, 50, 200007221750, False),
        (6, .7, .005, 50, 200007221750, False),
        (6, .85, .005, 50, 200007221750, False),

        (6, .33, .01, 3, 200007221750, False),
        (6, .2, .01, 3, 200007221750, True),
        (6, .5, .01, 3, 200007221750, False),
        (6, .7, .01, 3, 200007221750, True),
        (6, .85, .01, 3, 200007221750, False),
        (6, .33, .005, 3, 200007221750, True),
        (6, .2, .005, 3, 200007221750, False),
        (6, .5, .005, 3, 200007221750, True),
        (6, .7, .005, 3, 200007221750, False),
        (6, .85, .005, 3, 200007221750, True),
        ]

def mp_get_permutation(args):
    return args,get_permutation(**args)

if __name__=="__main__":
    proj_root_dir = Path("/rhome/mdodson/emulate-era5-land")
    pkl_dir = proj_root_dir.joinpath("data/permutations")

    workers = 12

    slabels,sdata = pkl.load(Path("data/static/era5_static.pkl").open("rb"))
    m_valid = sdata[slabels.index("m_valid")].astype(bool)
    latlon = np.stack([
        sdata[slabels.index("lat")][m_valid],
        sdata[slabels.index("lon")][m_valid],
        ], axis=1)
    init_perm = np.arange(latlon.shape[0])
    rng = np.random.default_rng(seed=200007221750)
    rng.shuffle(init_perm)

    default_args = {
            "coord_array":latlon,
            "initial_perm":init_perm,
            "max_iterations":50,
            #"max_iterations":3,
            "return_stats":True,
            "debug":True,
            }

    args = [{**dict(zip(config_labels,c)),**default_args} for c in configs]
    with Pool(workers) as pool:
        for i,(a,r) in enumerate(pool.imap_unordered(mp_get_permutation,args)):
            perm,stats = r
            r_perm = np.asarray(tuple(zip(*sorted(zip(
                list(perm), range(len(perm))
                ), key=lambda v:v[0])))[1])
            pkl.dump((a, np.stack([perm,r_perm], axis=-1), stats),
                    pkl_dir.joinpath(f"permutation_{i:03}.pkl").open("wb"))
