# coding: utf-8
import numpy as np
import pickle as pkl
from pathlib import Path

from era5_testbed.helpers import get_permutation

rng = np.random.default_rng(seed=200007221750)

slabels,sdata = pkl.load(Path("data/static/era5_static.pkl").open("rb"))
latlon = np.stack([
    sdata[slabels.index("lat")][sdata[slabels.index("m_valid")].astype(bool)],
    sdata[slabels.index("lon")][sdata[slabels.index("m_valid")].astype(bool)],
    ], axis=1)
init_perm = np.arange(latlon.shape[0])
rng.shuffle(init_perm)

perm = get_permutation(
    coord_array=latlon,
    initial_perm=init_perm,
    target_avg_dist=6,
    roll_threshold=.33,
    threshold_diminish=.01,
    recycle_count=10,
    seed=200007221750,
    dynamic_roll_threshold=False,
    debug=True,
    )

_,r_perm = zip(*sorted(zip(perm, range(perm.size)), key=lambda v:v[0]))

print(perm)
print(r_perm)

print()
print((np.arange(perm.size) - 17)[perm][r_perm])
print((np.arange(perm.size) - 17)[r_perm][perm])
