# coding: utf-8
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool
import cartopy.crs as ccrs
from pprint import pprint

from era5_testbed.helpers import get_permutation

config_labels = ["target_avg_dist", "roll_threshold", "threshold_diminish",
        "recycle_count", "seed", "dynamic_roll_threshold"]

configs = [
        (2, .5, .01, 10, 200007221750, False),
        (2, .5, .01, 50, 200007221750, False),
        (2, .5, .01, 3, 200007221750, False),

        (2, .4, .01, 10, 200007221750, False),
        (2, .4, .01, 50, 200007221750, False),
        (2, .4, .01, 3, 200007221750, False),

        (2, .3, .01, 10, 200007221750, False),
        (2, .3, .01, 50, 200007221750, False),
        (2, .3, .01, 3, 200007221750, False),

        (2, .3, .005, 10, 200007221750, False),
        (2, .3, .005, 50, 200007221750, False),
        (2, .3, .005, 3, 200007221750, False),

        (2, .3, .002, 10, 200007221750, False),
        (2, .3, .002, 50, 200007221750, False),
        (2, .3, .002, 3, 200007221750, False),
        ]

def plot_geo_rgb(rgb:np.ndarray, lat_range:tuple, lon_range:tuple,
        plot_spec:dict={}, fig_path=None, show=False):
    """
    """
    ps = {"title":"", "figsize":(16,12), "border_linewidth":2,
            "title_size":12 }
    ps.update(plot_spec)
    fig = plt.figure(figsize=ps.get("figsize"))

    pc = ccrs.PlateCarree()

    ax = fig.add_subplot(1, 1, 1, projection=pc)
    extent = [*lon_range, *lat_range]
    ax.set_extent(extent, crs=pc)

    ax.imshow(rgb, extent=extent, transform=pc)
    print(extent)

    ax.coastlines(
            color=ps.get("border_color", "black"),
            linewidth=ps.get("border_linewidth"))
    ax.add_feature(
            ccrs.cartopy.feature.STATES,
            #color=ps.get("border_color", "black"),
            linewidth=ps.get("border_linewidth")
            )

    plt.title(ps.get("title"), fontweight='bold',
            fontsize=ps.get("title_size"))

    if not fig_path is None:
        fig.savefig(fig_path.as_posix(), bbox_inches="tight", dpi=80)
    if show:
        plt.show()
    plt.close()
    return

def mp_get_permutation(args):
    return args,get_permutation(**args)

if __name__=="__main__":
    proj_root_dir = Path("/rhome/mdodson/emulate-era5-land")
    pkl_dir = proj_root_dir.joinpath("data/permutations")
    fig_dir = proj_root_dir.joinpath("figures/permutations")
    slabels,sdata = pkl.load(Path("data/static/era5_static.pkl").open("rb"))
    m_valid = sdata[slabels.index("m_valid")].astype(bool)
    workers = 12
    enum_start = 60

    latlon = np.stack([
        sdata[slabels.index("lat")][m_valid],
        sdata[slabels.index("lon")][m_valid],
        ], axis=1)

    ## Generate a bunch of permutations given different permutations
    '''
    init_perm = np.arange(latlon.shape[0])
    rng = np.random.default_rng(seed=200007221750)
    rng.shuffle(init_perm)

    default_args = {
            "coord_array":latlon,
            "initial_perm":init_perm,
            "max_iterations":150,
            #"max_iterations":3,
            "return_stats":True,
            "debug":True,
            }

    args = [{**dict(zip(config_labels,c)),**default_args} for c in configs]
    with Pool(workers) as pool:
        for i,(a,r) in enumerate(
                pool.imap_unordered(mp_get_permutation,args), enum_start):
            perm,stats = r
            r_perm = np.asarray(tuple(zip(*sorted(zip(
                list(perm), range(len(perm))
                ), key=lambda v:v[0])))[1])
            pkl.dump((a, np.stack([perm,r_perm], axis=-1), stats),
                    pkl_dir.joinpath(f"permutation_{i:03}.pkl").open("wb"))
    '''

    ## Check how many chunks it would take to extract certain subgrids
    #'''
    substrs = [
            #"026", ##  669  852 1042  806  |  9.03  4.26
            #"046", ##  528  661  800  572  |  6.75  3.09
            #"047", ##  453  575  685  489  |  5.92  3.43
            #"012", ##  322  417  473  377  |  4.29  3.19
            #"005", ##  310  415  453  372  |  4.16  3.28
            #"019", ##  311  415  462  368  |  4.15  3.27
            "064",
            "066",
            "067",
            "068",
            #"",
            ]
    test_subgrids = {
            "seus":((33.88, 36.86), (-88.1, -83.6)),
            "michigan":((41.59, 45.94), (-88.28, -81.9)),
            "colorado":((36.99, 40.96), (-109.1, -102.05)),
            "etx":((28.97, 33.39), (-98.67, -93.28)),
            "ne":((43.0, 47.23), (-79.34, -67.99)),
            "nw":((42.00, 48.96), (-122.74, -118.24)),
            "cplains":((39.32, 41.69), (-96.88, -92.78)),
            "hplains":((43.73, 48.33), (-104.01, -96.08)),
            }
    chunk_size = 64
    print_pkls = [p for p in pkl_dir.iterdir()
            if any(s in p.name for s in substrs)]
    print(np.amin(latlon, axis=0), np.amax(latlon, axis=0))
    for pf in print_pkls:
        args,perm,stats = pkl.load(pf.open("rb"))
        ## (N,2) latlon array after permutation
        ll_perm = latlon[perm[...,0]]
        ## (N,) latlon distances of permuted points from original location
        dists = np.sum((latlon-ll_perm)**2, axis=1)**(1/2)
        _,pname = pf.stem.split("_")

        print(pf.stem, stats[-1])
        print(np.amin(dists),np.amax(dists),np.average(dists),np.std(dists))

        for rlabel,((lat0,latf),(lon0,lonf)) in test_subgrids.items():
            ## 1d boolean mask of permuted points within the subgrid
            in_subset = (ll_perm[:,0] >= lat0) & (ll_perm[:,0] <= latf) & \
                    (ll_perm[:,1] >= lon0) & (ll_perm[:,1] <= lonf)
            ## Number of unique contiguous chunks associated with the subset
            unq_chunks = np.unique(np.argwhere(in_subset)[:,0] // chunk_size)
            print(f"{rlabel} {unq_chunks.size = }")

            '''
            ## (N,2) indeces of valid points in unpermuted space
            valid_idxs = np.argwhere(m_valid)
            ## (N,) indeces of subset points in permuted space
            subset_idxs = np.argwhere(in_subset).T

            ## Color unpermuted valid points white on a (lat,lon,3) rgb
            rgb = np.full(m_valid.shape, 0)
            rgb[valid_idxs[:,0], valid_idxs[:,1]] = 255
            rgb = np.stack([rgb for i in range(3)], axis=-1)
            ## color unpermuted subset pixels red and permuted blue
            sub_ixs_noperm = valid_idxs[perm[:,0]][in_subset]
            sub_ixs_perm = valid_idxs[in_subset]
            rgb[sub_ixs_noperm[:,0],sub_ixs_noperm[:,1]] = np.array([0,0,255])
            rgb[sub_ixs_perm[:,0],sub_ixs_perm[:,1]] = np.array([255,0,0])

            plot_geo_rgb(
                    rgb=rgb,
                    lat_range=(np.amin(latlon[:,0]),np.amax(latlon[:,0])),
                    lon_range=(np.amin(latlon[:,1]),np.amax(latlon[:,1])),
                    plot_spec={
                        "border_color":"black",
                        "border_linewidth":1,
                        "title":f"permutation {pname} ({rlabel})"
                        },
                    fig_path=fig_dir.joinpath(f"perm_{pname}_{rlabel}.png"),
                    show=False
                    )
            '''
    #'''
