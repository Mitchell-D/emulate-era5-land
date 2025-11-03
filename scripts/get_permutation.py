# coding: utf-8
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool
import cartopy.crs as ccrs
from pprint import pprint

from emulate_era5_land.helpers import mp_get_permutation
from emulate_era5_land.helpers import mp_get_permutation_conv

config_labels = ["target_avg_dist", "roll_threshold", "threshold_diminish",
        "recycle_count", "seed", "dynamic_roll_threshold"]

configs = [
        (2, .5, .01, 10, 20007221750, False),
        (2, .5, .01, 50, 20000221750, False),
        (2, .4, .01, 3, 20000721750, False),

        (3, .5, .01, 10, 20007221750, False),
        (3, .5, .01, 50, 20000721750, False),
        (3, .4, .01, 3, 20000721750, False),

        (2, .5, .01, 10, 2007221750, False),
        (2, .4, .01, 50, 2000072750, False),
        (2, .3, .01, 3, 2000721750, False),

        (4, .5, .01, 10, 200722150, False),
        (4, .4, .01, 50, 200721750, False),
        (4, .4, .01, 3, 7221750, False),

        (6, .5, .01, 10, 20072150, False),
        (6, .3, .01, 50, 20021750, False),
        (6, .3, .01, 3, 2000225, False),
        ]

config_labels_conv = ["dist_threshold", "reperm_cap", "shuffle_frac", "seed"]

configs_conv = [
        (.5, 4, .1, 57927859614),
        (.5, 8, .1, 57927859614),
        (.5, 16, .1, 57927859614),
        (.5, 32, .1, 57927859614),
        (.5, 4, .1, 57927859614),
        (.5, 8, .1, 57927859614),
        (.5, 16, .1, 57927859614),
        (.5, 32, .1, 57927859614),
        (.5, 4, .25, 57927859614),
        (.5, 16, .25, 57927859614),
        (.5, 32, .25, 57927859614),
        (.5, 4, .5, 57927859614),
        (.5, 16, .5, 57927859614),
        (.5, 32, .5, 57927859614),

        (1, 4, .1, 57927859614),
        (1, 8, .1, 57927859614),
        (1, 16, .1, 57927859614),
        (1, 32, .1, 57927859614),
        (1, 4, .1, 57927859614),
        (1, 8, .1, 57927859614),
        (1, 16, .1, 57927859614),
        (1, 32, .1, 57927859614),
        (1, 4, .25, 57927859614),
        (1, 16, .25, 57927859614),
        (1, 32, .25, 57927859614),
        (1, 4, .5, 57927859614),
        (1, 16, .5, 57927859614),
        (1, 32, .5, 57927859614),

        (1.5, 4, .1, 57927859614),
        (1.5, 8, .1, 57927859614),
        (1.5, 16, .1, 57927859614),
        (1.5, 32, .1, 57927859614),
        (1.5, 4, .1, 57927859614),
        (1.5, 8, .1, 57927859614),
        (1.5, 16, .1, 57927859614),
        (1.5, 32, .1, 57927859614),
        (1.5, 4, .25, 57927859614),
        (1.5, 16, .25, 57927859614),
        (1.5, 32, .25, 57927859614),
        (1.5, 4, .5, 57927859614),
        (1.5, 16, .5, 57927859614),
        (1.5, 32, .5, 57927859614),
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

def plot_multiy_lines(data, xaxis, plot_spec={},
        show=False, fig_path=None):
    """
    """
    ps = {"fig_size":(12,6), "dpi":80, "spine_increment":.01,
            "date_format":"%Y-%m-%d", "xtick_rotation":30}
    ps.update(plot_spec)
    if len(xaxis) != len(data[0]):
        raise ValueError(
                "Length of 'xaxis' must match length of each dataset.")

    fig,host = plt.subplots(figsize=ps.get("fig_size"))
    fig.subplots_adjust(left=0.2 + ps.get("spine_increment") \
            * (len(data) - 1))

    axes = [host]
    colors = ps.get("colors", ["C" + str(i) for i in range(len(data))])
    y_labels = ps.get("y_labels", [""] * len(data))
    y_ranges = ps.get("y_ranges", [None] * len(data))

    ## Create additional y-axes on the left, offset horizontally
    for i in range(1, len(data)):
        ax = host.twinx()
        #ax.spines["left"] = ax.spines["right"]
        ax.yaxis.set_label_position("left")
        ax.yaxis.set_ticks_position("left")
        ax.spines["left"].set_position(
                ("axes", -1*ps.get("spine_increment") * i))
        axes.append(ax)

    ## Plot each series
    for i, (ax, series) in enumerate(zip(axes, data)):
        ax.plot(xaxis, series, color=colors[i], label=y_labels[i])
        ax.set_ylabel(y_labels[i], color=colors[i],
                fontsize=ps.get("label_size"))
        ax.tick_params(axis="y", colors=colors[i])
        if y_ranges[i] is not None:
            ax.set_ylim(y_ranges[i])

    host.set_xlabel(ps.get("x_label", "Time"), fontsize=ps.get("label_size"))
    host.tick_params(axis="x", rotation=ps.get("xtick_rotation"))

    if plot_spec.get("xtick_align"):
        plt.setp(host.get_xticklabels(),
                horizontalalignment=plot_spec.get("xtick_align"))

    if ps.get("zero_axis"):
        host.axhline(0, color="black")

    plt.title(ps.get("title", ""), fontdict={"fontsize":ps.get("title_size")})
    plt.tight_layout()
    if show:
        plt.show()
    if not fig_path is None:
        fig.savefig(fig_path, bbox_inches="tight", dpi=plot_spec.get("dpi"))
    plt.close()

def plot_perm_pkl(perm_pkl:Path, coords:np.array, subgrids:dict,
        valid_mask:np.array, fig_dir:Path, chunk_size=64, plot_stats=False,
        plot_sparse_chunks=True, num_sparse_chunks=50, seed=None, debug=False):
    """
    :@param perm_pkl:
    :@param coords:
    :@param subgrids: dict mapping labels to ((y0,yf),(x0,xf)) coord bounds
    :@param fig_dir:
    :@param valid_mask: 2d boolean array with len(coords.shape[0]) True values
        corresponding to the locations of valid points
    :@param chunk_size: Hypothetical 1d chunk size used efficiency estimates
    :@param plot_stats:
    :@param seed: seed for choosing
    :@param debug:
    """
    args,perm,stats = pkl.load(perm_pkl.open("rb"))

    ## verify that the permutation includes all points
    assert perm[:,0].size==coords[:,0].size, \
            f"Perm size mismatch: {perm[:,0].size = }  {coords[:,0].size = }"
    check_valid = np.full(coords.shape[0],False)
    check_valid[perm[:,0]] = True
    assert np.all(check_valid), f"{perm_pkl.name} not a valid permutation!"

    if debug:
        pprint({k:v for k,v in args.items() if k!="coord_array"})

    ## (N,2) latlon array after permutation
    ll_perm = coords[perm[...,0]]
    ## (N,) latlon distances of permuted points from original location
    dists = np.sum((coords-ll_perm)**2, axis=1)**(1/2)
    ## (N,2) indeces of valid points in unpermuted space
    valid_idxs = np.argwhere(valid_mask)

    px_per_chunk = []
    for rlabel,((lat0,latf),(lon0,lonf)) in subgrids.items():
        ## 1d boolean mask of permuted points within the subgrid
        in_subset = (ll_perm[:,0] >= lat0) & (ll_perm[:,0] <= latf) & \
                (ll_perm[:,1] >= lon0) & (ll_perm[:,1] <= lonf)
        ## Number of unique contiguous chunks associated with the subset
        unq_chunks = np.unique(np.argwhere(in_subset)[:,0] // chunk_size)
        #if debug:
        #    print(f"{rlabel} {unq_chunks.size = }")
        px_per_chunk.append(np.count_nonzero(in_subset) / unq_chunks.size)

        ## (N,) indeces of subset points in permuted space
        subset_idxs = np.argwhere(in_subset).T

        ## Color unpermuted valid points white on a (lat,lon,3) rgb
        rgb = np.full(valid_mask.shape, 0)
        rgb[valid_idxs[:,0], valid_idxs[:,1]] = 255
        rgb = np.stack([rgb for i in range(3)], axis=-1)
        ## color unpermuted subset pixels red and permuted blue
        sub_ixs_noperm = valid_idxs[perm[:,0]][in_subset]
        sub_ixs_perm = valid_idxs[in_subset]
        rgb[sub_ixs_noperm[:,0],sub_ixs_noperm[:,1]] = np.array([0,0,255])
        rgb[sub_ixs_perm[:,0],sub_ixs_perm[:,1]] = np.array([255,0,0])

        plot_geo_rgb(
                rgb=rgb,
                lat_range=(np.amin(coords[:,0]),np.amax(coords[:,0])),
                lon_range=(np.amin(coords[:,1]),np.amax(coords[:,1])),
                plot_spec={
                    "border_color":"black",
                    "border_linewidth":1,
                    "title":f"{perm_pkl.stem} ({rlabel})"
                    },
                fig_path=fig_dir.joinpath(f"{perm_pkl.stem}_{rlabel}.png"),
                show=False
                )

    if plot_sparse_chunks:
        rng = np.random.default_rng(seed)
        chunk_starts = np.arange(ll_perm.shape[0] // chunk_size) * chunk_size
        rng.shuffle(chunk_starts)
        ext_chunks = chunk_starts[:num_sparse_chunks]
        in_subset = np.full(ll_perm.shape[0], False)
        for cs in ext_chunks:
            in_subset[cs:cs+chunk_size] = True
        rgb = np.full(valid_mask.shape, 0)
        rgb[valid_idxs[:,0], valid_idxs[:,1]] = 255
        rgb = np.stack([rgb for i in range(3)], axis=-1)
        sub_ixs_noperm = valid_idxs[perm[:,0]][in_subset]
        sub_ixs_perm = valid_idxs[in_subset]
        rgb[sub_ixs_noperm[:,0],sub_ixs_noperm[:,1]] = np.array([255,0,0])
        rgb[sub_ixs_perm[:,0],sub_ixs_perm[:,1]] = np.array([0,0,255])
        plot_geo_rgb(
                rgb=rgb,
                lat_range=(np.amin(coords[:,0]),np.amax(coords[:,0])),
                lon_range=(np.amin(coords[:,1]),np.amax(coords[:,1])),
                plot_spec={
                    "border_color":"black",
                    "border_linewidth":1,
                    "title":f"{perm_pkl.stem} " + \
                            f"({num_sparse_chunks} x {chunk_size})"
                    },
                fig_path=fig_dir.joinpath(f"{perm_pkl.stem}_chunked.png"),
                show=False
                )

    mean_px_per_chunk = np.average(px_per_chunk)
    print(f"{perm_pkl.stem}, {mean_px_per_chunk = :.4f}")

    if plot_stats:
        plot_multiy_lines(
                data=list(zip(*stats)),
                xaxis=np.arange(len(stats)),
                plot_spec={
                    "x_label":"Iterations",
                    "y_labels":["Avg. Dist", "Stdev. Dist"],
                    "y_ranges":[(0,15),(0,15)],
                    "title":f"{perm_pkl.stem} {mean_px_per_chunk=:.3f}",
                    "spine_increment":.1,
                    },
                show=False,
                fig_path=fig_dir.joinpath(f"{perm_pkl.stem}_stats.png")
                )

if __name__=="__main__":
    proj_root_dir = Path("/rhome/mdodson/emulate-era5-land")
    pkl_dir = proj_root_dir.joinpath("data/permutations")
    fig_dir = proj_root_dir.joinpath("figures/permutations")

    ## era5
    '''
    slabels,sdata = pkl.load(Path("data/static/era5_static.pkl").open("rb"))
    m_valid_base = sdata[slabels.index("m_valid")].astype(bool)
    m_lakec =  sdata[slabels.index("lakec")] < .15
    m_land = sdata[slabels.index("landmask")] >= .8
    m_valid = m_valid_base & m_lakec & m_land
    '''

    ## nldas2
    #'''
    slabels,sdata = pkl.load(Path(
        "data/static/nldas_static_cropped.pkl").open("rb"))
    m_valid = sdata[slabels.index("m_valid")].astype(bool)
    #'''

    workers = 20
    enum_start = 0
    dataset_name = "nldas2"

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
            pkl_path = pkl_dir.joinpath(
                    f"permutation_{dataset_name}_cycle_{i:03}.pkl")
            pkl.dump((a, np.stack([perm,r_perm], axis=-1), stats),
                    pkl_path.open("wb"))
    '''


    ## Use convolutional method to generate some permutations
    '''
    default_args = {"coord_array":latlon, "return_stats":True, "debug":True}
    args = [{
        **dict(zip(config_labels_conv,c)), **default_args
        } for c in configs_conv]
    with Pool(workers) as pool:
        for i,(a,r) in enumerate(
                pool.imap_unordered(mp_get_permutation_conv,args),enum_start):
            perm,stats = r
            r_perm = np.asarray(tuple(zip(*sorted(zip(
                list(perm), range(len(perm))
                ), key=lambda v:v[0])))[1])
            pkl_path = pkl_dir.joinpath(
                f"permutation_{dataset_name}_conv_{i:03}.pkl")
            pkl.dump((a, np.stack([perm,r_perm], axis=-1), stats),
                    pkl_path.open("wb"))
    '''

    ## generate single permutation with conv method
    '''
    perm,stats = get_permutation_conv(
            coord_array=latlon,
            dist_threshold=1.5,
            reperm_cap=3,
            shuffle_frac=.5,
            seed=200007221750,
            return_stats=True,
            debug=True
            )
    print(perm)
    '''


    ##exit(0)

    ## Check how many chunks it would take to extract certain subgrids
    substrs = [
            #"026", ##  669  852 1042  806  |  9.03  4.26
            #"046", ##  528  661  800  572  |  6.75  3.09
            #"047", ##  453  575  685  489  |  5.92  3.43
            #"012", ##  322  417  473  377  |  4.29  3.19
            #"005", ##  310  415  453  372  |  4.16  3.28
            #"019", ##  311  415  462  368  |  4.15  3.27
            #"064",
            #"066",
            #"067",
            #"068",
            #f"{v:03}" for v in range(219,264),
            "nldas2",
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
    #print(np.amin(latlon, axis=0), np.amax(latlon, axis=0))
    for pf in print_pkls:
        pnum = int(pf.stem.split("_")[-1])
        #mask = m_valid if pnum>=75 else m_valid_base ## for era5
        mask = m_valid
        coords = np.stack([
            sdata[slabels.index("lat")][mask],
            sdata[slabels.index("lon")][mask],
            ], axis=1)
        try:
            plot_perm_pkl(
                    perm_pkl=pf,
                    coords=coords,
                    subgrids=test_subgrids,
                    valid_mask=mask,
                    fig_dir=fig_dir,
                    chunk_size=64,
                    plot_stats=True,
                    plot_sparse_chunks=True,
                    num_sparse_chunks=50,
                    seed=200007221750,
                    debug=True,
                    )
        except Exception as e:
            print(f"Failed for {pf.name}")
            #print(e)
            raise e
