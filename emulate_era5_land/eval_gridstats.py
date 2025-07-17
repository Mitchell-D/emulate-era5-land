"""
Methods for interacting with 'timegrid' style HDF5s, which each cover 1/6 of
CONUS over a 3 month period, and store their data as a (T,P,Q,F) dynamic grid
with (P,Q,F) static grids and (T,1) timestamps
"""
import numpy as np
import pickle as pkl
import random as rand
import json
import gc
import h5py
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool

from emulate_era5_land.helpers import _parse_feat_idxs,_calc_feat_array

#from testbed.list_feats import nldas_record_mapping,noahlsm_record_mapping

def _mp_get_gs_varsum(args):
    return _get_gs_varsum(**args)

def _mp_get_gs_mmsc(args):
    return _get_gs_mmsc(**args)

def _get_gs_varsum(timegrid:Path, means:np.array, depermute=False,
        time_sector_size=None, derived_feats_serial:list=[]):
    """
    get the standard deviation of this timegrid's data per
    month/ToD/pixel/feature combination, and return the result as a
    (12,24,P,F,1) array of floats and a (12,24) array of int counts.

    :@param timegrid: File from which to extract the data
    :@param means: (12,24,P,F) global mean values per month,hour,pixel,feat
        combination used to calculate variance. If 'depermute' is set to True,
        it is assumed that the means are already in de-permuted order, so the
        stored variables will accordingly be depermuted before calculating.
    :@param depermute: If True, extract the inverse spatial permutation
        from the timegrid file and use it to reorder the extracted array
    :@param time_sector_size: Number of elements along the time axis to
        extract at a time. Default to hdf5 chunk size along time axis.
    :@param derived_feats_serial: List of tuples like:
        (dflabel:str, (dargs:list[str], sargs:list[str], lambda_str:str))
        which encode a way of calculating a new feature called {dflabel}
        using stored dynamic and static features {dargs} and {sargs} that
        are provided to the compiled {lambda_str}.
    """
    tg_open = h5py.File(timegrid,"r",rdcc_nbytes=128*1024**2,rdcc_nslots=256)

    ## extract month and ToD indeces associated with each timestep
    tmp_months = np.array([
        t.month-1 for t in tuple(map(
            lambda s:datetime.fromtimestamp(int(s)),
            tg_open["/data/time"][...]))
        ], dtype=np.uint8)
    tmp_tods = np.array([
        t.hour + (t.minute // 30) for t in tuple(map(
            lambda s:datetime.fromtimestamp(int(s)),
            tg_open["/data/time"][...]))
        ], dtype=np.uint8)

    ## extract the labels, static, dynamic, and permutation data
    tg_shape = tg_open["/data/dynamic"].shape
    tg_dlabels = json.loads(
            tg_open["data"].attrs["dynamic"])["flabels"]
    tg_slabels = json.loads(
            tg_open["data"].attrs["static"])["flabels"]
    tg_static = tg_open["/data/static"][...]
    tg_dynamic = tg_open["/data/dynamic"]
    if depermute:
        inv_perm = tg_open["/data/permutation"][1]
    else:
        inv_perm = None
    all_feats = [df[0] for df in derived_feats_serial] + list(tg_dlabels)

    ## determine sector slices along time axis
    dix = time_sector_size
    tslices = [slice(int(ix),int(ix+dix))
            for ix in np.arange(tg_shape[0]//dix)*dix]
    if rix := tg_shape[0] % dix:
        tslices.append(slice(int(tslices[-1].stop), int(tslices[-1].stop+rix)))

    ## get the mappings and functions for extracting stored and derived feats.
    sfix,drv_info,_ = _parse_feat_idxs(
            out_feats=all_feats,
            src_feats=list(tg_dlabels),
            static_feats=list(tg_slabels),
            derived_feats=dict(derived_feats_serial),
            )

    ## construct a (M,T) counts array and (M,T,P,Fd,1) array for
    ## (min, max, sum) per combination of (month, ToD, pixel, feature)
    varsum = np.zeros((12,24,*tg_shape[1:3],1), dtype=np.float64)
    counts = np.zeros((12,24), dtype=int)
    for slc in tslices:
        tmp_arrs = []
        ## evaluate derived features and reorder stored feats as requested.
        tmpx = _calc_feat_array(
                src_array=tg_dynamic[slc], ## (T, P, Fd)
                static_array=tg_static, ## (P, Fs)
                stored_feat_idxs=sfix,
                derived_data=drv_info,
                )
        ## means must also be depermuted; do so before calculating the sum
        if depermute:
            tmpx = tmpx[:,inv_perm]
        midxs = tmp_months[slc]
        tidxs = tmp_tods[slc]
        for i in range(tmpx.shape[0]):
            ## Determine min/max (P,F) for this timestep
            counts[midxs[i],tidxs[i]] += 1
            tmp_means = means[midxs[i],tidxs[i],:,:]
            varsum[midxs[i],tidxs[i],:,:,0] += (tmpx[i]-tmp_means)**2
        print(counts)
        gc.collect()
    return (counts,varsum)

def _get_gs_mmsc(timegrid:Path, depermute=False, time_sector_size=None,
        derived_feats_serial:list=[]):
    """
    get the minimum, maximum, sum, and counts of this timegrid's data per
    month/ToD/pixel/feature combination, and return the result as a
    (12,24,P,F,3) array of floats and a (12,24) array of int counts.

    :@param timegrid: File from which to extract the data
    :@param depermute: If True, extract the inverse spatial permutation
        from the timegrid file and use it to reorder the extracted array
    :@param time_sector_size: Number of elements along the time axis to
        extract at a time. Default to hdf5 chunk size along time axis.
    :@param derived_feats_serial: List of tuples like:
        (dflabel:str, (dargs:list[str], sargs:list[str], lambda_str:str))
        which encode a way of calculating a new feature called {dflabel}
        using stored dynamic and static features {dargs} and {sargs} that
        are provided to the compiled {lambda_str}.
    """
    tg_open = h5py.File(timegrid,"r",rdcc_nbytes=128*1024**2,rdcc_nslots=256)

    ## extract month and ToD indeces associated with each timestep
    tmp_months = np.array([
        t.month-1 for t in tuple(map(
            lambda s:datetime.fromtimestamp(int(s)),
            tg_open["/data/time"][...]))
        ], dtype=np.uint8)
    tmp_tods = np.array([
        t.hour + (t.minute // 30) for t in tuple(map(
            lambda s:datetime.fromtimestamp(int(s)),
            tg_open["/data/time"][...]))
        ], dtype=np.uint8)

    ## extract the labels, static, dynamic, and permutation data
    tg_shape = tg_open["/data/dynamic"].shape
    tg_dlabels = json.loads(
            tg_open["data"].attrs["dynamic"])["flabels"]
    tg_slabels = json.loads(
            tg_open["data"].attrs["static"])["flabels"]
    tg_static = tg_open["/data/static"][...]
    tg_dynamic = tg_open["/data/dynamic"]
    if depermute:
        inv_perm = tg_open["/data/permutation"][1]
    else:
        inv_perm = None
    all_feats = [df[0] for df in derived_feats_serial] + list(tg_dlabels)

    ## determine sector slices along time axis
    dix = time_sector_size
    tslices = [slice(int(ix),int(ix+dix))
            for ix in np.arange(tg_shape[0]//dix)*dix]
    if rix := tg_shape[0] % dix:
        tslices.append(slice(int(tslices[-1].stop), int(tslices[-1].stop+rix)))

    ## get the mappings and functions for extracting stored and derived feats.
    sfix,drv_info,_ = _parse_feat_idxs(
            out_feats=all_feats,
            src_feats=list(tg_dlabels),
            static_feats=list(tg_slabels),
            derived_feats=dict(derived_feats_serial),
            )

    ## construct a (M,T) counts array and (M,T,P,Fd,3) array for
    ## (min, max, sum) per combination of (month, ToD, pixel, feature)
    mms = np.zeros((12,24,*tg_shape[1:3],3), dtype=np.float64)
    counts = np.zeros((12,24), dtype=int)
    for slc in tslices:
        tmp_arrs = []
        ## evaluate derived features and reorder stored feats as requested.
        tmpx = _calc_feat_array(
                src_array=tg_dynamic[slc], ## (T, P, Fd)
                static_array=tg_static, ## (P, Fs)
                stored_feat_idxs=sfix,
                derived_data=drv_info,
                )
        midxs = tmp_months[slc]
        tidxs = tmp_tods[slc]
        for i in range(tmpx.shape[0]):
            #mms[midxs[i],tidxs[i],:,:] =
            counts[midxs[i],tidxs[i]] += 1
            ## Determine min/max (P,F) for this timestep
            mms[midxs[i],tidxs[i],:,:,0] = np.min([
                mms[midxs[i],tidxs[i],:,:,0], tmpx[i],
                ], axis=0)
            mms[midxs[i],tidxs[i],:,:,1] = np.max([
                mms[midxs[i],tidxs[i],:,:,1], tmpx[i],
                ], axis=0)
            mms[midxs[i],tidxs[i],:,:,2] += tmpx[i]
        print(counts)
        gc.collect()
    if depermute:
        mms = mms[:,:,inv_perm,:,:]
    return (counts,mms)

def par_var(n_a, mean_a, m2_a, n_b, mean_b, m2_b):
    """
    cumulatively estimate the second moment of 2 data sources.

    variance = m2 / (n-1)
    """
    n_ab = n_a + n_b
    mean_ab = (n_a*mean_a + n_b*mean_b) / n_ab
    m2_ab = m2_a + m2_b + (mean_b - mean_a)**2 * n_a*n_b/n_ab
    return (n_ab, mean_ab, m2_ab)

def make_gridstat_hdf5(timegrids:list, out_file:Path, depermute=True,
        time_sector_size=None, derived_feats_serial:list=[], nworkers=1,
        debug=False):
    """
    Calculate pixel-wise monthly min, max, mean, and standard deviation of each
    stored dynamic and derived feature in the timegrids and store the
    statistics alongside static data in a new hdf5 file.

    :@param timegrids: List of timegrids that cover the same spatial domain,
        which will all be incorporated into the pixelwise monthly calculations.
    :@param out_file: Path to a non-existing file to write gridstats to.
    :@param derived_feats_serial: Provide a dict mapping NEW feature labels to
        a 3-tuple (dynamic_args, static_args, lambda_str) where the args are
        each tuples of existing dynamic/static labels, and lambda_str contains
        a string-encoded function taking 2 arguments (dynamic,static) of tuples
        containing the corresponding arrays, and returns the subsequent new
        feature after calculating it based on the arguments. These will be
        invoked if the new derived feature label appears in one of the window,
        horizon, or pred feature lists.
    """
    ## extract shape, labels, permutation, static data, and valid mask and
    ## verify that the timegrids are uniform where they need to be.
    tg_shape = None
    for tg in timegrids:
        ## 128MB cache with 256 slots; each chunk is a little over 1/3 MB
        tg_open = h5py.File(tg, "r", rdcc_nbytes=128*1024**2, rdcc_nslots=256)
        if tg_shape is None:
            tg_shape = tg_open["/data/dynamic"].shape
            ## collect dynamic and static feature labels
            tg_dlabels = json.loads(
                    tg_open["data"].attrs["dynamic"])["flabels"]
            tg_slabels = json.loads(
                    tg_open["data"].attrs["static"])["flabels"]
            tg_static = tg_open["/data/static"][...]
            tg_perm = tg_open["/data/permutation"][...].astype(int)
            tg_mask = tg_open["/data/mask"][...]
            if depermute:
                tg_static = tg_static[tg_perm[1]] ## invert stored permutation
        else:
            assert tg_shape[1:] == tg_open["/data/dynamic"].shape[1:], \
                    "Timegrid grid shapes & feature count must be uniform"
            assert tg_static.shape == tg_open["/data/static"].shape, \
                    "Timegrid grid shapes & feature count must be uniform"
            assert np.all(tg_mask == tg_open["/data/mask"][...]), \
                    "Timegrid 2d valid mask must be uniform"
            assert tuple(tg_dlabels) == tuple(
                    json.loads(tg_open["data"].attrs["dynamic"])["flabels"])
            if not depermute:
                assert np.all(tg_perm == tg_open["/data/permutation"][...])
        tg_open.close()

    ## collect all feature labels, whether stored  or derived.
    ## process derived features first since they are most likely to fail
    all_flabels =  list(dict(derived_feats_serial).keys()) + tg_dlabels
    stats_shape = (12, 24, tg_shape[1], len(all_flabels), 4)

    ## initialize the gridstats hdf5 file
    F = h5py.File(name=out_file.as_posix(), mode="w-", rdcc_nbytes=256*1024**2)
    ## stats shape for 12 months on (P,Q,F) grid with 4 stats per feature
    ## create chunked hdf5 datasets for gridstats and counts
    G = F.create_dataset(
            name="/data/gridstats",
            shape=stats_shape,
            maxshape=stats_shape,
            chunks=(3,24,512,8,2),
            compression="gzip",
            dtype="f8",
            )
    C = F.create_dataset(
            name="/data/counts",
            shape=(12,24), ## counts only vary along time axes
            dtype="u4",
            )
    ## Create and load the static datasets, permutations, and 2d valid mask
    S = F.create_dataset(
            name="/data/static",
            shape=tg_static.shape,
            dtype="f8",
            )
    S[...] = tg_static
    P = F.create_dataset(
            name="/data/permutation",
            shape=tg_perm.shape,
            dtype="u4",
            )
    P[...] = tg_perm
    M = F.create_dataset(
            name="/data/mask",
            shape=tg_mask.shape,
            dtype="b",
            )
    M[...] = tg_mask

    ## attempt to follow CFD convention even though the api isn't finished
    F["data"].attrs["gridstats"] = json.dumps({
        "flabels":all_flabels,
        "clabels":("month","tod","space"),
        "mlabels":("min","max","meansum","varsum"),
        })
    F["data"].attrs["static"] = json.dumps({
        "flabels":tg_slabels,
        "clabels":("space",),
        })
    F["data"].attrs["counts"] = json.dumps({
        "flabels":tuple(),
        "clabels":("month","tod"),
        })
    F["data"].attrs["permutation"] = json.dumps({
        "flabels":("fwd","inv"),
        "clabels":("space",),
        })
    F["data"].attrs["mask"] = json.dumps({
        "flabels":tuple(),
        "clabels":("lat","lon"),
        })
    F["data"].attrs["meta"] = json.dumps({
        "derived_feats":json.dumps(derived_feats_serial),
        "timegrids":json.dumps([p.name for p in timegrids]),
        })

    ## first pass: compute min, max, and sum values
    args = [{
        "timegrid":tg, "depermute":depermute,
        "time_sector_size":time_sector_size,
        "derived_feats_serial":derived_feats_serial,
        } for tg in timegrids]

    with Pool(nworkers) as pool:
        mms_total = np.full((*stats_shape[:-1], 3), np.nan)
        counts_total = np.zeros((12,24))
        for mms,counts in pool.imap_unordered(_mp_get_gs_mmsc, args):
            if mms_total is None:
                mms_total = mms
            else:
                mms_total[...,0] = np.nanmin([
                    mms_total[...,0], mms[...,0]
                    ], axis=0)
                mms_total[...,1] = np.nanmax([
                    mms_total[...,1], mms[...,1]
                    ], axis=0)
                mms_total[...,2] += mms_total[...,2]
                counts_total += counts

    ## load the
    G[:,:,:,:,:3] = mms_total
    C[...] = counts_total
    F.flush()

    ## second pass: compute standard deviation using mean values
    global_means = mms_total[...,2] / counts_total[:,:,np.newaxis,np.newaxis]
    #global_means = np.zeros(stats_shape[:-1])
    args = [{
        "timegrid":tg, "means":global_means, "depermute":depermute,
        "time_sector_size":time_sector_size,
        "derived_feats_serial":derived_feats_serial
        } for tg in timegrids]
    with Pool(nworkers) as pool:
        varsum_total = np.zeros(*stats_shape[:-1], 1)
        counts_total_varsum = np.zeros((12,24))
        for varsum,counts in pool.imap_unordered(_mp_get_gs_varsum, args):
            varsum_total += varsum
            counts_total_varsum += counts

    print(f"{np.all(counts_total == counts_total_varsum) = }")
    G[:,:,:,:,3] = varsum
    F.flush()
    F.close()

if __name__=="__main__":
    data_dir = Path("/rstor/mdodson/era5")
    tg_dir = data_dir.joinpath("timegrids-new")
    #static_pkl_path = data_dir.joinpath("static/nldas_static_cropped.pkl")
    gridstat_dir = data_dir.joinpath("gridstats")

    substr = "timegrid_era5"
    timegrids = sorted([p for p in tg_dir.iterdir() if substr in p.name])

    ## Generate gridstats file over a single region
    #'''
    print(timegrids)
    make_gridstat_hdf5(
            timegrids=timegrids,
            out_file=gridstat_dir.joinpath(
                f"gridstats_era5_2012-2023.h5"),
            depermute=True,
            time_sector_size=24*14,
            nworkers=6,
            debug=True,
            )
    #'''
