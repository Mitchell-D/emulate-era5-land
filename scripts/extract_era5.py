"""
Method to extract data from a continuous series of hourly NLDAS forcing and
Noah model outputs as a 'timegrid' style hdf5 file, which serves as an
intermediate data format facilitating efficient sampling and analysis.
"""
import h5py
import numpy as np
import pickle as pkl
import json
from multiprocessing import Pool
from pprint import pprint
import netCDF4 as nc
import pygrib
import subprocess
import gc
import shlex
from pathlib import Path
from datetime import datetime,timezone

from emulate_era5_land.helpers import get_permutation_inverse

def get_grib_extract_gen(rec_labels, accumulation_vars):
    """
    returns a generator that accepts a monthly grib file and the grib file
    for the previous month, and generates hourly data on a daily basis such
    that the provided accumulation_vars have been de-accumulated according to:
    https://confluence.ecmwf.int/pages/viewpage.action?pageId=197702790
    """
    def _gen_era5_grib_daily(file_path:Path, prev_file_path:Path):
        """ """
        gf = pygrib.open(file_path.as_posix())
        lat,lon = gf[1].latlons()
        #rec_labels = ["fal", "slhf", "ssr", "str", "sshf", "ssrd", "strd"]
        ## get the last frame of records from the previous month's file for
        ## accumulations. The first timestep (0z on the first of this month) is
        ## accumulated from this value
        with pygrib.open(prev_file_path.as_posix()) as pgf:
            pgf.seek(len(pgf)-len(rec_labels))
            prev_frame = pgf.read(len(rec_labels))

        daily_recs = len(rec_labels) * 24
        assert gf.messages % daily_recs == 0
        for i in range(gf.messages // daily_recs):
            gf.seek(i*daily_recs)
            ## (F*24,Y,X)
            rec_array = [x.values for x in gf.read(daily_recs)]
            ## (24,Y,X,F)
            rec_array = np.stack([
                rec_array[j*len(rec_labels):(j+1)*len(rec_labels)]
                for j in range(24)
                ], axis=0)
            new_last_frame = np.copy(rec_array[-1,:,:,:])
            rec_array = np.transpose(rec_array, (0,2,3,1))
            for k,l in enumerate(rec_labels):
                if l in accumulation_vars:
                    rec_array[0,:,:,k] = rec_array[0,:,:,k] - prev_frame[:,:,k]
                    rec_array[2:,:,:,k] = np.diff(rec_array[1:,:,:,k], axis=0)
            prev_frame = new_last_frame
            yield (rec_labels,rec_array)
    return _gen_era5_grib_daily

def _gen_era5_wbgt_daily(file_path:Path, prev_file_path:Path):
    """ """
    pd = nc.Dataset(prev_file_path, "r")
    d = nc.Dataset(file_path, "r")
    ## ignoring ssrd because it is redundant with the rad files
    labels = ["d2m","t2m","u10","v10","sp"]
    num_times = d.variables["valid_time"].size
    assert num_times % 24 == 0
    for i in range(num_times // 24):
        tmp_array = []
        for l in labels:
            tmp_array.append(d.variables[l][24*i:24*(i+1)])
        tmp_array = np.stack(tmp_array, axis=-1)
        tmp_times = d.variables["valid_time"][24*i:24*(i+1)]
        yield (labels, tmp_array, tmp_times)

def mp_extract_era5_year(args):
    return extract_era5_year(**args)

def extract_era5_year(file_dict, out_h5_path, static_labels, static_array,
        chunk_shape, permutation=None, label_mapping={}, m_valid=None,
        file_dtype=np.float32, debug=False):
    """
    Extracts a full year of era5 data from monthly series of files including
    "rad" "snveg" and "soil" grib1 files and "wbgt" netCDFs, which are yielded
    by the day by _get_era5_snveg_daily, _gen_era5_rad_daily,
    _gen_era5_soil_daily, and _gen_era5_wbgt_daily.

    :@param file_dict: Dict mapping integer months [1,12] to a dict of 2-tuples
        identified by their data category, one of: (rad, snveg, soil, wbgt).
        The tuples must be organized like (prev_month, cur_month) for each
        data file type in order to rectify accumulated variables
    :@param out_h5_path: Path to the full-year hdf5 file created by this method
    :@param static_array: array of separately-extracted time-invariant values
        to be stored alongside the dynamic and time coordinate arrays.
    :@param chunk_shape: 4-tuple of integers (T,Y,X,F) describing the number of
        elements per chunk along each dynamic array axis.
    :@param permutation: 1D Integer array with the same size as number of valid
        pixels in m_valid. If provided, static and dynamic spatial axes will
        be permuted as such before storage.
    """
    ## generators expect args like (cur_month_file, prev_month_file)
    extract_methods = {
            "snveg":get_grib_extract_gen(
                rec_labels=["sd", "var67", "var66"],
                accumulation_vars=[],
                ),
            "soil":get_grib_extract_gen(
                rec_labels=["skt","stl1","stl2","stl3","stl4","swvl1","swvl2",
                    "swvl3","swvl4","var251","e","tp"],
                accumulation_vars=["var251","e","tp"],
                ),
            "rad":get_grib_extract_gen(
                rec_labels=["fal","slhf","ssr","str","sshf","ssrd","strd"],
                accumulation_vars=["slhf","ssr","str","sshf","ssrd","strd"],
                ),
            "wbgt":_gen_era5_wbgt_daily,
            }

    H5F,D,S,T = None,None,None,None
    cur_h5_ix = 0
    for mix in range(1,13):
        ## declare generators for this month
        gens = [extract_methods[vk](*file_dict[mix][vk])
                for vk in ["wbgt", "snveg", "rad", "soil"]]
        labels = []
        got_labels = False
        ## cycles once per day
        while True:
            arrays = []
            tmp_times = None
            try:
                for g in gens:
                    tmp_labels,*tmp_array = next(g)
                    ## grab the times returned by the netcdf generator
                    if len(tmp_array) == 1:
                        tmp_array = tmp_array[0]
                    else:
                        tmp_array,tmp_times = tmp_array
                    if not got_labels:
                        labels += tmp_labels
                    if m_valid is None:
                        m_valid = np.full(tmp_array.shape[1:3], True)
                    arrays.append(tmp_array[:,m_valid])
                got_labels = True
            except StopIteration:
                break
            gc.collect()
            arrays = np.concatenate(arrays, axis=-1)
            if H5F is None:
                assert not out_h5_path.exists(), out_h5_path.name
                H5F = h5py.File(out_h5_path, "w")
                D = H5F.create_dataset(
                        name="/data/dynamic",
                        shape=(0, *arrays.shape[1:]),
                        maxshape=(None, *arrays.shape[1:]),
                        chunks=chunk_shape,
                        compression="gzip",
                        dtype="f4",
                        )
                static_array = static_array[m_valid]
                if permutation is None:
                    print(f"WARNING: no spatial permutation provided")
                    permutation = np.arange(static_array.shape[0])
                S = H5F.create_dataset(
                        name="/data/static",
                        shape=static_array.shape,
                        maxshape=static_array.shape,
                        dtype="f8",
                        )
                S[...] = static_array[permutation]
                M = H5F.create_dataset(
                        name="/data/mask",
                        shape=m_valid.shape,
                        maxshape=m_valid.shape,
                        dtype="b",
                        )
                M[...] = m_valid
                T = H5F.create_dataset(
                        name="/data/time",
                        shape=(0,),
                        maxshape=(None,),
                        dtype="f8",
                        )
                P  = H5F.create_dataset(
                        name="/data/permutation",
                        shape=(2,permutation.size),
                        dtype="u4",
                        )
                P[...] = np.stack([
                    permutation, get_permutation_inverse(permutation)
                    ], axis=0)
                H5F["data"].attrs["dynamic"] = json.dumps({
                    "clabels":("time", "space"),
                    "flabels":[label_mapping.get(l, l) for l in labels],
                    })
                H5F["data"].attrs["static"] = json.dumps({
                    "clabels":("space",),
                    "flabels":[label_mapping.get(l, l) for l in static_labels],
                    })
            cur_slice = slice(cur_h5_ix, cur_h5_ix+arrays.shape[0])
            D.resize((cur_slice.stop, *arrays.shape[1:]))
            D[cur_slice] = arrays.astype(file_dtype)[:,permutation]
            T.resize((cur_slice.stop,))
            T[cur_slice] = tmp_times
            cur_h5_ix = cur_slice.stop
            if debug:
                tmpt = datetime.fromtimestamp(int(tmp_times[0]),
                        tz=timezone.utc)
                print(f"Extracted {arrays.shape} at {tmpt}; now {D.shape}")
            H5F.flush()
    return out_h5_path

"""
fg_dict_dynamic = {
        "clabels":("time","lat","lon"),
        "flabels":tuple(nldas_labels+noahlsm_labels),
        "meta":{ ## extract relevant info parsed from wgrib
            "nldas":[(d["name"], d["param_pds"], d["lvl_str"])
                for d in nldas_info],
            "noah":[(d["name"], d["param_pds"], d["lvl_str"])
                for d in noah_info],
            }
        }
fg_dict_static = {
        "clabels":("lat","lon"),
        "flabels":tuple(static_labels),
        "meta":{}
        }
g.attrs["dynamic"] = json.dumps(fg_dict_dynamic)
g.attrs["static"] = json.dumps(fg_dict_static)
"""

if __name__=="__main__":
    ## Directories should contain only files that should be loaded to the hdf5
    data_dir = Path("data")
    #out_dir = data_dir.joinpath("timegrids-new/")
    out_dir = data_dir.joinpath("/rstor/mdodson/era5/timegrids-new/")
    static_pkl = data_dir.joinpath("static/era5_static.pkl")
    perm_pkl = data_dir.joinpath("permutations/permutation_210.pkl")
    era5_dir = data_dir.joinpath("era5")
    workers = 12
    base_h5_path = "timegrid_era5_{year}.h5"

    ## load the static data and boolean valid mask
    slabels,sdata = pkl.load(static_pkl.open("rb"))
    m_valid_base = sdata[slabels.index("m_valid")].astype(bool)
    m_lakec =  sdata[slabels.index("lakec")] < .15
    m_land = sdata[slabels.index("landmask")] >= .8
    m_valid = m_valid_base & m_lakec & m_land
    ## load label conversions
    label_mapping = json.load(
            data_dir.joinpath("list_feats_era5.json").open("r")
            )["label-mapping"]
    ## load the desired spatial permutation
    _,perm,_ = pkl.load(perm_pkl.open("rb"))

    ## it's critical that the stored data abides this naming structure
    path_templates = {
            "rad":"{year}/era5land_rad_vars_{year}{month:02}.grib",
            "snveg":"{year}/era5land_snveg_vars_{year}{month:02}.grib",
            "soil":"{year}/era5land_soil_vars_{year}{month:02}.grib",
            "wbgt":"{year}/era5land_wbgt_vars_{year}{month:02}.nc",
            }

    extract_years = list(range(2012,2024))
    ## include the december of the prior year for de-accumulation
    all_my = [(m,y) for m in range(1,13) for y in extract_years]
    all_my = [(12,extract_years[0]-1)] + all_my
    all_my = list(zip(all_my[:-1], all_my[1:]))
    ## list of (previous,current) 2-tuples like (month,year)
    extract_paths = {}
    ## construct dict hashed like [year][month][file_type] that maps to a
    ## 2-tuple (current_month_path, prev_month_path)
    for (pm,py),(cm,cy) in all_my:
        month_paths = {
                ftype:(era5_dir.joinpath(tmplt.format(month=cm,year=cy)),
                    era5_dir.joinpath(tmplt.format(month=pm,year=py)))
                for ftype,tmplt in path_templates.items()
                }
        for cur_mp,prev_mp in month_paths.values():
            assert cur_mp.exists(),cur_mp.as_posix()
            assert prev_mp.exists(),prev_mp.as_posix()
        if cy not in extract_paths.keys():
            extract_paths[cy] = {}
        extract_paths[cy][cm] = month_paths

    args = [{
        "file_dict":extract_paths[year],
        "out_h5_path":out_dir.joinpath(base_h5_path.format(year=year)),
        "static_labels":slabels,
        ## masked arrays seem to always have none of their values masked
        "static_array":np.stack([
            x if type(x)==np.ndarray else x.data for x in sdata
            ], axis=-1),
        "chunk_shape":(192,64,27),
        "m_valid":m_valid,
        "label_mapping":label_mapping,
        "permutation":perm[:,0],
        "debug":True,
        } for year in extract_paths.keys()]

    with Pool(workers) as pool:
        for result in pool.imap_unordered(mp_extract_era5_year, args):
            print(f"Generated {result}")
