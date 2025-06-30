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
from datetime import datetime

def _gen_era5_snveg_daily(file_path:Path):
    """ """
    gf = pygrib.open(file_path.as_posix())
    lat,lon = gf[1].latlons()
    rec_labels = ["sd", "var67", "var66"]
    daily_recs = len(rec_labels) * 24
    assert gf.messages % daily_recs == 0
    for i in range(gf.messages // daily_recs):
        gf.seek(i*daily_recs)
        rec_array = [x.values for x in gf.read(daily_recs)]
        rec_array = np.stack([
            rec_array[j*len(rec_labels):(j+1)*len(rec_labels)]
            for j in range(24)
            ], axis=0)
        rec_array = np.transpose(rec_array, (0,2,3,1))
        yield (rec_labels,rec_array)

def _gen_era5_soil_daily(file_path:Path):
    """ """
    gf = pygrib.open(file_path.as_posix())
    lat,lon = gf[1].latlons()
    rec_labels = ["skt","stl1","stl2","stl3","stl4","swvl1","swvl2","swvl3",
            "swvl4","var251","e","tp"]
    daily_recs = len(rec_labels) * 24
    assert gf.messages % daily_recs == 0
    for i in range(gf.messages // daily_recs):
        gf.seek(i*daily_recs)
        rec_array = [x.values for x in gf.read(daily_recs)]
        rec_array = np.stack([
            rec_array[j*len(rec_labels):(j+1)*len(rec_labels)]
            for j in range(24)
            ], axis=0)
        rec_array = np.transpose(rec_array, (0,2,3,1))
        yield (rec_labels,rec_array)

def _gen_era5_rad_daily(file_path:Path):
    """ """
    gf = pygrib.open(file_path.as_posix())
    lat,lon = gf[1].latlons()
    rec_labels = ["fal", "slhf", "ssr", "str", "sshf", "ssrd", "strd"]
    daily_recs = len(rec_labels) * 24
    assert gf.messages % daily_recs == 0
    for i in range(gf.messages // daily_recs):
        gf.seek(i*daily_recs)
        rec_array = [x.values for x in gf.read(daily_recs)]
        rec_array = np.stack([
            rec_array[j*len(rec_labels):(j+1)*len(rec_labels)]
            for j in range(24)
            ], axis=0)
        rec_array = np.transpose(rec_array, (0,2,3,1))
        yield (rec_labels,rec_array)

def _gen_era5_wbgt_daily(file_path:Path):
    """ """
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
        chunk_shape, label_mapping={}, m_valid=None, debug=False):
    """
    Extracts a full year of era5 data from monthly series of files including
    "rad" "snveg" and "soil" grib1 files and "wbgt" netCDFs, which are yielded
    by the day by _get_era5_snveg_daily, _gen_era5_rad_daily,
    _gen_era5_soil_daily, and _gen_era5_wbgt_daily.

    :@param file_dict: Dict mapping integer months [1,12] to a dict of files
        identified by their data category, one of: (rad, snveg, soil, wbgt)
        like "era5land_{data_category}_vars_{YYYYmm}.{grib | nc}"
    :@param out_h5_path: Path to the full-year hdf5 file created by this method
    :@param static_array: array of separately-extracted time-invariant values
        to be stored alongside the dynamic and time coordinate arrays.
    :@param chunk_shape: 4-tuple of integers (T,Y,X,F) describing the number of
        elements per chunk along each dynamic array axis.
    """
    extract_methods = {
            "rad":_gen_era5_rad_daily,
            "snveg":_gen_era5_snveg_daily,
            "soil":_gen_era5_soil_daily,
            "wbgt":_gen_era5_wbgt_daily,
            }

    H5F,D,S,T = None,None,None,None
    cur_h5_ix = 0
    for mix in range(1,13):
        ## declare generators for this month
        gens = [extract_methods[vk](file_dict[mix][vk])
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
                H5F = h5py.File(out_h5_path, "w")
                D = H5F.create_dataset(
                        name="/data/dynamic",
                        shape=(0, *arrays.shape[1:]),
                        maxshape=(None, *arrays.shape[1:]),
                        chunks=chunk_shape,
                        compression="gzip",
                        )
                static_array = static_array[m_valid]
                S = H5F.create_dataset(
                        name="/data/static",
                        shape=static_array.shape,
                        maxshape=static_array.shape,
                        )
                S[...] = static_array
                M = H5F.create_dataset(
                        name="/data/mask",
                        shape=m_valid.shape,
                        maxshape=m_valid.shape,
                        )
                M[...] = m_valid
                T = H5F.create_dataset(
                        name="/data/time",
                        shape=(0,),
                        maxshape=(None,),
                        )
                H5F["data"].attrs["dynamic"] = json.dumps({
                    "clabels":("time", "lat", "lon"),
                    "flabels":[label_mapping.get(l, l) for l in labels],
                    })
                H5F["data"].attrs["static"] = json.dumps({
                    "clabels":("lat", "lon"),
                    "flabels":[label_mapping.get(l, l) for l in static_labels],
                    })
            cur_slice = slice(cur_h5_ix, cur_h5_ix+arrays.shape[0])
            D.resize((cur_slice.stop, *arrays.shape[1:]))
            D[cur_slice] = arrays.astype(np.float32)
            T.resize((cur_slice.stop,))
            T[cur_slice] = tmp_times
            cur_h5_ix = cur_slice.stop
            if debug:
                tmpt = datetime.fromtimestamp(int(tmp_times[0]))
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
    #data_dir = Path("data")
    data_dir = Path("data")
    static_pkl = data_dir.joinpath("static/era5_static.pkl")
    out_dir = data_dir.joinpath("timegrids/")
    workers = 12
    base_h5_path = "timegrid_era5_{year}.h5"

    slabels,sdata = pkl.load(static_pkl.open("rb"))
    m_valid = sdata[slabels.index("m_valid")].astype(bool)
    label_mapping = json.load(
            data_dir.joinpath("list_feats_era5.json").open("r")
            )["label-mapping"]

    extract_years = list(range(2012,2024))
    era5_dirs = [data_dir.joinpath(f"era5/{y}") for y in extract_years]
    extract_paths = {}
    for dpath in era5_dirs:
        for fp in dpath.iterdir():
            ts = datetime.strptime(fp.stem.split("_")[-1], "%Y%m")
            vt = fp.stem.split("_")[1]
            if ts.year not in extract_paths.keys():
                extract_paths[ts.year] = {}
            if ts.month not in extract_paths[ts.year].keys():
                extract_paths[ts.year][ts.month] = {}
            extract_paths[ts.year][ts.month][vt] = fp

    args = [{
        "file_dict":extract_paths[year],
        "out_h5_path":out_dir.joinpath(base_h5_path.format(year=year)),
        "static_labels":slabels,
        ## masked arrays seem to always have none of their values masked
        "static_array":np.stack([
            x if type(x)==np.ndarray else x.data for x in sdata
            ], axis=-1),
        "chunk_shape":(96,32,27),
        "m_valid":m_valid,
        "label_mapping":label_mapping,
        "debug":True,
        } for year in extract_paths.keys()]

    with Pool(workers) as pool:
        for result in pool.imap_unordered(mp_extract_era5_year, args):
            print(f"Generated {result}")
