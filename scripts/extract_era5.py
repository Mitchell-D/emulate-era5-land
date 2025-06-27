"""
Method to extract data from a continuous series of hourly NLDAS forcing and
Noah model outputs as a 'timegrid' style hdf5 file, which serves as an
intermediate data format facilitating efficient sampling and analysis.
"""
import numpy as np
import pickle as pkl
import json
import multiprocessing as mp
from pprint import pprint
import netCDF4 as nc
import pygrib
import subprocess
import gc
import shlex
from pathlib import Path
from datetime import datetime

#from krttdkit.acquire import grib_tools, gesdisc

def get_grib1_grid(grb1_path:Path):
    """
    Parses only the gridded data from a grib1 file, not the latlons or wgrib
    """
    f = grb1_path
    assert f.exists()
    gf = pygrib.open(f.as_posix())
    gf.seek(0)
    return [d.data()[0] for d in gf]

def extract_timegrid(nldas_grib_paths:Path, noahlsm_grib_paths:Path,
        static_pkl_path:Path, out_h5_dir:Path, out_path_prepend_str:str,
        nldas_labels:list, noahlsm_labels:list, subgrid_x=32, subgrid_y=32,
        time_chunk=128, space_chunk=16, feat_chunk=8, wgrib_bin="wgrib",
        crop_y=(0,0), crop_x=(0,0), valid_mask=None, fill_value=9999.,
        workers=1):
    """
    Multiprocessed method for converting directories of NLDAS2 and Noah-LSM
    grib1 files (acquired from the GES DISC DAAC) into a single big hdf5 file
    containing only the records specified above in noahlsm_record_mapping and
    nldas_record_mapping, adhering to the order that they appear in the lists.

    The files in each directory must be uniformly separated in time (ie same
    dt between each consecutive file), and each NLDAS2 file must directly
    correspond to a simultaneous NoahLSM file.

    :@param nldas_grib_dir: Dir containing NLDAS2 data from GES DISC
    :@param noahlsm_grib_dir: Dir containing NLDAS2 NoahLSM data from GES DISC
    :@param out_h5_dir: Directory where extracted hdf5 files are placed
    :@param out_path_template: String out file stem prepended to generated
        file namees. This method appends '_yNNN_xMMM.h5' to indicate the bounds
        of each subgrid.
    :@param subgrid_x: Number of horizontal grid points per subgrid file
    :@param subgrid_y: Number of vertical grid points per subgrid file
    :@param time_chunk: Number of hourly files to chunk in each generated file
    :@param wgrib_bin: Binary file to wgrib (used to extract metadata)
    :@param workers:
    """
    ## Refuse to overwrite an existing hdf5 output file.
    #assert out_h5_dir.exists()
    #assert not out_file.exists()

    ## Pair files by the acquisition time reported in the file name

    ## Verify that the time steps between frames are consistent according
    ## to the acquisition time in the file name
    times = [t[0] for t in file_pairs]
    dt = times[1]-times[0]
    assert all(b-a==dt for b,a in zip(times[1:],times[:-1]))

    ## Extract a sample grid for setting hdf5 shape
    tmp_time,tmp_nldas,tmp_noah = file_pairs[0]
    nldas_data,nldas_info,_ = get_grib1_data(
            tmp_nldas, wgrib_bin=wgrib_bin)
    noah_data,noah_info,_ = get_grib1_data(
            tmp_noah, wgrib_bin=wgrib_bin)

    ## Determine the total shape of all provided files
    crop_y0,crop_yf = crop_y
    crop_x0,crop_xf = crop_x
    ## Make a spatial slice tuple for sub-gridding dynamic and static data
    crop_slice = (
            slice(crop_y0,nldas_data[0].shape[0]-crop_yf),
            slice(crop_x0,nldas_data[0].shape[1]-crop_xf)
            )
    full_shape = (
            len(times),
            nldas_data[0].shape[0]-crop_y0-crop_yf,
            nldas_data[0].shape[1]-crop_x0-crop_xf,
            len(nldas_labels) + len(noahlsm_labels),
            )

    print(f"hdf5 feature shape: {full_shape}")

    ## establish slices over the spatial dimensions that describe each file
    y_bins = list(subgrid_y*np.arange(full_shape[1]//subgrid_y+1))
    y_offset = full_shape[1]%subgrid_y
    y_bins += [[],[y_bins[-1]+y_offset]][y_offset>0]

    x_bins = list(subgrid_x*np.arange(full_shape[2]//subgrid_x+1))
    x_offset = full_shape[2]%subgrid_x
    x_bins += [[],[x_bins[-1]+x_offset]][x_offset>0]

    y_slices = [slice(ya,yb) for ya,yb in zip(y_bins[:-1],y_bins[1:])]
    x_slices = [slice(xa,xb) for xa,xb in zip(x_bins[:-1],x_bins[1:])]
    out_slices = [
            (y_slices[j], x_slices[i])
            for j in range(len(y_slices))
            for i in range(len(x_slices))
            ]

    ## Extract static data from the pkl made by get_static_data
    static_labels,static_data = pkl.load(static_pkl_path.open("rb"))
    static_data = np.stack(static_data, axis=-1)
    static_data = static_data[*crop_slice]
    chunk_shape = (time_chunk, space_chunk, space_chunk, feat_chunk)
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
    out_paths = []
    out_h5s = []
    ## initialize the hdf5 files with datasets for static, dynamic, and time
    ## data formatted so that they can initialize FeatureGrid classes.
    for s in out_slices: #zip(out_h5s,out_paths,out_slices):
        sgstr = f"_y{s[0].start:03}-{s[0].stop:03}" +  \
                f"_x{s[1].start:03}-{s[1].stop:03}.h5"
        new_h5_path = out_h5_dir.joinpath(out_path_prepend_str+sgstr)
        out_paths.append(new_h5_path)
        f = h5py.File(
                name=new_h5_path,
                mode="w-",
                rdcc_nbytes=128*1024**2, ## use a 128MB cache
                )
        out_h5s.append(f)
        dynamic_shape = (
                len(times),
                s[0].stop-s[0].start,
                s[1].stop-s[1].start,
                len(nldas_labels)+len(noahlsm_labels),
                )
        g = f.create_group("/data")
        ## create datasets for dynamic, static, and timestep data
        d_dynamic = g.create_dataset(
                name="dynamic",
                shape=dynamic_shape,
                chunks=chunk_shape,
                compression="gzip"
                )
        d_static = g.create_dataset(
                name="static",
                shape=(*dynamic_shape[1:3],len(static_labels)),
                compression="gzip"
                )
        d_times = g.create_dataset(name="time", shape=(len(times),))
        ## add the FeatureGrid-like json dictionaries to the attributes
        g.attrs["dynamic"] = json.dumps(fg_dict_dynamic)
        g.attrs["static"] = json.dumps(fg_dict_static)
        ## load the static data corresponding to this slice
        d_static[...] = static_data[*s]
        ## load the epoch int timesteps associated with this set of grib files
        d_times[...] = np.array([int(t.strftime("%s")) for t in times])
        print(f"Initialized {new_h5_path.as_posix()}")

    nl_rec_dict = dict(t[::-1] for t in nldas_record_mapping)
    no_rec_dict = dict(t[::-1] for t in noahlsm_record_mapping)
    nldas_records = [nl_rec_dict[k] for k in nldas_labels]
    noahlsm_records = [no_rec_dict[k] for k in noahlsm_labels]

    cur_chunk = []
    fill_count = 0
    chunk_idx = 0
    with mp.Pool(workers) as pool:
        args = [(nl,no,nldas_info,noah_info,nldas_records,noahlsm_records)
                for t,nl,no in file_pairs]
        for r in pool.imap(_parse_file, args):
            ## If a mask of valid grid values is provided, fill the invalid
            if not valid_mask is None:
                r[np.logical_not(valid_mask)] = fill_value
            ## Crop according to the user-provided boundaries, which are
            ## applied AFTER flipping to the proper vertical orientation
            r = r[*crop_slice]
            cur_chunk.append(r)
            if len(cur_chunk)==time_chunk:
                cur_chunk = np.stack(cur_chunk, axis=0)
                print(f"Loading chunk with shape: {cur_chunk.shape}")
                for i in range(len(out_h5s)):
                    ds = out_h5s[i]["/data/dynamic"]
                    ds[chunk_idx:chunk_idx+time_chunk,...] = \
                            cur_chunk[:,*out_slices[i],:]
                chunk_idx += time_chunk
                cur_chunk = []
            print(f"Completed timestep {times[fill_count]}")
            fill_count += 1
        ## After extracting all files, load any remaining partial chunks
        if len(cur_chunk) != 0:
            cur_chunk = np.stack(cur_chunk, axis=0)
            for i in range(len(out_h5s)):
                ds = out_h5s[i]["/data/dynamic"]
                ds[chunk_idx:chunk_idx+cur_chunk.shape[0]] = \
                        cur_chunk[:,*out_slices[i],:]
    for f in out_h5s:
        f.close()

def _parse_file(args:tuple):
    """
    Extract the specified nldas and noahlsm grib1 files and return the
    requested records as a uniform-size 3d array for each timestep like
    (lat, lon, feature) in the same order as the provided records, with
    all nldas records coming first, then noahlsm records.

    Note that this method only returns the consequent numpy array, so the
    record labels, geolocation, info dicts need to be kept track of externally
    (although the record entry in the info dict is used to order the array).

    args := (nldas_path, noahlsm_path,
             nldas_info, noahlsm_info,
             nldas_records, noahlsm_records)
    """
    all_data = []
    nldas_path,noahlsm_path,nldas_info,noahlsm_info, \
            nldas_records, noahlsm_records = args
    ## extract all the data from the files
    nldas_data = np.stack(get_grib1_grid(nldas_path), axis=-1)
    noahlsm_data = np.stack(get_grib1_grid(noahlsm_path), axis=-1)

    ## Extract record numbers in the order they appear in the file, then
    ## make a list of file record indexes in the user-requested order.
    nldas_file_records = tuple(nl["record"] for nl in nldas_info)
    noah_file_records = tuple(no["record"] for no in noahlsm_info)
    nldas_idxs = tuple(nldas_file_records.index(r) for r in nldas_records)
    noahlsm_idxs = tuple(noah_file_records.index(r) for r in noahlsm_records)

    nldas_data = nldas_data[...,nldas_idxs]
    noahlsm_data = noahlsm_data[...,noahlsm_idxs]
    ## flip zonally since the grib files are flipped for some reason.
    all_data = np.concatenate((nldas_data, noahlsm_data), axis=-1)[::-1]
    return all_data

def wgrib_tuples(grb1:Path, wgrib_bin="wgrib"):
    """
    Calls wgrib on the provided file as a subprocess and returns the result
    as a list of tuples corresponding to each record, the tuples having string
    elements corresponding to the available fields in the grib1 file.
    """
    wgrib_command = f"{wgrib_bin} {grb1.as_posix()}"

    out = subprocess.run(
            args=shlex.split(wgrib_command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            #capture_output=True,
            )
    return [tuple(o.split(":")) for o in out.stdout.decode().split("\n")[:-1]]

def wgrib(grb1:Path, wgrib_bin="wgrib"):
    """
    Parses wgrib fields for a grib1 file into a dict of descriptive values.
    See: https://ftp.cpc.ncep.noaa.gov/wd51we/wgrib/readme
    """
    return [{"record":int(wg[0]),
             "name":wg[3],
             "lvl_str":wg[11], # depth level
             "mdl_type":wg[12], # Model type; anl or fcst
             "date":wg[2].split("=")[-1],
             "byte":int(wg[1]),
             "param_pds":int(wg[4].split("=")[-1]), # parameter/units
             "type_pds":int(wg[5].split("=")[-1]), # layer/level type
             "vert_pds":int(wg[6].split("=")[-1]), # Vertical coordinate
             "dt_pds":int(wg[7].split("=")[-1]),
             "t0_pds":int(wg[8].split("=")[-1]),
             "tf_pds":int(wg[9].split("=")[-1]),
             "fcst_pds":int(wg[10].split("=")[-1]), # Forecast id
             #"navg":int(wg[13].split("=")[-1]), # Number of grid points in avg
             } for wg in wgrib_tuples(grb1, wgrib_bin=wgrib_bin)]

def get_grib1_data(grb1_path:Path, wgrib_bin="wgrib"):
    """
    Parses grib1 file into a series of scalar arrays of the variables,
    geographic coordinate reference grids, and information about the dataset.

    :@param grb1_path: Path of an existing grb1 file file with all scalar
        records on uniform latlon grids.
    :@return: (data, info, geo) such that:
        data -> list of uniform-shaped 2d scalar arrays for each record
        info -> list of dict wgrib results for each record, in order.
        geo  -> 2-tuple (lat,lon) of reference grid, assumed to be uniform
                for all 2d record arrays.
    """
    f = grb1_path
    assert f.exists()
    gf = pygrib.open(f.as_posix())
    geo = gf[1].latlons()
    gf.seek(0)
    # Only the first entry in data is valid for FORA0125 files, the other
    # two being the (uniform) lat/lon grid. Not sure how general this is.
    data = [ d.data()[0] for d in gf ]
    return (data, wgrib(f, wgrib_bin=wgrib_bin), geo)


def _gen_era5_snveg_daily(file_path:Path):
    """ """
    gf = pygrib.open(file_path.as_posix())
    lat,lon = gf[1].latlons()
    rec_labels = ["sd", "var67", "var66"]
    daily_recs = len(rec_labels) * 24
    assert gf.messages % daily_recs == 0
    for i in range(gf.messages // (daily_recs)):
        gf.seek(i*daily_recs)
        daily_recs = [x.values for x in gf.read(daily_recs)]
        rec_array = np.stack([
            daily_recs[j*len(rec_labels):(j+1)*len(rec_labels)]
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
    for i in range(gf.messages // (daily_recs)):
        gf.seek(i*daily_recs)
        daily_recs = [x.values for x in gf.read(daily_recs)]
        rec_array = np.stack([
            daily_recs[j*len(rec_labels):(j+1)*len(rec_labels)]
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
        daily_recs = [x.values for x in gf.read(daily_recs)]
        rec_array = np.stack([
            daily_recs[j*len(rec_labels):(j+1)*len(rec_labels)]
            for j in range(24)
            ], axis=0)
        rec_array = np.transpose(rec_array, (0,2,3,1))
        yield (rec_labels,rec_array)

def _gen_era5_wbgt_daily(file_path:Path):
    """ """
    d = nc.Dataset(file_path, "r")
    labels = ["d2m","t2m","ssrd","u10","v10","sp"]
    num_times = d.variables["valid_time"].size
    daily_recs = len(labels)*24
    assert num_times % 24 == 0
    for i in range(num_times // 24):
        tmp_array = []
        for l in labels:
            tmp_array.append(d.variables[l][24*i:24*(i+1)])
        tmp_array = np.stack(tmp_array, axis=-1)
        yield (labels, tmp_array)

if __name__=="__main__":
    ## Directories should contain only files that should be loaded to the hdf5
    #data_dir = Path("data")
    wgrib_bin = "/nas/rhome/mdodson/.micromamba/envs/learn/bin/wgrib"
    data_dir = Path("data")
    static_pkl = data_dir.joinpath("static/era5_static.pkl")
    out_dir = data_dir.joinpath("timegrids/")

    (slabels,_),sdata = pkl.load(static_pkl.open("rb"))
    m_valid = sdata[slabels.index("landmask")] > .85

    extract_methods = {
            "rad":_gen_era5_rad_daily,
            "snveg":_gen_era5_snveg_daily,
            "soil":_gen_era5_soil_daily,
            "wbgt":_gen_era5_wbgt_daily,
            }

    extract_years = list(range(2012,2024))
    era5_dirs = [data_dir.joinpath(f"era5/{y}") for y in extract_years]
    extract_paths = {}
    for dpath in era5_dirs:
        for fp in dpath.iterdir():
            ts = datetime.strptime(fp.stem.split("_")[-1], "%Y%m")
            if ts.year not in extract_paths.keys():
                extract_paths[ts.year] = {}
            if ts.month not in extract_paths[ts.year].keys():
                extract_paths[ts.year][ts.month] = []
            extract_paths[ts.year][ts.month].append(fp)

def extract_era5_year(file_dict, out_h5_path, static_array, chunk_shape):
    H5F,D,S,T = None,None,None,None
    for mix in range(1,13):
        cur_h5_ix = 0
        ## declare generators for this month
        gens = [extract_methods[fp.stem.split("_")[1]](fp)
                for fp in file_dict[mix]]
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
                    arrays.append(tmp_array)
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
                        )
                S = H5F.create_dataset(
                        name="/data/static",
                        shape=static_array.shape,
                        maxshape=static_array.shape,
                        )
                T = H5F.create_dataset(
                        name="/data/time",
                        shape=(0,),
                        maxshape=(None,),
                        )
            cur_slice = slice(cur_h5_ix, cur_h5_ix+arrays.shape[0])
            D.resize((cur_slice.stop, *arrays.shape[1:]))
            D[cur_slice] = arrays
            T.resize((cur_slice.stop,))
            T[cur_slice] = tmp_times
            cur_h5_ix = cur_slice.stop
            H5F.flush()
