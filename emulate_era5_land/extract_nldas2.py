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

def _extract_nldas2_files(nldas_path, noahlsm_path, nldas_nfeats,
        noahlsm_nfeats, crop_y=(0,0), crop_x=(0,0), m_valid=None):
    assert nldas_path.exists(),nldas_path
    assert noahlsm_path.exists(),noahlsm_path
    grb_nldas = pygrib.open(nldas_path.as_posix())
    assert grb_nldas.messages == len(nldas_nfeats)
    grb_noahlsm = pygrib.open(noahlsm_path.as_posix())
    assert grb_noahlsm.messages == len(noahlsm_nfeats)
    grb_nldas.seek(0)
    feats = [x.values for x in grb_nldas.read(nldas_nfeats)]
    grb_noahlsm.seek(0)
    feats += [x.values for x in grb_noahlsm.read(noahlsm_nfeats)]
    feats = np.stack(feats, axis=-1)
    crop_y0,crop_yf = crop_y
    crop_x0,crop_xf = crop_x
    ## Make a spatial slice tuple for sub-gridding dynamic and static data
    crop_slice = (slice(crop_y0,feats.shape[0]-crop_yf),
            slice(crop_x0,feats.shape[1]-crop_xf))
    return feats[*crop_slice][m_valid]

def _mp_extract_nldas2_files(args):
    return _extract_nldas2_files(**args)

def extract_nldas2_year(
        nldas_files, noahlsm_files, nldas_labels, noahlsm_labels, out_h5_path,
        etimes, static_labels, static_array, chunk_shape, permutation=None,
        crop_x=(0,0), crop_y=(0,0), m_valid=None, nworkers=1,
        write_chunk_size=32, file_dtype=np.float32, debug=False):
    """
    :@param nldas_files: Year of paths to chronological nldas forcing gribs.
    :@param noahlsm_files: Year of paths to chronological model output gribs.
    :@param nldas_labels: List of unique string labels corresponding to each
        of the features stured hourly in the nldas files.
    :@param noahlsm_labels: List of unique string labels corresponding to each
        of the features stured hourly in the noahlsm files.
    :@param out_h5_path: Path to the full-year hdf5 file created by this method
    :@param etimes: List of epoch times corresponding to each file
    :@param static_labels: List of unique string labels for the final
        (feature) axis of static_array
    :@param static_array: array of time-invariant values to be stored alongside
        the dynamic and time coordinate arrays. This should be shaped like
        (Y,X,Fs) and not permuted in any way.
    :@param chunk_shape: 3-tuple of integers (T,P,F) describing the number of
        elements per chunk along each dynamic array axis.
    :@param time_start: Datetime representing the first time in the file
    :@param time_diff: Timedelta representing the time resolution of files.
    :@param permutation: 1D Integer array with the same size as number of valid
        pixels in m_valid. If provided, static and dynamic spatial axes will
        be permuted as such before storage.
    :@param m_valid: boolean array shaped like (Y,X) indicating which spatial
        points are considered valid. The number of True values in m_valid must
        be equal to the number of pixels in the permutation
    """
    ## generators expect args like (cur_month_file, prev_month_file)
    conversions = {
            ## Originally J/m^2/hr ; J/m^2/hr * hr/sec -> W/m^2
            "sshf":lambda f:f/3600, ## convert to W/m^2
            "slhf":lambda f:f/3600,
            "ssrd":lambda f:f/3600,
            "strd":lambda f:f/3600,
            "ssr":lambda f:f/3600,
            "str":lambda f:f/3600,

            ## originally m equivalent; m * kg/m^3 -> kg/m^2
            "e":lambda w:w*1000,
            "var251":lambda w:w*1000, ## potential evaporatin
            "tp":lambda w:np.clip(w*1000,0,None),

            "src":lambda w:np.clip(w*1000,0,None), ## src not an accumulation
            "ssro":lambda w:w*1000,
            "sro":lambda w:w*1000,
            "ro":lambda w:w*1000,
            "es":lambda w:w*1000,

            "evatc":lambda w:w*1000,
            "evabs":lambda w:w*1000,
            "evaow":lambda w:w*1000,
            "evavt":lambda w:w*1000,

            #"smlt":lambda w:w*1000,
            #"sf":lambda w:w*1000,

            ## originally m^3/m^3; m^3/m^3 * kg/m^3 * dz -> kg/m^2
            "swvl1":lambda w:w*1000*.07,
            "swvl2":lambda w:w*1000*.21,
            "swvl3":lambda w:w*1000*.72,
            "swvl4":lambda w:w*1000*1.89,
            }

    ## _extract_nldas2_files combines the arrays with nldas values first
    flabels = nldas_labels + noahlsm_labels
    ars = [{
        "nldas_path":pf, "noahlsm_path":pm, "crop_y":crop_y, "crop_x":crop_x,
        "nldas_nfeats":len(nldas_labels), "noahlsm_nfeats":len(noahlsm_labels),
        "m_valid":m_valid,
        } for pf,pm in zip(nldas_files,noahlsm_files)]
    H5F = None
    cur_h5_ix = 0
    with Pool(nworkers) as pool:
        cur_buf = []
        for i,darr in enumerate(1,pool.imap(_mp_extract_nldas2_files, args)):
            #'''
            if H5F is None:
                assert not out_h5_path.exists(), out_h5_path.name
                H5F = h5py.File(out_h5_path, "w")
                D = H5F.create_dataset(
                        name="/data/dynamic",
                        shape=(0, *darr.shape[1:]),
                        maxshape=(None, *darr.shape[1:]),
                        chunks=chunk_shape,
                        compression="gzip",
                        dtype="f4",
                        )
                if permutation is None:
                    print(f"WARNING: no spatial permutation provided")
                    permutation = np.arange(static_array.shape[0])
                L = H5F.create_dataset(
                        name="/data/latlon",
                        shape=(*static_array.shape[:2],2),
                        maxshape=(*static_array.shape[:2],2),
                        dtype="f8",
                        )
                L[...,0] = static_array[...,static_labels.index("lat")]
                L[...,1] = static_array[...,static_labels.index("lon")]
                ## allow for arbitrary extension along feature axis
                static_array = static_array[m_valid]
                S = H5F.create_dataset(
                        name="/data/static",
                        shape=static_array.shape,
                        maxshape=(*static_array.shape[:-1],None),
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
                    "flabels":flabels,
                    })
                H5F["data"].attrs["latlon"] = json.dumps({
                    "clabels":("lat","lon"),
                    "flabels":("lat","lon"),
                    })
                H5F["data"].attrs["static"] = json.dumps({
                    "clabels":("space",),
                    "flabels":static_labels,
                    })
                H5F["data"].attrs["mask"] = json.dumps({
                    "clabels":("lat","lon"),
                    "flabels":tuple(),
                    })
                H5F["data"].attrs["time"] = json.dumps({
                    "clabels":("time",),
                    "flabels":tuple(),
                    })
                H5F["data"].attrs["permutation"] = json.dumps({
                    "clabels":("space",),
                    "flabels":("fwd","inv"),
                    })
                #'''
            cur_buf.append(darr)
            if i%write_chunk_size:
                continue
            cur_slice = slice(cur_h5_ix, cur_h5_ix+darr.shape[0])
            D.resize((cur_slice.stop, *darr.shape[1:]))
            D[cur_slice] = darr.astype(file_dtype)[:,permutation]
            T.resize((cur_slice.stop,))
            T[cur_slice] = etimes[cur_slice]
            cur_h5_ix = cur_slice.stop
            if debug:
                tmpt = datetime.fromtimestamp(int(etimes[cur_slice.start]),
                        tz=timezone.utc)
                print(f"Extracted {arrays.shape} at {tmpt}; now {D.shape}")
            H5F.flush()
    return out_h5_path

if __name__=="__main__":
    ## Directories should contain only files that should be loaded to the hdf5
    data_dir = Path("data")
    #out_dir = data_dir.joinpath("timegrids")
    out_dir = Path("/rstor/mdodson/era5/timegrids-new/")
    static_pkl = data_dir.joinpath("static/era5_static.pkl")
    perm_pkl = data_dir.joinpath(
            "permutations/permutation_nldas2_conv_020.pkl")
    nldas2_dir = data_dir.joinpath("nldas2")
    noahlsm_dir = data_dir.joinpath("noahlsm")

    year = 2012
    workers = 12
    base_h5_path = f"timegrid_nldas2_{year}.h5"

    ## load the static data and boolean valid mask
    slabels,sdata = pkl.load(static_pkl.open("rb"))
    m_valid = sdata[slabels.index("m_valid")].astype(bool)

    ## load label conversions
    label_mapping = json.load(
            data_dir.joinpath("list_feats_era5.json").open("r")
            )["label-mapping"]
    ## load the desired spatial permutation
    _,perm,_ = pkl.load(perm_pkl.open("rb"))

    nldas_times,nldas_files = sorted([
        [datetime.strptime(f.stem.split("_")[-1], "H.A%Y%m%d.%H00.002.grb"
            ).astimezone(timezone.utc).timestamp(), f]
        for f in nldas2_dir.joinpath(str(year)).iterdir()
        ], key=lambda t:t[0])

    noahlsm_times,noahlsm_files = sorted([
        [datetime.strptime(f.stem.split("_")[-1], "H.A%Y%m%d.%H00.002.grb"
            ).astimezone(timezone.utc).timestamp(), f]
        for f in noahlsm_dir.joinpath(str(year)).iterdir()
        ], key=lambda t:t[0])

    assert np.all(np.isclose(nldas_times, noahlsm_times, rtol=1e-2))

    extract_nldas2_year(
        nldas_files=nldas_files,
        noahlsm_files=noahlsm_files,
        nldas_labels=[
            "tmp","spfh","pres","ugrd","vgrd","dlwrf","ncrain",
            "cape","pevap","apcp","dswrf"],
        noahlsm_labels=[
            "swnet","lwnet", ## net sw/lw flux at the surface
            "lhtfl","shtfl","gflux","snohf", ## general heat fluxes
            "dswrf","dlwrf", ## total sw/lw radiative fluxes
            "asnow","arain", ## solid/liquid precip partitions
            "evp", "roff-sf","roff-bg","snom", ## water-removing processes
            "tskin","alb", ## Average surface skin temp, albedo
            "wm-snow","wm-cnpy", ## Mass of water in snow/canopy
            "tsoil-10","tsoil-40","tsoil-100","tsoil-200", ## soil temp
            "swm-fc","swm-rz1","swm-rz2", ## full-col/root zone water mass
            "swm-10","swm-40","swm-100","swm-200", ## total layer water mass
            "lswm-10","lswm-40","lswm-100","lswm-200", ## liquid water mass
            "mstav-fc","mstav-rz", ## moisture availability
            "evp-cnpy","evp-trsp","evp-bare","evp-snow",
            "pevap", ## potential latent heat flux
            "acond", ## surface aerodynamic conductance (m/s)
            "snod","snowc", ## snow depth and fractional cover
            "ccond","rc-s","rc-t","rc-q","rc-m", ## canopy conductance + params
            "lai","gvf","smacf", ## vegetation parameters, availability control
            ],
        nldas_ignore=["ncrain","cape","pevap"],
        noahlsm_ignore=["dlwrf","dswrf"]
        out_h5_path=out_dir.joinpath(f"timegrid_nldas2_{year}.h5"),
        etimes=nldas_times,
        static_labels=slabels,
        static_array=np.stack(sdata, axis=-1),
        chunk_shape=(192,64,58),
        permutation=per[:,0],
        crop_x=(0,0),
        crop_y=(0,0),
        m_valid=None,
        nworkers=1,
        write_chunk_size=32,
        file_dtype=np.float32,
        debug=False
        )
    '''
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
    '''

    with Pool(workers) as pool:
        for result in pool.imap_unordered(mp_extract_era5_year, args):
            print(f"Generated {result}")
