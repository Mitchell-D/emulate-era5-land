import netCDF4 as nc
import numpy as np
import pickle as pkl
from pathlib import Path

label_mapping = {
        "tvl":("vt-low", "low vegetation type"),
        "tvh":("vt-high", "high vegetation type"),
        "slt":("soilt", "soil type"),
        "cvl":("vc-low", "low vegetation cover"),
        "cvh":("vc-high", "high vegetation cover"),
        "dl":("laked", "lake depth"),
        "z":("geopot", "surface geopotential height"),
        "cl":("lakec", "lake cover"),
        "si10":("glacmask", "glacier mask"),
        "lsm":("landmask", "land-sea mask"),
        }

int_types = ["slt", "tvl", "tvh"]

if __name__=="__main__":
    static_data_dir = Path("/rhome/mdodson/emulate-era5-land/data/static")
    static_source_dir = static_data_dir.joinpath("source-files")

    ## must be exact
    lat_range = (50,24) ## north to south
    lon_range = (-125,-66.5) ## east to west

    sdata = []
    slabels = []
    base_labels = ["latitude", "longitude", "time"]
    got_coords = False
    for p in [p for p in static_source_dir.iterdir() if p.suffix==".nc"]:
        d = nc.Dataset(p, "r")
        if not got_coords:
            lat1d = d["latitude"][...]
            ## convert to (-180,180] longitude
            lon1d = np.where(
                    d["longitude"][...] <= 180,
                    d["longitude"][...],
                    d["longitude"][...]-360,
                    )
            lat_ix_min = np.argmin(np.abs(lat1d - lat_range[0]))
            lat_ix_max = np.argmin(np.abs(lat1d - lat_range[1]))
            lon_ix_min = np.argmin(np.abs(lon1d - lon_range[0]))
            lon_ix_max = np.argmin(np.abs(lon1d - lon_range[1]))
            sub_slice = (slice(lat_ix_min,lat_ix_max+1),
                    slice(lon_ix_min,lon_ix_max+1))
            print(sub_slice)

            coord_shape = (lat1d.size, lon1d.size)
            lat = np.stack([lat1d for i in range(lon1d.size)], axis=1)
            lon = np.stack([lon1d for i in range(lat1d.size)], axis=0)
            sdata.append(lat[*sub_slice])
            slabels.append(("lat", "latitude"))
            print(sdata[-1])
            sdata.append(lon[*sub_slice])
            slabels.append(("lon", "longitude"))
            print(sdata[-1])
            got_coords = True
        unq_label = [l for l in d.variables.keys() if l not in base_labels][0]
        tmp_data = d[unq_label][...][0] ## drop superfluous first dim
        if unq_label in int_types:
            tmp_data = np.round(tmp_data).astype(int)
        sdata.append(tmp_data[*sub_slice])
        slabels.append(label_mapping[unq_label])
        d.close()

    slabels,sdesc = zip(*slabels)
    #sdata = np.stack(sdata, axis=-1)
    for l,d,x in zip(slabels,sdesc,sdata):
        print(x.shape, f"{l:<10} {d:<20}")
    pkl_path = static_data_dir.joinpath("era5_static.pkl")
    pkl.dump(((slabels,sdesc),sdata), pkl_path.open("wb"))
