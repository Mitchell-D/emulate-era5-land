
import torch
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime
import json

#from testbed import generators
from era5_testbed.helpers import get_permutation_pair

class TimegridSequenceDataset(torch.utils.data.IterableDataset):
    def __init__(self):
        pass

def _timegrid_spatial_permute(tg_path:Path, base_seed):
    """
    Permute the timegrid's static and dynamic arrays along the spatial axis
    and rechunk accordingly
    """
    F = h5py.File(tg_path, "r+")
    D = F["/data/dynamic"]
    S = F["/data/static"]
    T = list(map(lambda t:datetime.fromtimestamp(int(t)),F["/data/time"][...]))
    seed_sum = base_seed + int(tg_path.stem.split("_")[-1])
    forward,backward = get_permutation_pair(D.shape[1], seed_sum)
    tchunks,_,_ = zip(*D.iter_chunks())
    perms = np.stack([forward,backward], axis=-1)

    print((np.arange(forward.size)-17)[forward][backward])
    print((np.arange(forward.size)-17)[backward][forward])

    if "permute" not in F["data"].keys():
        prev_attrs = json.loads(F["data"].attrs["dynamic"])
        prev_attrs.update({"clabels":("time","space")})
        F["data"].attrs["dynamic"] = json.dumps(prev_attrs)
        prev_attrs = json.loads(F["data"].attrs["static"])
        prev_attrs.update({"clabels":("space",)})
        F["data"].attrs["static"] = json.dumps(prev_attrs)
        print(f"Updated dynamic and static attributes")

        pshape = perms.shape
        F.create_dataset("/data/permute", shape=pshape, maxshape=pshape)

    F["/data/permute"][...] = perms
    F["data"].attrs["permute"] = json.dumps({
        "clabels":("space",),
        "flabels":["forward", "backward"],
        "seed":seed_sum,
        })

    tmps = S[...][forward]
    for t in set(tchunks):
        print(f"Permuting {T[t][0]} - {T[t][-1]}")
        tmpd = D[t,...][:,forward]
        D[t,...] = tmpd
        #F.flush()
    F.close()

if __name__=="__main__":

    proj_root_dir = Path("/rhome/mdodson/emulate-era5-land")

    tg_paths = [
            p for p in proj_root_dir.joinpath("data/timegrids").iterdir()
            if p.stem.split("_")[2] in map(str,range(2012,2018))
            ]

    print(tg_paths[1])
    _timegrid_spatial_permute(tg_paths[1], 200007221750)
    tg_paths[1].rename(Path(
        "/rstor/mdodson/era5/timegrids-new"
        ).joinpath(tg_paths[0].name))

    '''
    check_labels = ["dlwrf","dswrf","shtfl","lhtfl","swnet","lwnet","tmp"]
    gen = generators.timegrid_sequence_dataset(
            timegrid_paths=tg_paths,
            window_size=24,
            horizon_size=24,
            #window_feats=[
            #    "weasd", "apcp", "tmp", "pres", "lai-high", "lai-low",
            #    "dswrf", "dlwrf", "ugrd", "vgrd",
            #    "vsm-07", "vsm-28", "vsm-100", "vsm-289",
            #    ],
            window_feats=check_labels,
            horizon_feats=[
                "weasd", "apcp", "tmp", "pres", "lai-high", "lai-low",
                "dswrf", "dlwrf", "ugrd", "vgrd",
                ],
            pred_feats=["vsm-07", "vsm-28", "vsm-100", "vsm-289"],
            static_feats=["vc-high", "vc-low"],
            static_int_feats=[("vt-low",3), ("vt-high",3), ("soilt",3)],
            static_conditions=[],
            derived_feats={
                "windmag":(("ugrd","vgrd"), tuple(),
                    "lambda d,s:(d[0]**2+d[1]**2)**(1/2)",),
                },
            num_procs=1,
            deterministic=False,
            block_size=16,
            buf_size_mb=128,
            samples_per_timegrid=16384,
            max_offset=24,
            sample_separation=19,
            include_init_state_in_predictors=True,
            load_full_grid=False,
            seed=200007221750,
            )

    count = 0
    for ((w,h,s,si,t),y) in gen.batch(64):
        print(w.shape, h.shape, s.shape, si.shape, t.shape, y.shape)
        print(w.numpy().size, np.count_nonzero(np.isfinite(w.numpy())))
        avgs = np.nanmean(w.numpy(), axis=(0,1))
        for l,v in zip(check_labels, list(avgs)):
            print(l,v)
        if count > 10:
            break
        count += 1
    '''

    ## permute timegrids along axis
