
import numpy as np
from pathlib import Path

from testbed import generators
#from era5_testbed import

if __name__=="__main__":

    proj_root_dir = Path("/rhome/mdodson/emulate-era5-land")

    tg_paths = [
            p for p in proj_root_dir.joinpath("data/timegrids").iterdir()
            if p.stem.split("_")[2] in map(str,range(2012,2018))
            ]

    #'''
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
    #'''
