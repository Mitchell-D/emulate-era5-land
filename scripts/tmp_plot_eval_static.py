import numpy as np
import pickle as pkl
from pathlib import Path

from emulate_era5_land.evaluators import Evaluator
from emulate_era5_land.plotting import plot_geo_scalar

if __name__=="__main__":
    pkl_path = Path("data/eval-new/" + \
            "eval_full_acclstm-era5-swm-64_EvalStatic_grid_err-mean.pkl")
    slabels,sdata = pkl.load(Path("data/static/era5_static.pkl").open("rb"))
    ev = Evaluator.from_pkl(pkl_path)
    res = ev.final_results()
    lat,lon = [sdata.index(l) for l in ("lat", "lon")]

    plot_geo_scalar(
            data=res["mean"],
            latitude=lat,
            longitude=lon,
            )
