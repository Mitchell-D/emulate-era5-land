import json
from pathlib import Path
import numpy as np
from pprint import pprint

from emulate_era5_land.evaluators import Evaluator
from emulate_era5_land.plotting import plot_lines

def probability_threshold_explore(eval_pkl):
    ev = Evaluator.from_pkl(eval_pkl)
    print(f"Static axes shape: {ev.sshape}")
    fr = ev.final_results()
    m_valid = fr["count"] > 2
    C = fr["count"][m_valid]
    P = C / np.sum(C)

    print(f"\nSamples:{np.sum(C)}, Valid Combos:{C.shape[0]}")

    print(f"\nStats on count per valid combination:")
    print(f"min:{np.amin(C)} mean:{np.average(C):.5f} " \
            + f"median:{np.median(C)} max:{np.amax(C)}")

    print(f"\nStats on probability per valid combination:")
    print(f"min:{np.amin(P):.5f} mean:{np.average(P):.5f} " \
            + f"median:{np.median(P):.5f} max:{np.amax(P):.5f}")

    saup = float(input(f"\nSelect all samples under probability: "))

    ## set a lower bound on probability of selection
    m_saup = P < saup
    minc = np.min(C[~m_saup])
    ## establish a new count with the minimum value from the threshold
    D = np.where(m_saup, minc, C)
    Q = D / np.sum(D)

    ## ratio of samples to keep before probability threshold
    drop_prob_p = np.min(P) / P
    ## ratio of samples to keep after probability threshold
    drop_prob_q = np.min(Q) / Q

    print(f"After applying minimum probability threshold {saup}...")

    print(f"\nStats on drop probability per combo before threshold")
    print(f"min:{np.amin(drop_prob_p):.5f} " + \
            f"mean:{np.average(drop_prob_p):.5f} " + \
            f"median:{np.median(drop_prob_p):.5f} " + \
            f"max:{np.amax(drop_prob_p):.5f}")

    print(f"\nStats on drop probability per combo after threshold {saup}")
    print(f"min:{np.amin(drop_prob_q):.5f} " + \
            f"mean:{np.average(drop_prob_q):.5f} " + \
            f"median:{np.median(drop_prob_p):.5f} " + \
            f"max:{np.amax(drop_prob_q):.5f}")

def plot_probability_thresholds(
        eval_pkl, pmin=1e-6,pmax=.1, nlines=16, nbins=32, logspace=False,
        plot_spec={},fig_path=None, show=False,):
    ev = Evaluator.from_pkl(eval_pkl)
    print(f"Static axes shape: {ev.sshape}")
    fr = ev.final_results()
    m_valid = fr["count"] > 2
    C = fr["count"][m_valid]
    P = C / np.sum(C)
    f_hist = np.vectorize(lambda X:np.floor(np.clip(X*nbins, 0, nbins-1)))
    dpp = np.min(P) / P
    dpq = []
    dphists = []
    thresholds = []
    range_func = [np.linspace,np.geomspace][logspace]
    for s in range_func(pmin,pmax,nlines):
        m_s = P<s ## combos with prob under threshold
        if not np.any(m_s): ## ignore thresholds that don't affect the results
            continue
        if np.all(m_s): ## ignore thresholds that capture everything
            continue
        thresholds.append(s)
        d = np.where(m_s, np.min(C[~m_s]), C)
        q = d / np.sum(d)
        dpq.append(1 - np.min(q) / q)
        values,counts = np.unique_counts(f_hist(dpq[-1]).reshape(-1))
        dphists.append((values/nbins,counts))

    domain,ylines = zip(*dphists)

    plot_lines(
        domain=domain, ## probability of dropping
        ylines=ylines, ## probability of dropping
        fig_path=fig_path,
        show=show,
        labels=[f"{t:.3E}" for t in thresholds],
        plot_spec=plot_spec,
        multi_domain=True,
        )

if __name__=="__main__":
    proj_root = Path("/Users/mtdodson/desktop/projects/emulate-era5-land")
    eval_pkl = proj_root.joinpath(
        "data/eval-new/eval_full_acclstm-era5-swm-64_EvalStatic_" + \
        "veg-soil-combos_err-mean.pkl"
        )
    list_era5 = json.load(proj_root.joinpath(
        "data/list_feats_era5.json").open("r"))

    #probability_threshold_explore(eval_pkl)

    ev = Evaluator.from_pkl(eval_pkl)
    pprint(ev.params)

    ## 1d line plots of class size & covariate error
    #'''
    ev.plot(
        plot_type="hist-1d",
        static_feats=[("auxs", "vt-high")],
        data_feats=[("err-abs", "swm-7"), ("err-abs", "swm-28")],
        data_metrics=["mean", "stddev"],
        domain_labels=list_era5["static-classes"]["vt-high"],
        plot_params={
            "line_colors":["black", "green", "blue"],
            "line_style":["-","-","-."],
            },
        plot_spec={
            "spine_increment":.07,
            "ylabel_position":"top",
            "ytick_rotation":60,
            },
        show=False,
        fig_path=Path("/Users/mtdodson/desktop/tmp-image/eval-new/tmp-0.png"),
        )
    ev.plot(
        plot_type="hist-1d",
        static_feats=[("auxs", "soilt")],
        data_feats=[("err-abs", "swm-7"), ("err-abs", "swm-28")],
        data_metrics=["mean", "stddev"],
        domain_labels=list_era5["static-classes"]["soilt"],
        plot_params={
            "line_colors":["black", "green", "blue"],
            "line_style":["-", "-","-."],
            },
        plot_spec={
            "spine_increment":.07,
            "ylabel_position":"top",
            },
        show=False,
        fig_path=Path("/Users/mtdodson/desktop/tmp-image/eval-new/tmp-1.png"),
        )
    ev.plot(
        plot_type="hist-1d",
        static_feats=[("auxs", "vt-low")],
        data_feats=[("err-abs", "swm-7"), ("err-abs", "swm-28")],
        data_metrics=["mean", "stddev"],
        domain_labels=list_era5["static-classes"]["vt-low"],
        plot_params={
            "line_colors":["black", "green", "blue"],
            "line_style":["-", "-","-."],
            },
        plot_spec={
            "spine_increment":.07,
            "ylabel_position":"top",
            },
        show=False,
        fig_path=Path("/Users/mtdodson/desktop/tmp-image/eval-new/tmp-2.png"),
        )
    #'''

    ## 3d point clouds
    #'''
    ev.plot(
        plot_type="points-3d",
        domain_labels=list_era5["static-classes"],
        data_feats=[("err-abs", "swm-7")],
        data_metrics="mean",
        plot_spec={
            "size_scale":.035,
            #"size_scale":100,
            "cmap":"jet",
            "cb_shrink":.8,
            "vmin":0.,
            "vmax":.020,
            "alpha":.7,
            "xtick_rotation":45,
            "ytick_rotation":45,
            "ztick_rotation":45,
            },
        show=True,
        )

    ev.plot(
        plot_type="points-3d",
        domain_labels=list_era5["static-classes"],
        data_feats=[("err-bias", "swm-7")],
        data_metrics="mean",
        plot_spec={
            "size_scale":.035,
            #"size_scale":100,
            "vmin":-.015,
            "vmax":.015,
            "cmap":"bwr_r",
            "cb_shrink":.8,
            "alpha":.7,
            "xtick_rotation":45,
            "ytick_rotation":45,
            "ztick_rotation":45,
            },
        show=True,
        )
    #'''

    #'''
    plot_probability_thresholds(
            eval_pkl,
            pmin=1e-6,
            pmax=.1,
            logspace=True,
            nlines=32,
            nbins=24,
            plot_spec={
                "title":"probability of dropping samples per threshold",
                "xlabel":"probability of dropping samples",
                "ylabel":"num of combos with drop rate",
                "legend_ncols":3,
                "cmap":"gnuplot",
                "line_width":3,
                #"yscale":"log",
                },
            fig_path=None,
            show=True,
            )
    #'''

