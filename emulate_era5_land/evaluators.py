from abc import ABC,abstractmethod
from copy import deepcopy
import numpy as np
import pickle as pkl
from datetime import datetime,timezone
from typing import Callable
from pathlib import Path

from emulate_era5_land.plotting import plot_joint_hist_and_cov
from emulate_era5_land.plotting import plot_point_cloud_3d
from emulate_era5_land.plotting import plot_lines_multiy

REDUCE_FUNCS = {"min":np.amin, "mean":np.average, "max":np.amax, "sum":np.sum}

def get_epoch_to_index_func(include_year=False):
    """
    Get a vectorizable function that converts an epoch time to integer values
    that can be used as an index, accounting for rounding errors.

    When include_year is False, the function signature is "()->(2)" and the
    subsequent vector is the [day_of_year, time_of_day] as indeces starting
    with zero.

    When include_year is True, the function signature is "()->(3)" and the
    vector is the [year, day_of_year, ]
    """
    def _epoch_to_tod_doy_index(epoch_time):
        """
        Convert an epoch time to an index for the time of day and day of year
        """
        t = datetime.fromtimestamp(int(epoch_time), tz=timezone.utc)
        t = t.timetuple()
        ## round hour up if > 50 minutes
        tmp_tod = t.tm_hour + t.tm_min // 50
        ## cycle day if hour rounds up
        ix_tod = tmp_tod % 24
        ## convert day to index and increment if rounded to next day.
        tmp_doy = t.tm_yday - 1 + int(tmp_tod == 24)

        ## handle years rolling over due issues rounding to the hour
        tmp_year = t.tm_year
        ix_doy = tmp_doy % [366,365][bool(tmp_year % 4)]
        ix_year = tmp_year + int(ix_doy != tmp_doy)

        if include_year:
            return np.array([ix_year, ix_doy, ix_tod])
        else:
            return np.array([ix_doy, ix_tod])
    return _epoch_to_tod_doy_index


class Evaluator(ABC):
    """
    Base class for Evaluator objects, which must abide the following standards:

    1. All attributes that must persist through serialization are stored in one
       of the following dictionaries:
        _p : parameters provided upon initialization that determine the
             behavior of the evaluator, ie dataset/feature combos to analyze,
             prediction coarseness, absolute/bias error type, etc.
        _r : results collected through iteratively adding batches. The
             particular structure of this dictionary depends on the instance.
        _f : dict mapping string dataset names to lists of unique string names
             of the features they contain.

    2. Each batch must be represented as a dict mapping a dataset string name
       to an array or list of arrays containing the values for that batch's
       inputs, targets, outputs, auxiliary data values, etc. If an array, the
       feature axis is assumed to be the last one, which must have the same
       size as the number of labels in the dataset's corresponding _f list.
       Similarly, lists are assumed to contain one array for each feature,
       which may be of non-uniform shapes, as is the case for one-hot encoded
       static features.

    3. Subclasses must have a static attribute "required" that lists the string
       keys of parameter items that must be present for a viable Evaluator of
       that type.
    """
    def __init__(self, subtype:str, params:dict, feats:dict,
            results:dict=None, meta:dict={}):
        """
        Generalized initializer for Evaluator instances, which should handle
        pretty much all the functionality needed for initializing subclass
        instances. Defines the parameter, feature, and result dicts, and
        validates the parameters according to the subclass' required
        implementation of _validate_params.

        :@param subtype: String name of the subclass of Evaluator being
            initialized. This is needed in order to generally store and
            re-load serialized subclass instances.
        :@param params: Dict of parameters needed to define the behavior of
            an Evaluator subclass instance. Each of the keys contained in this
            dict should be provided in a 'required' staticmethod of subclasses.
        :@param feats: Dict mapping dataset strings to a list of the string
            labels naming features in the dataset array (or list of arrays).
        :@param results: Dict where result states accumulated per batch are
            stored. If this is a newly-initialized Evaluator, it's expected
            that results is None, but re-loaded Evaluators may have a populated
            results dict.
        :@param meta: Dict of ancillary information like model configurations
            which is not strictly required for Evaluator functionality
        """
        self._p = params ## parameter dict
        self._f = feats ## feature dict
        self._r = results ## result dict
        self._m = meta ## meta-info dict
        self._t = subtype ## subclass type
        self._validate_params()
        pass

    @classmethod
    def required(cls):
        """ List of string labels for required parameter arguments """
        try:
            return cls._required
        except AttributeError as ae:
            print("_required attribute containing a list of string " \
                    + f"parameter keys should be defined in {cls}")
            raise ae

    @abstractmethod
    def _validate_params(self):
        """ Verify that the required parameters exist for this eval type """
        pass

    @abstractmethod
    def add_batch(self, batch_dict:dict):
        """ Update the partial evaluation data with a new batch of samples """
        pass

    @abstractmethod
    def final_results(self):
        """
        Formalize the results dict by performing aggregations, normalizing
        data, etc, producing the most usable form of the Evaluator state.
        """
        pass

    @staticmethod
    def from_pkl(pkl_path:Path):
        """ Recover attribute dicts of an instance from a pkl """
        return Evaluator.from_tuple(pkl.load(pkl_path.open("rb")))

    @staticmethod
    def from_tuple(tup:tuple):
        t, p, f, r, m = tup
        return EVALUATORS[t](params=p, feats=f, results=r, meta=m)

    @property
    def params(self):
        return self._p
    @property
    def feats(self):
        return self._f
    @property
    def results(self):
        return self._r
    @property
    def meta(self):
        return self._m
    @property
    def evtype(self):
        return self._t

    def to_tuple(self):
        """ Dump the attribute dicts for this instance into a tuple """
        return (self._t, self._p, self._f, self._r, self._m)

    def to_pkl(self, pkl_path:Path):
        """ Dump the attribute dicts for this instance into a pkl """
        pkl.dump(self.to_tuple(), pkl_path.open("wb"))
        return pkl_path

    def get_farray(self, dataset:str, feat:str, batch_dict:dict):
        """
        Extract a single feature array from a batch dictionary given the
        array's dataset and feature string keys. Along the way, ensure that
        datasets have the right number of features.

        :@param dataset: string dataset name of feature to extract.
        :@param feat: string feat name of array within the dataset to return.
        :@param batch_dict: dict mapping dataset names to arrays or lists of
            arrays corresponding to data values for a particular batch.
        """
        assert dataset in self._f.keys(), \
            f"{dataset} not one of {list(self._f.keys())}"
        assert dataset in batch_dict.keys(), \
            f"{dataset} not one of {list(batch_dict.keys())}"
        assert feat in self._f[dataset], \
            f"{feat} not one of {self._f[dataset]}"
        if isinstance(batch_dict[dataset], (list, tuple)):
            assert len(batch_dict[dataset])==len(self._f[dataset])
            return batch_dict[dataset][self._f[dataset].index(feat)]
        assert batch_dict[dataset].shape[-1]==len(self._f[dataset]), \
            f"{batch_dict[dataset].shape = } ; {len(self._f[dataset]) = }"
        return batch_dict[dataset][...,self._f[dataset].index(feat)]

class EvalStatic(Evaluator):
    """
    Collects statistics with respect to combinations of static parameters.
    On init, the user must provide a list of 1d arrays static_values
    corresponding to each independent static parameter (each array containing
    all valid values of that parameter), and a equal-sized list of 2-tuples
    static_feats like (dataset:str, feat:str) which resolve to a single number
    per batch sample during evaluation.

    During evaluation, the static values of each sample in the batch are
    checked for equality with each of the static coordinate axes.
    Valid samples' data_feats are extracted, then reduced by reduce_func
    before being incorporated into the results data. For each static value
    combination, results recorded include:

    global maximum, global minimum, mean, variance, count

    Fundamentally, this is very similar to EvalJointHist in the sense that
    creating a derived feat that returns a static value would yield the same
    thing, but the use cases favor distinct bin indexing methods, so it is
    worth keeping them separate.
    """
    _required = ["static_feats", "static_values", "data_feats", "reduce_func",
            "collect_mean_var", "collect_min_max"]
    def __init__(self, params:dict, feats:dict,
            results:dict=None, meta:dict={}):
        """ See superclass initializer. """
        super(EvalStatic, self).__init__(
                "EvalStatic",
                params=params,
                feats=feats,
                results=results,
                meta=meta,
                )

    def _validate_params(self):
        """ """
        ## set defaults for arguments that aren't really required
        if "reduce_func" not in self._p.keys():
            self._p["reduce_func"] = "mean"
        else:
            assert self._p["reduce_func"] in REDUCE_FUNCS, \
                    f"{self._p['reduce_func']} NOT FOUND in:\n{REDUCE_FUNCS=}"

        assert len(self._p["static_feats"])==len(self._p["static_values"])
        self.scoords = []
        ## Make sure all the static features are valid
        for (dk,fk),v in zip(self._p["static_feats"],self._p["static_values"]):
            assert dk in self._f.keys(), f"{dk} not in {list(self._f.keys())}"
            assert fk in self._f[dk], f"{fk} not in dataset {dk} {self._f[dk]}"
            v = np.array(v)
            assert np.all(np.unique(v) == np.sort(v))
            assert v.ndim==1,v.shape
            self.scoords.append(v)
        self.sshape = tuple(s.size for s in self.scoords)

        ## Make sure all the data features are valid
        for dk,fk in self._p["data_feats"]:
            assert dk in self._f.keys(), f"{dk} not in {list(self._f.keys())}"
            assert fk in self._f[dk], f"{fk} not in dataset {dk} {self._f[dk]}"

    def add_batch(self, bdict:dict, dtype=np.float32):
        """
        1. extract static features and check each sample's inclusion in each
           static coordinate axis. keep & record index if it is in all.
        2. extract the data feats and reduce remaining samples according to
           the provided reduce function
        3. update the results array at each sample's static coordinate index.
        """
        ## extract static features
        sfeats = [self.get_farray(*sf, bdict)
            for sf in self._p["static_feats"]]
        nb = sfeats[0].shape[0]
        nsf = len(self._p["static_feats"])
        ndf = len(self._p["data_feats"])

        if self._r is None:
            ## float64 for now just since it must be used as denominator
            self._r = {"count":np.full(self.sshape, 0, dtype=np.float64)}
            ## float32 for everything else
            res_params = ((*self.sshape,ndf), np.nan, np.float32)
            if self._p["collect_min_max"]:
                self._r["min"] = np.full(*res_params)
                self._r["max"] = np.full(*res_params)
            if self._p["collect_mean_var"]:
                self._r["mean"] = np.full(*res_params)
                self._r["m2"] = np.full(*res_params)

        ## determine which samples are valid, and their static indeces
        sixs = np.full((nb, nsf), -1, dtype=int)
        m_valid = np.full(nb, True)
        for i in range(nsf):
            y,x = np.meshgrid(sfeats[i], self.scoords[i], indexing="ij")
            matches = (x==y)
            m_valid = m_valid & np.any(matches, axis=1)
            for j in range(nb):
                if not m_valid[j]:
                    continue
                sixs[j,i] = np.argwhere(matches[j])

        ## update number of samples in batch after m_valid applied
        nb = np.count_nonzero(m_valid)
        ## just return if no valid samples in this batch
        if nb==0:
            return self
        sixs = sixs[m_valid]

        ## extract valid data feats and stack to (B,S,F)
        dfeats = np.stack([
            self.get_farray(*df, bdict)[m_valid]
            for df in self._p["data_feats"]
            ], axis=-1)

        ## reduce the data feats to all but the batch and feature axes
        red_axes = tuple(set(range(dfeats.ndim))-{0,dfeats.ndim-1})
        dfeats = REDUCE_FUNCS[self._p["reduce_func"]](dfeats, axis=red_axes)

        ## update the results dictionary
        for bix in range(nb):
            six = sixs[bix]
            self._r["count"][*six] += 1
            ## on first iteration, set defaults
            if self._r["count"][*six]==1:
                if self._p["collect_mean_var"]:
                    self._r["mean"][*six] = dfeats[bix]
                    self._r["m2"][*six] = 0.
                if self._p["collect_min_max"]:
                    self._r["min"][*six] = dfeats[bix]
                    self._r["max"][*six] = dfeats[bix]
            elif self._p["collect_min_max"]:
                m_min = self._r["min"][*six] > dfeats[bix]
                if np.any(m_min):
                    self._r["min"][*six][m_min] = dfeats[bix][m_min]
                m_max = self._r["max"][*six] < dfeats[bix]
                if np.any(m_max):
                    self._r["max"][*six][m_max] = dfeats[bix][m_max]
            ## welford's algorithm
            if self._p["collect_mean_var"]:
                d_1 = dfeats[bix] - self._r["mean"][*six]
                self._r["mean"][*six] += d_1 / self._r["count"][*six]
                d_2 = dfeats[bix] - self._r["mean"][*six]
                self._r["m2"][*six] += d_1 * d_2 ## sum of squares of diffs
        return self

    def final_results(self):
        """ """
        ## can't calculate variance of only one value
        results = {}
        m_valid = (self._r["count"] > 0)
        c = np.where(m_valid, self._r["count"], np.nan)
        m_valid = m_valid[...,None]
        if self._p["collect_min_max"]:
            results["min"] = np.where(m_valid, self._r["min"], np.nan)
            results["max"] = np.where(m_valid, self._r["max"], np.nan)
        if self._p["collect_mean_var"]:
            results["mean"] = np.where(m_valid, self._r["mean"], np.nan)
            results["m2"] = np.where(m_valid, self._r["m2"], np.nan)
            var = results["m2"] / c[...,None]
            var[c==1] = 0 ## one sample has zero/undefined variance
            results["var"] = var
        results["count"] = np.where(m_valid[...,0], c, 0).astype(int)
        return results

    def reduce(self, axes, keepdims=False):
        """
        Reduce the welford results along the provided axes, and return a
        results dict with the reduced count, mean, and m2 values
        """
        rd = deepcopy(self.final_results())
        count = rd["count"][...,None]
        count_rdc = np.sum(count, axis=axes, keepdims=True, dtype=float)
        count_rdc[count_rdc==0] = np.nan
        m_valid = count!=0
        m_valid_rdc = np.any(m_valid, axis=axes, keepdims=True)
        if self._p["collect_mean_var"]:
            assert all(k in rd.keys() for k in ["mean", "m2"])
            assert all(v in range(0,count.ndim) for v in axes)
            mean =  np.where(m_valid, rd["mean"], 0)
            m2 = np.where(m_valid, rd["m2"], 0)
            mean_rdc = np.sum(count*mean, axis=axes, keepdims=True) / count_rdc
            ## m2 merged: sum of internal m2s + correction for different means
            ## See "Parallel Algorithm" based on (Chan et al., 1979)
            ## https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
            mean_diff_sq = (mean - mean_rdc) ** 2
            m2_rdc = np.sum(m2, axis=axes, keepdims=True) \
                    + np.sum(count * mean_diff_sq, axis=axes, keepdims=True)
            var_rdc = m2_rdc / count_rdc
            rd["mean"] = np.where(m_valid_rdc, mean_rdc, np.nan)
            rd["m2"] = np.where(m_valid_rdc, m2_rdc, np.nan)
            rd["var"] = np.where(m_valid_rdc, var_rdc, np.nan)
            if not keepdims:
                rd["mean"] = np.squeeze(rd["mean"], axis=axes)
                rd["m2"] = np.squeeze(rd["m2"], axis=axes)
                rd["var"] = np.squeeze(rd["var"], axis=axes)

        if self._p["collect_min_max"]:
            assert all(k in rd.keys() for k in ["max", "min"])
            assert all(v in range(0,count.ndim) for v in axes)
            rd["min"] = np.nanmin(rd["min"], axis=axes, keepdims=True)
            rd["max"] = np.nanmax(rd["max"], axis=axes, keepdims=True)
            if not keepdims:
                rd["min"] = np.squeeze(rd["min"], axis=axes)
                rd["max"] = np.squeeze(rd["max"], axis=axes)

        rd["count"] = np.where(m_valid_rdc, count_rdc, 0).astype(int)
        if not keepdims:
            rd["count"] = np.squeeze(rd["count"], axis=axes)[...,0]
        return rd


    def plot(self, plot_type:str="hist-1d", static_feats=None, data_feats=[],
             data_metrics=None, domain_labels:dict={}, fig_path=None,
             plot_spec={}, plot_params={}, show=False,):
        """
        """
        valid_data_metrics = ["mean","var","stddev","max","min"]
        ps = deepcopy(plot_spec)
        ## Make sure all the features are valid in the label dict
        if not static_feats is None:
            assert all(f in self._p["static_feats"] for f in static_feats), \
                f"\nTrying to generate: {fig_path.name}" + \
                f"\nProvided axis feats: {static_feats}" + \
                f"\nValid axis feats:\n{list(self._p['static_feats'])} "
        if not data_feats == []:
            assert all(f in self._p["data_feats"] for f in data_feats), \
                f"\nTrying to generate: {fig_path.name}" + \
                f"\nProvided cov feats:\n{data_feats}" + \
                f"\nValid cov feats:\n{list(self._p['data_feats'])} "

        if plot_type=="hist-1d":
            if static_feats is None:
                sf = self._p["static_feats"][0]
            else:
                assert len(static_feats)==1, \
                    f"hist-1d only supports one axis feature at a time."
                sf = static_feats[0]

            ix_ax = self._p["static_feats"].index(sf)
            all_axes = set(range(self._r["count"].ndim))
            sum_axes = tuple(all_axes-{ix_ax})

            fr = self.reduce(sum_axes)

            ## sort in order of increasing count
            perm_ixs,sorted_count = map(np.array,zip(*sorted(enumerate(
                fr["count"]), key=lambda v:v[1])))


            domain = np.arange(len(self.scoords[ix_ax]))
            line_labels = ["count"]
            y_labels= ["count"]
            line_colors = []
            line_styles = []
            plot_cov = not data_feats is None
            covs = []
            if plot_cov:
                if isinstance(data_metrics, (list, tuple)):
                    assert all(cm in valid_data_metrics for cm in data_metrics)
                elif isinstance(data_metrics, str):
                    assert data_metrics in valid_data_metrics, \
                            f"covariate metric must be specified\n" + \
                            f"{valid_data_metrics=}"
                    data_metrics = [data_metrics]
                else:
                    raise ValueError(f"{data_metric=} must be string or list")

                ## one line color per covariate feature
                if "line_colors" in plot_params.keys():
                    assert len(plot_params["line_colors"])==len(data_feats)+1,\
                        "count followed by each covariate feat must " + \
                        + "all have a unique color, not:" + \
                        + str(plot_params["line_colors"])
                    line_colors.append(plot_params["line_colors"][0])
                    for ixf in range(len(data_feats)):
                        for ixm in range(len(data_metrics)):
                            line_colors.append(
                                    plot_params["line_colors"][ixf+1])
                    del plot_params["line_colors"]
                    ps["line_colors"] = ps.get(
                            "line_colors", line_colors)

                ## one line style per covariate metric
                if "line_style" in plot_params.keys():
                    assert len(plot_params["line_style"]) \
                            ==len(data_metrics)+1,\
                        "count followed by each covariate metric must " \
                        + "each have a unique line style, not:" \
                        + str(plot_params["line_style"])
                    line_styles.append(plot_params["line_style"][0])
                    for ixf in range(len(data_feats)):
                        for ixm in range(len(data_metrics)):
                            line_styles.append(
                                    plot_params["line_style"][ixm+1])
                    del plot_params["line_style"]
                    ps["line_style"] = ps.get(
                            "line_style", line_styles)

                ## extract every metric for every covariate feature
                ix_covs = [self._p["data_feats"].index(cf)
                           for cf in data_feats]
                for ixc,cf in zip(ix_covs,data_feats):
                    covs.append([])
                    y_labels.append(" ".join(cf))
                    for cm in data_metrics:
                        if cm=="stddev":
                            tmp_cov = fr["var"][...,ixc]**0.5
                        else:
                            assert cm in fr.keys(),f"{cm=} {list(fr.keys())=}"
                            tmp_cov = fr[cm][...,ixc]
                        line_labels.append(" ".join(cf)+f" {cm}")
                        ## apply the sorting permutation to the array
                        covs[-1].append(tmp_cov[perm_ixs])

            #print(perm_ixs)
            #print([self.scoords[ix_ax][pix] for pix in perm_ixs])
            xticks = [
                domain_labels.get(
                    str(self.scoords[ix_ax][pix]),
                    self.scoords[ix_ax][pix])
                for pix in perm_ixs]
            ps["title"] = ps.get("title", " ".join(sf))
            ps["xlabel"] = ps.get("xlabel", " ".join(sf))
            ps["xticks"] = ps.get("xticks", xticks)
            ps["y_labels"] = ps.get("y_labels", y_labels)
            ps["line_labels"] = ps.get("line_labels", line_labels)
            ## probably should go with bar graph
            plot_lines_multiy(
                domain=domain,
                ylines=[[sorted_count], *covs],
                plot_spec=ps,
                show=show,
                fig_path=fig_path,
                **plot_params,
                )

        ## plot
        elif plot_type=="size-cov":
            if isinstance(data_metrics, (list,tuple)):
                assert all(cm in valid_data_metrics for cm in data_metrics)
                assert len(data_metrics) == len(data_feats)
            raise ValueError(f"Incomplete plotting method")

        elif plot_type=="points-3d":
            if static_feats is None:
                assert len(self._p["static_feats"])>=3
                axfs = self._p["static_feats"][:3]
            else:
                assert len(static_feats)==3
                assert all(f in self._p["static_feats"] for f in static_feats)
                axfs = static_feats

            if isinstance(data_metrics, (list, tuple)):
                assert all(cm in valid_data_metrics for cm in data_metrics)
                assert len(data_metrics) == len(data_feats)
            elif isinstance(data_metrics, str):
                assert data_metrics in valid_data_metrics, \
                        f"covariate metric must be specified\n" + \
                        f"{valid_data_metrics=}"
                data_metrics = [data_metrics] * len(data_feats)
            elif data_metrics is None:
                if len(data_feats):
                    raise ValueError(
                        f"A data metric must be specified when using " + \
                        "data features to color or size a 3d point cloud")
                if "mean" in self._r.keys():
                    data_metrics = ("mean") * len(data_feats)
                    print(f"{data_metrics=}")
                elif "max" in self._r.keys():
                    data_metrics = ["max"] * len(data_feats)


            axixs = [self._p["static_feats"].index(f) for f in axfs]
            red_axes = tuple(set(range(self._r["count"].ndim))-set(axixs))
            fr = self.reduce(red_axes)

            axis_labels = [[
                domain_labels[axf[-1]].get(str(v),None)
                for v in self.scoords[ix_ax]
                ] for ix_ax,axf in zip(axixs,axfs)]

            cfix,sfix = None,None
            if len(data_feats)==0:
                ## use count for size by default
                sfeat = "count"
                sfix = None
                sdata = fr["count"]
                ## if there are some data features use the first one for color
                if len(self._p["data_feats"])>0:
                    cfeat = self._p["data_feats"][0]
                    cfix = 0
                    if data_metrics:
                        met = data_metrics[0]
                    elif "mean" in self._r.keys():
                        met = "mean"
                    elif "max" in self._r.keys():
                        met = "max"
                    cdata = fr[met][...,0]
                else:
                    cfeat,cfix,cdata = None,None,None
            ## otherwise default to color first argument, then size
            elif len(data_feats)==1:
                cfeat = data_feats[0]
                cfix = self._p["data_feats"].index(data_feats[0])
                cdata = fr[data_metrics[0]][...,cfix]
                sfeat = "count"
                sfix = None
                sdata = fr["count"]
            elif len(data_feats)==2:
                ## allow (None, data_feat) for only size
                if data_feats[0] is None:
                    cfeat,cfix,cdata = None,None,None
                else:
                    cfeat = data_feats[0]
                    cfix = self._p["data_feats"].index(data_feats[0])
                    cdata = fr[data_metrics[0]][...,cfix]
                sfeat = data_feats[1]
                sfix = self._p["data_feats"].index(data_feats[1])
                sdata = fr[data_metrics[1]][...,sfix]
            else:
                raise ValueError(
                    f"You may only provide 2 data feats (color)")

            ps["cb_label"] = ps.get("cb_label", " ".join(cfeat))

            m_valid = fr["count"] > 0
            ixs = np.stack(np.indices(fr["count"].shape), axis=-1)
            ixs = ixs[m_valid]
            sdata = None if sdata is None else sdata[m_valid]
            cdata = None if cdata is None else cdata[m_valid]
            plot_point_cloud_3d(
                coords=ixs,
                cvalues=cdata,
                svalues=sdata,
                axis_labels=axis_labels,
                plot_spec=ps,
                show=show,
                **plot_params,
                )


class EvalJointHist(Evaluator):
    """
    Develop a joint histogram(s) with an arbitrary number of axes having
    individual bounds and resolutions.

    Optionally provide one or more conditions that must evaluate to True per
    sample in order for inclusion in the corresponding hist.

    Also optionally provide one or more covariate features for which the mean
    (standard deviation?) are tracked individually for each histogram bin.

    Required parameters in `params` dict:

    :@param axis_feats: List of 2-tuples of strings (dataset, feat) indicating
        the data and order of each histogram axis. The datasets referenced by
        axis_feats must have the same shape.
    :@param axis_params: List of 3-tuples (min, max, resolution) corresponding
        to each of the histogram axes. Resolution must be an integer.
    :@param cov_feats: Optional list of 2-tuples of strings (dataset, feat)
        indicating the data and order of features to capture means and
        standard deviations for as covariates for each bin in of each
        histogram. The covariate feats must also have the same shape as the
        axis features so that there is a 1:1 correspondence between values.
        Default to None.
    :@param hist_conditions: Optional list of 2-tuples like (inputs, funcstr)
        where `inputs` is a list of 2-tuples (dataset, feat) indicating
        positional arguments to string-encoded lambda expression `funcstr`.
        Given the arguments, the expression should either return a 1d boolean
        array along the 1st (batch) axis, or a boolean array matching the
        shape of the axis and covariate features (excluding the feat axis).
        Default to None.
    :@param round_oob: If True, values that are out of bounds will be rounded
        to the nearest valid bin rather than discarded.
    """
    ## required parameters
    _required = ["axis_feats", "axis_params", "cov_feats", "hist_conditions",
            "round_oob"]
    _plot_types = ["hist-cov-2d"] ## supported plot names
    def __init__(self, params:dict, feats:dict,
            results:dict=None, meta:dict={}):
        """ See superclass initializer. """
        super(EvalJointHist, self).__init__(
                "EvalJointHist",
                params=params,
                feats=feats,
                results=results,
                meta=meta,
                )

    def _validate_params(self):
        """ Verify that the required parameters exist for this eval type """
        ## set defaults for arguments that aren't really required
        if "cov_feats" not in self._p.keys():
            self._p["cov_feats"] = None
        if "hist_conditions" not in self._p.keys():
            self._p["hist_conditions"] = None
        if "round_oob" not in self._p.keys():
            self._p["round_oob"] = True
        self._do_cf = bool(self._p["cov_feats"])
        self._do_hc = bool(self._p["hist_conditions"])

        ## Make sure all the features are valid in the label dic
        for dk,fk in self._p["axis_feats"]:
            assert dk in self._f.keys(), f"{dk} not in {list(self._f.keys())}"
            assert fk in self._f[dk], f"{fk} not in dataset {dk} {self._f[dk]}"

        self._hcoords = [] ## spanning coordinates of bins
        self._hdiffs = [] ## coordinate 'width' of each bin
        for i,(v0,vf,res) in enumerate(self._p["axis_params"]):
            assert v0<vf, "Lower bound {v0=} must be less than {vf=} for " + \
                    f"{self._p['axis_feats'][i]}"
            assert isinstance(res,int), f"{res=} must be an integer for " + \
                    f"{self._p['axis_feats'][i]}"
            self._hcoords.append(np.linspace(v0, vf, res+1))
            self._hdiffs.append((vf-v0)/res)

        ## validate covariate features
        if self._do_cf:
            for dk,fk in self._p["cov_feats"]:
                assert dk in self._f.keys(), \
                        f"{dk} not in {list(self._f.keys())}"
                assert fk in self._f[dk], \
                        f"{fk} not in dataset {dk} {self._f[dk]}"

        ## validate hist conditions and prepare the function objects
        self._hcs = []
        if self._do_hc:
            for args,func in self._p["hist_conditions"]:
                for dk,fk in args:
                    assert dk in self._f.keys(), \
                        f"{dk} not in {list(self._f.keys())}"
                    assert fk in self._f[dk], \
                        f"{fk} not in dataset {dk} {self._f[dk]}"
                self._hcs.append(args,eval(func))

        self._fshape = None

        assert isinstance(self._p["round_oob"], bool)

    def add_batch(self, bdict:dict):
        """
        Update the partial evaluation data with a new batch of samples

        1. If histogram arrays are not declared, create them.
        2. Extract data needed for and evaluate hist conditions, or else
           declare a uniformly True array of the appropriate shape.
        3. If not all condition bool arrays are uniformly False, extract the
           requested features for axis and covariate data.
        4. For each condition bool array...
           a. If 1d along sample axis, repeat along all other axes.
           b. If not rounding OOB values, "or" together boolean masks of out
              of bounds values along each axis, and negate them from the
              condition bool array.
           c. Apply the boolean array so all axis and covariate arrays in the
              batch are 1d and of identical size.
           d. Rescale the axis arrays into histogram indeces.
           e. Given the histogram arrays for this conditional bool array,
              ccumulate the sum of counts per each indexed bin as well as the
              sum of covariate values within each bin.
        """
        ## number of histograms
        nh = 1 if not self._do_hc else len(self._hcs)
        ncf = 0 if not self._do_cf else len(self._p["cov_feats"])
        dtype=np.float32
        ## lists of bounds and resolution per histogram axis.
        v0,vf,res = map(np.asarray, zip(*self._p["axis_params"]))
        if self._r is None:
            self._r = {
                ## ints (nb1,...,nbN) per hist
                "counts":[np.full(res, 0, dtype=int) for _ in range(nh)],
                ## float (nb1,...,nbN,Fc) per hist
                "cov_mean":None,
                ## float (nb1,...,nbN,Fc) per hist
                "cov_m2":None,
                }
            if self._do_cf:
                self._r["cov_mean"] = [
                        np.full((*res, ncf), 0, dtype=dtype)
                        for _ in range(nh)]
                self._r["cov_m2"] = [
                        np.full((*res, ncf), 0, dtype=dtype)
                        for _ in range(nh)]

        afeats = np.stack(
                [self.get_farray(*af, bdict) for af in self._p["axis_feats"]],
                axis=-1, dtype=dtype)
        cfeats = None if not self._do_cf else np.stack(
                [self.get_farray(*cf, bdict) for cf in self._p["cov_feats"]],
                axis=-1, dtype=dtype)

        if self._fshape is None:
            self._fshape = afeats.shape[:-1]

        if self._do_hc:
            masks =  []
            for args,func in self._hcs:
                mask = func(*[self.get_farray(*af, bdict) for af in args])
                assert mask.shape[0]==self._fshape[0]
                ## expand the mask to match the full feat array shape
                for aix in range(1,len(self._fshape)):
                    mask = np.stack(
                        [mask for _ in range(self._fshape[aix])],
                        axis=aix)
                masks.append(mask)
        else:
            masks = [np.full(self._fshape, True)]

        for i,m in enumerate(masks):
            aixs = np.floor(np.clip(res*(afeats[m]-v0)/(vf-v0), 0, res-1))
            aixs = aixs.astype(int)
            if self._do_cf:
                tmp_cf = cfeats[m]
            else:
                tmp_cf = None
            for j in range(aixs.shape[0]):
                self._r["counts"][i][*aixs[j]] += 1
                if self._do_cf:
                    if self._r["counts"][i][*aixs[j]] == 1:
                        self._r["cov_mean"][i][*aixs[j]] = tmp_cf[j]
                    d_1 = tmp_cf[j] - self._r["cov_mean"][i][*aixs[j]]
                    self._r["cov_mean"][i][*aixs[j]] += d_1 / \
                            self._r["counts"][i][*aixs[j]]
                    d_2 = tmp_cf[j] - self._r["cov_mean"][i][*aixs[j]]
                    self._r["cov_m2"][i][*aixs[j]] += d_1 * d_2
        return self

    def final_results(self):
        """ """
        ## can't calculate variance of only one value
        results = {"counts":[], "cov_mean":[], "cov_var":[],
            "hist_coords":self._hcoords, "hist_diffs":self._hdiffs}
        for i in range(len(self._r["counts"])):
            m_valid = (self._r["counts"][i] > 1)
            #c = self._r["counts"][i]
            #c[~m_valid] = np.nan
            c = np.where(m_valid, self._r["counts"][i], np.nan)
            results["counts"].append(c)
            if self._do_cf:
                m = np.where(m_valid[...,None], self._r["cov_mean"][i], np.nan)
                v = self._r["cov_m2"][i]/self._r["counts"][i][...,None]
                v = np.where(m_valid[...,None], v, np.nan)
                results["cov_mean"].append(m)
                results["cov_var"].append(v)
        return results

    def plot(self, plot_type:str="hist-2d", axis_feats=None, cov_feats=None,
            hist_idx=None, cov_metric=None, fig_path=None, plot_spec={},
            plot_params={}, show=False):
        """
        Plot EvalJointHist data.

        :@param plot_type: String representing the type of plot (ie the
            purpose of the feature/covariate/histogram chosen)
        :@param axis_feats: List of 2-tuple (dataset, feat) axis features to
            plot. The number of axis features needed depends on the plot type.
        :@param cov_feats: List of 2-tuple (dataset, feat) covariate features
            to plot. The number of features needed depends on the plot type,
        :@param hist_idx: the index of the histogram to use. Each index
            corresponds to a unique hist_conditions method used to select
            samples to include. If no hist_conditions were applied, keep the
            default value of zero.
        :@param plot_spec: plot_spec dict to supply to the plotting method
            for formatting
        :@param plot_params: Arguments passed to the
        """
        fr = self.final_results()
        #cov = None if not fr["cov_mean"] else fr["cov_mean"][0][...,0]

        ## Make sure all the features are valid in the label dict
        if not axis_feats is None:
            assert all(f in self._p["axis_feats"] for f in axis_feats), \
                f"\nTrying to generate: {fig_path.name}" + \
                f"\nProvided axis feats: {axis_feats}" + \
                f"\nValid axis feats:\n{list(self._p['axis_feats'])} "
        if not cov_feats is None:
            assert all(f in self._p["cov_feats"] for f in cov_feats), \
                f"\nTrying to generate: {fig_path.name}" + \
                f"\nProvided cov feats:\n{cov_feats}" + \
                f"\nValid cov feats:\n{list(self._p['cov_feats'])} "

        if plot_type == "hist-2d":
            if axis_feats is None:
                assert len(self._p["axis_feats"])>=2
                axis_feats = self._p["axis_feats"][:2]
            plot_cov = not cov_feats is None
            if plot_cov:
                assert cov_metric in ["cov_mean", "cov_var", "cov_stddev"], \
                        f"covariate metric must be specified"
            assert len(axis_feats)==2

            ## sum over all axes other than the selected ones.
            ix_ax1 = self._p["axis_feats"].index(axis_feats[0])
            ix_ax2 = self._p["axis_feats"].index(axis_feats[1])

            if hist_idx is None:
                hist_idx = 0

            sum_axes = set(range(fr["counts"][hist_idx].ndim))-{ix_ax1,ix_ax2}
            c = np.sum(fr["counts"][hist_idx], axis=tuple(sum_axes))

            cov = None
            if plot_cov:
                ix_cov = self._p["cov_feats"].index(cov_feats[0])
                if cov_metric=="cov_stddev":
                    var = np.where(
                            c>1,fr["cov_var"][hist_idx][...,ix_cov],np.nan)
                    cov = var**0.5
                else:
                    cov = np.where(
                            c>1,fr[cov_metric][hist_idx][...,ix_cov],np.nan)
            plot_joint_hist_and_cov(
                    counts=c,
                    ax1_params=self._p["axis_params"][ix_ax1],
                    ax2_params=self._p["axis_params"][ix_ax2],
                    fig_path=fig_path,
                    show=show,
                    **{"covariate":cov, "plot_covariate":True,
                       "separate_covariate_axes":True} if plot_cov else {},
                    plot_spec=plot_spec,
                    **plot_params,
                    )

class EvalSampleSources(Evaluator):
    """
    Keep track of when/where batch samples are drawn from
    """
    _required = ["vidx_feat", "hidx_feat", "time_feat", "cov_feats",
            "cov_reduce_metric", "cov_reduce_axes"]
    def __init__(self, params:dict, feats:dict,
            results:dict=None, meta:dict={}):
        """ See superclass initializer. """
        super(EvalSampleSources, self).__init__(
                "EvalSampleSources",
                params=params,
                feats=feats,
                results=results,
                meta=meta,
                )

    def _validate_params(self):
        """ Verify that the required parameters exist for this eval type """
        for fk in self._required:
            assert fk in self._p.keys(), f"Required param: {fk}"
        for p in ["vidx_feat", "hidx_feat", "time_feat"]:
            dk,fk = self._p[p]
            assert dk in self._f.keys(), f"{dk} not in {list(self._f.keys())}"
            assert fk in self._f[dk], f"{fk} not in dataset {dk} {self._f[dk]}"
        for dk,fk in self._p["cov_feats"]:
            assert dk in self._f.keys(), f"{dk} not in {list(self._f.keys())}"
            assert fk in self._f[dk], f"{fk} not in dataset {dk} {self._f[dk]}"
        if len(self._p["cov_feats"]):
            assert self._p["cov_reduce_metric"] in REDUCE_FUNCS.keys()
            assert isinstance(self._p["cov_reduce_axes"], tuple)
            assert all(isinstance(v,int) and v>0
                    for v in self._p["cov_reduce_axes"])

    def add_batch(self, bdict:dict):
        """ Update the partial evaluation data with a new batch of samples """
        if self._r is None:
            self._r = {"vidxs":[], "hidxs":[], "etimes":[], "cov":[]}
        self._r["vidxs"].append(self.get_farray(
                *self._p["vidx_feat"], bdict).astype(np.uint16))
        self._r["hidxs"].append(self.get_farray(
                *self._p["hidx_feat"], bdict).astype(np.uint16))
        ## take only the first timestep from each sample (includes window)
        self._r["etimes"].append(self.get_farray(
                *self._p["time_feat"], bdict)[:,0].astype(np.uint32))
        if len(self._p["cov_feats"]):
            self._r["cov"].append(np.stack([
                REDUCE_FUNCS[self._p["cov_reduce_metric"]](
                    self.get_farray(*cf,bdict,axis=self._p["cov_reduce_axes"]))
                for cf in self._p["cov_feats"]
                ], axis=-1))
        return self

    def final_results(self, time_format="epoch"):
        """
        Present the batch-wise arrays of indeces

        :@param time_format: "epoch" or "ydh". Determines how times are
            represented. "epoch" results in integer epoch seconds, while "ydh"
            is has a 3-vector of integers (year, doy, hour)
        """
        assert time_format in ("epoch", "ydh")
        if time_format == "ydh":
            f = get_epoch_to_index_func(include_year=True)
            ix_vfunc = np.vectorize(f, signature="()->(3)")
            t = [ix_vfunc(x).astype(np.uint16) for x in self._r["etimes"]]
        else:
            t = self._r["etimes"]
        return {"vidxs":self._r["vidxs"], "hidxs":self._r["hidxs"],
                "times":t, "cov":self._r["cov"]}

class EvalTemporal(Evaluator):
    """
    Store the average and standard deviation with respect to time of
    day and day of year using Welford's online algorithm.

    Required parameters in `params` dict:

    :@param eval_feats: list of 2-tuples of strings (dataset, feat) indicating
        the data features for which to collect statistics. All of the arrays
        MUST have simultaneous time axes, and the time axis must be along the
        same dimension for all of them.
    :@param batch_axis: integer axis indicating which dimension of the arrays
        specified by eval_feats represents the batches. It must be the same
        for all provided eval_feats.
    :@param reduce_func: String indicating which function to use for reducing
        axes other than the batch, time, and feature axes to a scalar for
        timestep-wise calculations. Naturally, this parameter can be left as
        None if the data won't have any other dimensions. Otherwise it must
        be a string key of REDUCE_FUNCS, ie "mean", "min", "max".
    :@param time_feat: 2-tuple of strings (dataset, feat) indicating where a
        2d array (batch, timesteps) of epoch times is provided, which are
        needed for indexing the stored statistics.
    :@param time_axis: integer axis indicating which dimension of the arrays
        specified by eval_feats represents the time axis. It must be the same
        for all provided eval_feats; if a different axis is needed, just
        declare a separate evaluator for those data features.
    :@param time_slice: 2-tuple of integer or None indicating the subset of
        the 1d time array corresponding to the data features' time axis.
    """
    _required = ["eval_feats", "batch_axis", "reduce_func",
                "time_feat", "time_axis", "time_slice"]

    def __init__(self, params:dict, feats:dict,
            results:dict=None, meta:dict={}):
        """ Declare an EvalTemporal evaluator. See superclass initializer. """
        super(EvalTemporal, self).__init__(
                "EvalTemporal",
                params=params,
                feats=feats,
                results=results,
                meta=meta,
                )

    def _validate_params(self):
        """ """
        assert all(k in self._p.keys() for k in self._required), \
            f"All of these must be provided as params: {self._required}"
        for dk,fk in self._p["eval_feats"]:
            assert dk in self._f.keys(), f"{dk} not in {list(self._f.keys())}"
            assert fk in self._f[dk], f"{fk} not in dataset {dk} {self._f[dk]}"
        dk,fk = self._p["time_feat"]
        assert dk in self._f.keys(), \
            f"time dataset {dk} not in {list(self._f.keys())}"
        assert fk in self._f[dk], \
            f"time feature {fk} not in dataset {dk} {self._f[dk]}"
        assert isinstance(self._p["batch_axis"], int)
        assert isinstance(self._p["time_axis"], int)
        assert isinstance(self._p["time_slice"], (list, tuple)), \
            "time_slice parameter must be a 2-tuple of ints or None " + \
            "representing a [start,stop) index slice along the time axis"
        assert len(self._p["time_slice"])==2, \
            "time_slice parameter must be a 2-tuple of ints or None " + \
            "representing a [start,stop) index slice along the time axis"
        assert self._p["reduce_func"] in REDUCE_FUNCS.keys() \
                or self._p["reduce_func"] is None


    def add_batch(self, bdict:dict):
        """ """
        ## slice for subsetting time array to index data arrays along time axis
        seq_slice = slice(*self._p["time_slice"])
        ## extract the feature associated with epoch times. It's assumed that
        ## after extraction this is a 2d array (batch,times) of epoch times.
        etimes = self.get_farray(*self._p["time_feat"], bdict)[:,seq_slice]
        assert etimes.ndim==2, f"{etimes.shape = }"

        ## vectorize the function to get (ToD, DoY) indeces from etimes nice
        #if self._ix_vfunc is None:
        f = get_epoch_to_index_func(include_year=False)
        _ix_vfunc = np.vectorize(f, signature="()->(2)")
        ## use the vectorized function to calculate the indeces;
        ## the result should be a (B,S,2) array
        ix_doy_tod = _ix_vfunc(etimes)

        ## Extract the data arrays for which to calculate statistics
        bdata = []
        for df in self._p["eval_feats"]:
            ## probably (B,S) shaped when returned, but treat more generally
            x = self.get_farray(*df, bdict)
            assert x.shape[self._p["time_axis"]] == (nt := etimes.shape[1]), \
                f"The time axis size for {df} doesn't match the time slice; " \
                + f"{x.shape = }, while {etimes.shape = }"
            assert x.shape[self._p["batch_axis"]] == (nt := etimes.shape[0]), \
                f"The batch axis size for {df} doesn't match the time batch;" \
                + f" {x.shape = }, while {etimes.shape = }"
            extra_axes = tuple(set(range(x.ndim)) - set([
                self._p["time_axis"], self._p["batch_axis"]]))

            ## reorder the axes so batch and time come first, in that order.
            ax_order = (self._p["batch_axis"],self._p["time_axis"],*extra_axes)
            x = x.transpose(ax_order)
            ## use the specified function to reduce extra axes if they exist
            if extra_axes:
                x = REDUCE_FUNCS[self._p["reduce_func"]](
                        x, axis=tuple(range(2,x.ndim)))
            ## collapse batch and time dimensions together
            bdata.append(x.ravel())

        ## collapse the batch and time dimensions together for time indeces
        ix_doy_tod = ix_doy_tod.reshape((-1,2)) ## (T,2) array of indeces
        bdata = np.stack(bdata, axis=-1) ## (T,F) array of batch data values

        if self._r is None:
            self._r = {
                "count":np.zeros((366, 24, 1)),
                "mean":np.full((366, 24, len(self._p["eval_feats"])), np.nan),
                "m2":np.zeros((366, 24, len(self._p["eval_feats"]))),
                }

        ## accumulate using welford's online algorithm
        for i in range(bdata.shape[0]):
            tmpix = ix_doy_tod[i]
            self._r["count"][*tmpix] += 1 ## increment the count
            ## initialize the mean if not already established
            if self._r["count"][*tmpix] == 1:
                self._r["mean"][*tmpix] = bdata[i]
            d_1 = bdata[i] - self._r["mean"][*tmpix]
            self._r["mean"][*tmpix] += d_1 / self._r["count"][*tmpix]
            d_2 = bdata[i] - self._r["mean"][*tmpix]
            self._r["m2"][*tmpix] += d_1 * d_2 ## sum of squares of diffs
        return self

    def final_results(self):
        """
        Return a dict of (DoY, ToD, F) numpy arrays representing the sample
        count, mean value, variance, and sample variance of accumulated data.
        """
        ## can't calculate variance of only one value
        m_valid = (self._r["count"] >= 2)
        c = np.where(m_valid, self._r["count"], np.nan)
        m = np.where(m_valid, self._r["mean"], np.nan)
        v = np.where(m_valid, self._r["m2"]/self._r["count"], np.nan)
        s = np.where(m_valid, self._r["m2"]/(self._r["count"]-1), np.nan)
        return { "count":c, "mean":m, "var":v, "svar":s }

EVALUATORS = {
        "EvalTemporal":EvalTemporal,
        "EvalSampleSources":EvalSampleSources,
        "EvalJointHist":EvalJointHist,
        "EvalStatic":EvalStatic,
        #"EvalHorizon":EvalHorizon,
        #"EvalEfficiency":EvalEfficiency,
        #"EvalGridAxes":EvalGridAxes,
        }

if __name__=="__main__":
    pass
