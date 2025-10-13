from abc import ABC,abstractmethod
from copy import deepcopy
import numpy as np
import pickle as pkl
from datetime import datetime,timezone
from typing import Callable
from pathlib import Path
import matplotlib.pyplot as plt


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
    ix_doy = (t.tm_yday - 1 + int(tmp_tod == 24)) % 366
    return np.array([ix_doy, ix_tod])


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
        t, p, r, f, m = pkl.load(pkl_path.open("rb"))
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

    def to_pkl(self, pkl_path:Path):
        """ Dump the attribute dicts for this instance into a pkl """
        ptup = (self._t, self._p, self._f, self._r, self._m)
        pkl.dump(ptup, pkl_path.open("wb"))
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
        assert batch_dict[dataset].shape[-1]==len(self._f[dataset])
        return batch_dict[dataset][...,self._f[dataset].index(feat)]

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
    _required = ["eval_feats", "batch_axis", "reduce_func"
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
        self._ix_vfunc = None

    def _validate_params(self):
        """ """
        assert all(k in self._p.keys() for k in self.required), \
            f"All of the following must be provided as params: {self.required}"
        assert isinstance(self._p["use_absolute_error"], bool)
        for dk,fk in self._p["eval_feats"]:
            assert dk in self._f.keys(), f"{dk} not in {list(self._f.keys())}"
            assert fk in self._f[dk], f"{fk} not in dataset {dk} {self._f[dk]}"
        dk,fk = self._p["time_feat"]
        assert dk in self._f.keys(), \
            f"time dataset {dk} not in {list(self._f.keys())}"
        assert fk in self._f[dk], \
            f"time feature {fk} not in dataset {dk} {self._f[dk]}"

    def add_batch(self, bdict:dict):
        """ """
        ## slice for subsetting time array to index data arrays along time axis
        seq_slice = slice(*self._p["time_slice"])
        ## extract the feature associated with epoch times. It's assumed that
        ## after extraction this is a 2d array (batch,times) of epoch times.
        etimes = self.get_farray(*self._p["time_feat"], bdict)[:,seq_slice]
        assert etimes.ndim==2, f"{etimes.shape = }"

        ## vectorize the function to get (ToD, DoY) indeces from etimes nice
        if self._ix_vfunc is None:
            self._ix_vfunc = np.vectorize(
                    _epoch_to_tod_doy_index, signature="()->(2)")

        ## use the vectorized function to calculate the indeces;
        ## the result should be a (B,S,2) array
        ix_doy_tod = self._ix_vfunc(etimes)

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
                x = REDUCE_FUNC[self._p["reduce_func"]](
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
            d_2 = bdatat[i] - self._r["mean"][*tmpix]
            self._r["m2"][*tmpix] += d_1 * d_2 ## sum of squares of diffs

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

class EvalEfficiency(Evaluator):
    """
    Store the mean and approximate standard deviation of
    multiple efficiency metrics for residual and state error including:
     - Correlation Coefficient (CC)
     - Mean Absolute Error (MSE)
     - Root Mean Squared Error (RMSE)
     - Kling-Gupta Efficiency (KGE)
     - Nash-Sutcliffe Efficiency (NSE)
    """
    def __init__(self, pred_feat_idx=None, pred_coarseness=1, attrs={}):
        """ """
        self._pfix = pred_feat_idx
        self._pred_coarseness = pred_coarseness
        self._counts = None ## Number of samples included in sums
        self._dropped_counts = None
        ## Residual efficiency metrics
        self._r_cc_sum = None
        self._r_cc_var_sum = None
        self._r_mae_sum = None
        self._r_mae_var_sum = None
        self._r_mse_sum = None
        self._r_mse_var_sum = None
        self._r_kge_sum = None
        self._r_kge_var_sum = None
        self._r_nse_sum = None
        self._r_nse_var_sum = None
        self._r_nnse_sum = None
        self._r_nnse_var_sum = None
        ## State efficiency metrics
        self._s_cc_sum = None
        self._s_cc_var_sum = None
        self._s_mae_sum = None
        self._s_mae_var_sum = None
        self._s_mse_sum = None
        self._s_mse_var_sum = None
        self._s_kge_sum = None
        self._s_kge_var_sum = None
        self._s_nse_sum = None
        self._s_nse_var_sum = None
        self._s_nnse_sum = None
        self._s_nnse_var_sum = None

        self._attrs = attrs ## additional attributes

    @property
    def attrs(self):
        return self._attrs

    def add_batch(self, inputs, true_state, predicted_residual, indeces=None):
        """ """
        assert not self._pfix is None
        ys,pr = true_state[...,self._pfix],predicted_residual[...,self._pfix]
        ## the predicted state time series
        ps = ys[:,0][:,np.newaxis] + np.cumsum(pr, axis=1)
        ## Calculate the label residual from labels
        yr = ys[:,1:]-ys[:,:-1]
        ## After calculating true residual, ignore the initial (known) state
        ys = ys[:,1:]
        ds = ps - ys
        dr = pr - yr

        ## If the state doesn't change due to frozen soil, there will be an
        ## infinite denominator for cc, kge, and mse. Drop these cases for
        ## only the affected layers.
        m_drop_idle = (np.std(ys, axis=1, keepdims=False) == 0)
        tmp_dropped_counts = np.count_nonzero(m_drop_idle, axis=0)

        ## Calculate the absolute error in the residual and state predictions
        s_mae = np.average(np.abs(ds), axis=1)
        r_mae = np.average(np.abs(dr), axis=1)
        s_mse = np.average(ds**2, axis=1)
        r_mse = np.average(dr**2, axis=1)

        ## Calculate the statistical efficiency metrics along the sequence axis
        s_cc = np.squeeze(pearson_coeff(
            ys, ps, axis=1, keepdims=True))
        r_cc = np.squeeze(pearson_coeff(
            yr, pr, axis=1, keepdims=True))
        s_kge = np.squeeze(kling_gupta_efficiency(
            ys, ps, axis=1, keepdims=True))
        r_kge = np.squeeze(kling_gupta_efficiency(
            yr, pr, axis=1, keepdims=True))
        s_nse = np.squeeze(nash_sutcliffe_efficiency(
            ys, ps, axis=1, keepdims=True))
        r_nse = np.squeeze(nash_sutcliffe_efficiency(
            yr, pr, axis=1, keepdims=True))
        s_nnse = 1/(2-s_nse)
        r_nnse = 1/(2-r_nse)

        ## drop any sequence with zero denominator due to frozen soil
        m_valid = np.isfinite(s_cc) & np.isfinite(r_cc) \
                & np.isfinite(s_kge) & np.isfinite(r_kge) \
                & np.isfinite(s_nse) & np.isfinite(r_nse)
        if np.any(~m_valid):
            s_cc = s_cc[m_valid]
            r_cc = r_cc[m_valid]
            s_kge = s_kge[m_valid]
            r_kge = r_kge[m_valid]
            s_nse = s_nse[m_valid]
            r_nse = r_nse[m_valid]
            r_nnse = r_nnse[m_valid]
            s_nnse = s_nnse[m_valid]

        if self._counts is None:
            self._counts = 0
            self._dropped_counts = 0
            self._s_mae_sum = 0
            self._s_mae_var_sum = 0
            self._r_mae_sum = 0
            self._r_mae_var_sum = 0
            self._s_mse_sum = 0
            self._s_mse_var_sum = 0
            self._r_mse_sum = 0
            self._r_mse_var_sum = 0
            self._s_cc_sum = 0
            self._s_cc_var_sum = 0
            self._r_cc_sum = 0
            self._r_cc_var_sum = 0
            self._s_kge_sum = 0
            self._s_kge_var_sum = 0
            self._r_kge_sum = 0
            self._r_kge_var_sum = 0
            self._s_nse_sum = 0
            self._s_nse_var_sum = 0
            self._r_nse_sum = 0
            self._r_nse_var_sum = 0
            self._s_nnse_sum = 0
            self._s_nnse_var_sum = 0
            self._r_nnse_sum = 0
            self._r_nnse_var_sum = 0

        ## Accumulate the mean and variance sums of each metric for this batch
        self._counts += s_mae.shape[0]
        self._dropped_counts += tmp_dropped_counts
        self._s_mae_sum += np.sum(s_mae, axis=0, dtype=np.float64)
        self._s_mae_var_sum += np.sum(
                (s_mae-self._s_mae_sum/self._counts)**2,
                axis=0, dtype=np.float64)
        self._r_mae_sum += np.sum(r_mae, axis=0, dtype=np.float64)
        self._r_mae_var_sum += np.sum(
                (r_mae-self._r_mae_sum/self._counts)**2,
                axis=0, dtype=np.float64)
        self._s_mse_sum += np.sum(s_mse, axis=0, dtype=np.float64)
        self._s_mse_var_sum += np.sum(
                (s_mse-self._s_mse_sum/self._counts)**2,
                axis=0, dtype=np.float64)
        self._r_mse_sum += np.sum(r_mse, axis=0, dtype=np.float64)
        self._r_mse_var_sum += np.sum(
                (r_mse-self._r_mse_sum/self._counts)**2,
                axis=0, dtype=np.float64)

        rdc_counts = self._counts - self._dropped_counts
        ## Don't create nans if there aren't any valid samples this time around
        if s_mae.shape[0]==tmp_dropped_counts:
            return
        self._s_cc_sum += np.sum(s_cc, axis=0, dtype=np.float64)
        self._s_cc_var_sum += np.sum(
                (s_cc-self._s_cc_sum/rdc_counts)**2,
                axis=0, dtype=np.float64)
        self._r_cc_sum += np.sum(r_cc, axis=0, dtype=np.float64)
        self._r_cc_var_sum += np.sum(
                (r_cc-self._r_cc_sum/rdc_counts)**2,
                axis=0, dtype=np.float64)
        self._s_kge_sum += np.sum(s_kge, axis=0, dtype=np.float64)
        self._s_kge_var_sum += np.sum(
                (s_kge-self._s_kge_sum/rdc_counts)**2,
                axis=0, dtype=np.float64)
        self._r_kge_sum += np.sum(r_kge, axis=0, dtype=np.float64)
        self._r_kge_var_sum += np.sum(
                (r_kge-self._r_kge_sum/rdc_counts)**2,
                axis=0, dtype=np.float64)
        self._s_nse_sum += np.sum(s_nse, axis=0, dtype=np.float64)
        self._s_nse_var_sum += np.sum(
                (s_nse-self._s_nse_sum/rdc_counts)**2,
                axis=0, dtype=np.float64)
        self._r_nse_sum += np.sum(r_nse, axis=0, dtype=np.float64)
        self._r_nse_var_sum += np.sum(
                (r_nse-self._r_nse_sum/rdc_counts)**2,
                axis=0, dtype=np.float64)

        ## Honestly KGE is only really useful as its term-wise decomposition:
        ## pearson, pred_mean/true_mean, and pred_stdev/true_stdev; as such,
        ## only including normalized nash-sutcliffe to capture the same effect
        self._s_nnse_sum += np.sum(s_nnse, axis=0, dtype=np.float64)
        self._s_nnse_var_sum += np.sum(
                (s_nnse-self._s_nnse_sum/rdc_counts)**2,
                axis=0, dtype=np.float64)
        self._r_nnse_sum += np.sum(r_nnse, axis=0, dtype=np.float64)
        self._r_nnse_var_sum += np.sum(
                (r_nnse-self._r_nnse_sum/rdc_counts)**2,
                axis=0, dtype=np.float64)

    def add(self, other:"EvalEfficiency"):
        """ """
        eff1 = self.get_results()
        eff2 = other.get_results()
        ## Assume by default all config comes from this object
        new_data = deepcopy(eff1)
        sum_fields = [
                "s_mae_sum", "s_mae_var_sum", "r_mae_sum", "r_mae_var_sum",
                "s_mse_sum", "s_mse_var_sum", "r_mse_sum", "r_mse_var_sum",
                "s_cc_sum", "s_cc_var_sum", "r_cc_sum", "r_cc_var_sum",
                "s_kge_sum", "s_kge_var_sum", "r_kge_sum", "r_kge_var_sum",
                "s_nse_sum", "s_nse_var_sum", "r_nse_sum", "r_nse_var_sum",
                "s_nnse_sum", "s_nnse_var_sum", "r_nnse_sum", "r_nnse_var_sum",
                ]
        ## Update the added data with the summed field
        new_data.update({f:eff1[f]+eff2[f] for f in sum_fields})
        return EvalEfficiency().from_dict(new_data)

    def get_var(self, state_or_res:str, metric:str):
        """ """
        assert state_or_res[0] in {"s", "r"}
        if metric in {"cc", "kge", "nse", "nnse"}:
            denom = self._counts - self._dropped_counts
        elif metric in {"mae", "mse"}:
            denom = self._counts
        else:
            raise ValueError(f"Invalid {metric = }")
        key = f"{state_or_res[0]}_{metric}_var_sum"
        return self.get_results().get(key) / denom

    def get_mean(self, state_or_res:str, metric:str):
        """ """
        assert state_or_res[0] in {"s", "r"}
        if metric in {"cc", "kge", "nse", "nnse"}:
            denom = self._counts - self._dropped_counts
        elif metric in {"mae", "mse"}:
            denom = self._counts
        else:
            raise ValueError(f"Invalid {metric = }")
        key = f"{state_or_res[0]}_{metric}_sum"
        return self.get_results().get(key) / denom

    def get_results(self):
        """ """
        return {
                "pred_feat_idx":self._pfix,
                "counts":self._counts,
                "dropped_counts":self._dropped_counts,
                "s_mae_sum":self._s_mae_sum,
                "s_mae_var_sum":self._s_mae_var_sum,
                "r_mae_sum":self._r_mae_sum,
                "r_mae_var_sum":self._r_mae_var_sum,
                "s_mse_sum":self._s_mse_sum,
                "s_mse_var_sum":self._s_mse_var_sum,
                "r_mse_sum":self._r_mse_sum,
                "r_mse_var_sum":self._r_mse_var_sum,
                "s_cc_sum":self._s_cc_sum,
                "s_cc_var_sum":self._s_cc_var_sum,
                "r_cc_sum":self._r_cc_sum,
                "r_cc_var_sum":self._r_cc_var_sum,
                "s_kge_sum":self._s_kge_sum,
                "s_kge_var_sum":self._s_kge_var_sum,
                "r_kge_sum":self._r_kge_sum,
                "r_kge_var_sum":self._r_kge_var_sum,
                "s_nse_sum":self._s_nse_sum,
                "s_nse_var_sum":self._s_nse_var_sum,
                "r_nse_sum":self._r_nse_sum,
                "r_nse_var_sum":self._r_nse_var_sum,
                "s_nnse_sum":self._s_nnse_sum,
                "s_nnse_var_sum":self._s_nnse_var_sum,
                "r_nnse_sum":self._r_nnse_sum,
                "r_nnse_var_sum":self._r_nnse_var_sum,
                "pred_coarseness":self._pred_coarseness,
                "attrs":self._attrs,
                }

    def to_pkl(self, pkl_path:Path, additional_attributes:dict={}):
        """
        Write the residual and state efficiency error results to a pkl file

        :@param pkl_path: Path to a non-existing pkl path to dump results to.
        :@param additional_attributes: Dict of additional information to
            include alongside the horizon error distribution data. If any of
            the keys match existing auxillary attributes the new ones provided
            here will replace them.
        """
        pkl.dump(self.get_results(), pkl_path.open("wb"))

    def from_dict(self, config_dict):
        """ """
        p = config_dict
        self._pfix = p["pred_feat_idx"]
        self._counts = p["counts"]
        self._dropped_counts = p["dropped_counts"]
        self._s_mae_sum = p["s_mae_sum"]
        self._s_mae_var_sum = p["s_mae_var_sum"]
        self._r_mae_sum = p["r_mae_sum"]
        self._r_mae_var_sum = p["r_mae_var_sum"]
        self._s_mse_sum = p["s_mse_sum"]
        self._s_mse_var_sum = p["s_mse_var_sum"]
        self._r_mse_sum = p["r_mse_sum"]
        self._r_mse_var_sum = p["r_mse_var_sum"]
        self._s_cc_sum = p["s_cc_sum"]
        self._s_cc_var_sum = p["s_cc_var_sum"]
        self._r_cc_sum = p["r_cc_sum"]
        self._r_cc_var_sum = p["r_cc_var_sum"]
        self._s_kge_sum = p["s_kge_sum"]
        self._s_kge_var_sum = p["s_kge_var_sum"]
        self._r_kge_sum = p["r_kge_sum"]
        self._r_kge_var_sum = p["r_kge_var_sum"]
        self._s_nse_sum = p["s_nse_sum"]
        self._s_nse_var_sum = p["s_nse_var_sum"]
        self._r_nse_sum = p["r_nse_sum"]
        self._r_nse_var_sum = p["r_nse_var_sum"]
        self._s_nnse_sum = p["s_nnse_sum"]
        self._s_nnse_var_sum = p["s_nnse_var_sum"]
        self._r_nnse_sum = p["r_nnse_sum"]
        self._r_nnse_var_sum = p["r_nnse_var_sum"]
        self._pred_coarseness = p.get("pred_coarseness", 1)
        self._pred_coarseness = p.get("pred_coarseness", 1)
        self._attrs = p["attrs"]
        return self

    def from_pkl(self, pkl_path:Path):
        """ """
        return self.from_dict(pkl.load(pkl_path.open("rb")))

class EvalConditional(Evaluator):
    def __init__(self, conditions=[], feat_args=[], pred_coarseness=1,
            coarse_reduce_func="mean", attrs={}):
        """ """
        self._pred_coarseness = pred_coarseness
        self._axes = (axes,) if type(axes)==int else tuple(axes)
        self._counts = None
        ## keep feat args with un-compiled lambda strings for serializability
        self._feat_args_unevaluated = feat_args
        self._feat_args,self._feat_is_func = zip(
                *map(self._validate_feat_arg,feat_args)
                ) if len(feat_args) else (None,None)

        self._conditions_unevaluated = conditions
        self._conditions = []
        for ingredients,funcstr in conditions:
            tmp_ing,is_func = zip(*map(self._validate_feat_arg, ingredients))
            assert all(not f for f in is_func), \
                    "condition ingredients cannot be functions"
            self._conditions.append((tmp_ing, eval(funcstr)))

        self._static = None
        self._time = None
        self._indeces = None
        self._attrs = attrs ## additional attributes
        self._rfuncs = {"min":np.amin, "mean":np.average, "max":np.amax}
        self._coarse_reduce_str = coarse_reduce_func
        try:
            self._crf = self._rfuncs[coarse_reduce_func]
        except:
            raise ValueError(f"coarse_reduce_func must be in: " + \
                    "{self._rfuncs.keys()}")


    @staticmethod
    def _validate_feat_arg(feat_arg):
        """
        Verify the validity of a feature-specifying argument, which may
        identify a horizon, true, predicted, or error data feature, or
        specify a function of one or more of the above.

        :@param feat_arg: feature arg 2-tuple following the format specified
            in the class docstrign
        :@return: 2-tuple (feat_arg, is_callable) where feat_arg has compiled
            function strings and is_callable specified whether the feature
            is functional.
        """
        sources = ("horizon", "true_res", "pred_res", "err_res",
                "true_state", "pred_state", "err_state")
        assert len(feat_arg)==2, feat_arg
        ## feat args are for stored feats iff they have type profile (str, int)
        if type(feat_arg[0])==str and type(feat_arg[1])==int:
            assert feat_arg[0] in sources,f"{feat_arg[0]} must be in {sources}"
            return feat_arg,False
        ## Otherwise it is a functional arg or invalid
        for arg in feat_arg[0]:
            _,is_func = EvalConditional._validate_feat_arg(arg)
            assert is_func, "Functional feat arg must itself be a stored " + \
                    "feat, not {arg}"
        if isinstance(feat_arg[1], str):
            axis_args = (axis_args[0], eval(axis_args[1]))
        else:
            assert isinstance(axis_args[1], Callable)
        return axis_args,True

    def add_batch(self, inputs, true_state, predicted_residual, indeces=None):
        """ """
        (_,h,s,si,t),ys,pr = inputs,true_state,predicted_residual
        ## store grid indeces if requested, provided, and not done already
        if not indeces is None and self._indeces is None:
            self._indeces = indeces
        ## the predicted state time series
        ps = ys[:,0,:][:,np.newaxis,:] + np.cumsum(pr, axis=1)
        ## Calculate the label residual from labels
        yr = ys[:,1:]-ys[:,:-1]
        es = ps - ys[:,1:]
        er = pr - yr

        si_int = np.argwhere(si)[:,1]
        x_si = np.repeat(si_int[:,np.newaxis], es.shape[1], axis=1)
        x_s = np.repeat(s[:,np.newaxis], es.shape[1])

        if self._absolute_error:
            es = np.abs(es)
            er = np.abs(er)

        ## Make a dict of the data arrays to make extraction easier
        if self._pred_coarseness != 1:
            b,_,f = h.shape
            h = h.reshape(h.shape[0],-1,self._pred_coarseness,h.shape[-1])
            h = self._crf(h, axis=2)

        data = {"horizon":h, "true_res":yr, "pred_res":pr, "err_res":er,
                "true_state":ys[:,1:], "pred_state":ps, "err_state":es}
        feats = []
        for f,is_func in zip(self._feat_args, self._feat_is_func):
            ## Collect arguments and evaluate the method if feat is functional
            if is_func:
                args = [data[s][...,ix] for s,ix in f[0]]
                feats.append(f[1](*args))
            ## Otherwise just extract the data from the proper source array
            else:
                s,ix = f
                feats.append(data[s][...,ix])
        feats = np.stack(feats, axis=-1)
        mask_2d = np.logical_and(*[
            func([data[s][...,ix] for s,ix in ing])
            for ing,func in conds
            ])
        mask_1d = np.any


    def get_results(self):
        """ """
        pass

    def to_pkl(self):
        pass

    def from_pkl():
        pass


class EvalGridAxes(Evaluator):
    """
    Stores mean and stdev of values along a subset of axes separately for each
    batch. For gridded data, each batch is implied to correspond to a different
    init time. Ultimately, the dataset is (T, P, S, F) shape for T init times,
    P valid pixels, S sequence elements, and F features.

    The provided axis numbers are the ones that will be preserved, while error
    will be reduced along each of the remaining ones.

    This should work for any dataset as long as the sequence axis is always
    the 2nd one per batch.

    The features with respect to which bulk values are collected are defined
    by feat_args, which is a list of 2-tuples specifying either a stored or
    functional feature.

    Stored feature args corresponds to horizon, true, predicted, or error data
    that are explicitly returned by the generator. Stored feat args must be
    a 2-tuple like (data_source, feat_idx).

    Functional feature args recieve 1 or more arguments each defined by a
    stored feature configuration as specified above, and execute an arbitrary
    function according to the provided method.

    Functional feat args must be a 2-tuple like (args, func) where args is a
    list of 2-tuples (data_source, feat_idx) and func is a Callable or a string
    defining a lambda function.

    data_source must be one of:
    {"horizon", "true_res", "pred_res", "err_res",
     "true_state", "pred_state", "err_state"}
    """
    def __init__(self, feat_args=[], axes=tuple(), pred_coarseness=1,
            store_static=False, store_time=False, coarse_reduce_func="mean",
            use_absolute_error=False, attrs={}):
        """ """
        self._pred_coarseness = pred_coarseness
        self._axes = (axes,) if type(axes)==int else tuple(axes)
        self._counts = None
        self._batch_count = 0
        ## keep feat args with un-compiled lambda strings for serializability
        self._feat_args_unevaluated = feat_args
        self._feat_args,self._feat_is_func = zip(
                *map(self._validate_feat_arg,feat_args)
                ) if len(feat_args) else (None,None)
        self._sum = None ## Sum of feature wrt horizon
        self._var_sum = None ## State error partial variance sum
        self._store_static = store_static
        self._static = None
        self._static_int = None
        self._store_time = store_time
        self._time = None
        self._indeces = None
        self._attrs = attrs ## additional attributes
        self._rfuncs = {"min":np.amin, "mean":np.average, "max":np.amax}
        self._absolute_error = use_absolute_error
        self._coarse_reduce_str = coarse_reduce_func
        try:
            self._crf = self._rfuncs[coarse_reduce_func]
        except:
            raise ValueError(f"coarse_reduce_func must be in: " + \
                    "{self._rfuncs.keys()}")
    @property
    def attrs(self):
        return self._attrs
    @property
    def time(self):
        return self._time
    @property
    def static(self):
        return self._static
    @property
    def static_int(self):
        return self._static_int
    @property
    def indeces(self):
        return self._indeces
    @property
    def average(self):
        return self._sum / self._counts
    @property
    def variance(self):
        return self._var_sum / self._counts

    @staticmethod
    def _validate_feat_arg(feat_arg):
        """
        Verify the validity of a feature-specifying argument, which may
        identify a horizon, true, predicted, or error data feature, or
        specify a function of one or more of the above.

        :@param feat_arg: feature arg 2-tuple following the format specified
            in the class docstrign
        :@return: 2-tuple (feat_arg, is_callable) where feat_arg has compiled
            function strings and is_callable specified whether the feature
            is functional.
        """
        sources = ("horizon", "true_res", "pred_res", "err_res",
                "true_state", "pred_state", "err_state")
        assert len(feat_arg)==2, feat_arg
        ## feat args are for stored feats iff they have type profile (str, int)
        if type(feat_arg[0])==str and type(feat_arg[1])==int:
            assert feat_arg[0] in sources,f"{feat_arg[0]} must be in {sources}"
            return feat_arg,False
        ## Otherwise it is a functional arg or invalid
        for arg in feat_arg[0]:
            _,is_func = EvalGridAxes._validate_feat_arg(arg)
            assert is_func, "Functional feat arg must itself be a stored " + \
                    "feat, not {arg}"
        if isinstance(feat_arg[1], str):
            axis_args = (axis_args[0], eval(axis_args[1]))
        else:
            assert isinstance(axis_args[1], Callable)
        return axis_args,True

    def add_batch(self, inputs, true_state, predicted_residual, indeces=None):
        """ """
        (_,h,s,si,t),ys,pr = inputs,true_state,predicted_residual
        ## store grid indeces if requested, provided, and not done already
        if not indeces is None and self._indeces is None:
            self._indeces = indeces
        ## the predicted state time series
        ps = ys[:,0,:][:,np.newaxis,:] + np.cumsum(pr, axis=1)
        ## Calculate the label residual from labels
        yr = ys[:,1:]-ys[:,:-1]
        if self._static is None and self._store_static:
            self._static = s
            self._static_int = si
        if self._store_time:
            if self._time is None:
                self._time = t[np.newaxis, -ys.shape[1]::self._pred_coarseness]
            else:
                tmpt = t[np.newaxis, -h.shape[1]::self._pred_coarseness]
                self._time = np.concatenate([self._time, tmpt], axis=0)
        es = ps - ys[:,1:]
        er = pr - yr
        if self._absolute_error:
            es = np.abs(es)
            er = np.abs(er)
        ## Make a dict of the data arrays to make extraction easier
        if self._pred_coarseness != 1:
            b,_,f = h.shape
            h = h.reshape(h.shape[0],-1,self._pred_coarseness,h.shape[-1])
            h = self._crf(h, axis=2)

        data = {"horizon":h, "true_res":yr, "pred_res":pr, "err_res":er,
                "true_state":ys[:,1:], "pred_state":ps, "err_state":es}
        feats = []
        for f,is_func in zip(self._feat_args, self._feat_is_func):
            ## Collect arguments and evaluate the method if feat is functional
            if is_func:
                args = [data[s][...,ix] for s,ix in f[0]]
                feats.append(f[1](*args))
            ## Otherwise just extract the data from the proper source array
            else:
                s,ix = f
                feats.append(data[s][...,ix])

        feats = np.stack(feats, axis=-1)

        ## Keep requested axes, and never reduce along the feature axis. Also
        ## ignore the first axis for now since it is only implied.
        r_axes = tuple([
                a+1 for a in range(len(feats.shape)-1)
                if a+1 not in self._axes
                ])
        ## set the counts for the sum/var arrays, which is the product of the
        ## number of elements along each marginalized axis
        self._batch_count += 1
        self._counts = np.prod([feats.shape[a-1] for a in r_axes]) \
                * [self._batch_count, 1][0 in self._axes]

        ## Create new init time axis
        feats = feats[None]
        tmp_sum = np.sum(feats, keepdims=True, axis=r_axes, dtype=np.float64)

        ## Case where batch axis is kept
        if 0 in self._axes:
            ## Only calculate variance within this timestep if not
            ## marginalizing over the first axis
            tmp_var = np.sum(
                    (feats - tmp_sum/self._counts)**2,
                    axis=r_axes, keepdims=True, dtype=np.float64)
            if self._sum is None:
                self._sum = tmp_sum
                self._var_sum = tmp_var
            else:
                self._sum = np.concatenate([self._sum, tmp_sum], axis=0)
                self._var_sum = np.concatenate([self._var_sum,tmp_var], axis=0)
        ## Case where batch axis is marginalized over
        else:
            ## For averaging over first axis, use mean values that gradually
            ## update over multiple batches to calculate variance
            if self._sum is None:
                self._sum = tmp_sum
                tmp_var = np.sum(
                        (feats - self._sum/self._counts)**2,
                        axis=r_axes, keepdims=True, dtype=np.float64)
                self._var_sum = tmp_var
            else:
                self._sum += tmp_sum
                tmp_var = np.sum(
                        (feats - self._sum/self._counts)**2,
                        axis=r_axes, keepdims=True, dtype=np.float64)
                self._var_sum += tmp_var

    def get_results(self):
        """ """
        return {
                "avg":self._sum,
                "var":self._var_sum,
                "static":self._static,
                "static_int":self._static_int,
                "time":self._time,
                "counts":self._counts,
                "axes":self._axes,
                "indeces":self._indeces,
                "batch_count":self._batch_count,
                "feat_args":self._feat_args_unevaluated,
                "pred_coarseness":self._pred_coarseness,
                "coarse_reduce_func":self._coarse_reduce_str,
                "use_absolute_error":self._absolute_error,
                "attrs":self._attrs,
                }

    def concatenate(self, other:"EvalGridAxes", axis):
        """
        Concatenate a EvalGridAxes object with another one along an axis.
        I'm only really going to use it for the spatial axis, but I tried to
        write it to be more general. No promises it works though.
        """
        assert axis in self._axes, f"Concatenation axis {axis} must be one" + \
                f" of the preserved ones ({self._axes})"
        evr1 = self.get_results()
        evr2 = other.get_results()
        ## Assume by default all config comes from this object
        conc_data = deepcopy(evr1)
        conc_data.update({
                "avg":np.concatenate(
                    [evr1["avg"], evr2["avg"]], axis=axis),
                "var":np.concatenate(
                    [evr1["var"], evr2["var"]], axis=axis),
                })
        if all(not ix is None for ix in [evr1["indeces"],evr2["indeces"]]):
            conc_data["indeces"] = np.concatenate(
                    [evr1["indeces"], evr2["indeces"]], axis=0)
        if all(not ix is None for ix in [evr1["static"],evr2["static"]]):
            conc_data["static"] = np.concatenate(
                    [evr1["static"], evr2["static"]], axis=0)
            conc_data["static_int"] = np.concatenate(
                    [evr1["static_int"], evr2["static_int"]], axis=0)
        new_ev = EvalGridAxes()
        return new_ev.from_dict(conc_data)

    def to_pkl(self, pkl_path:Path, additional_attributes:dict={}):
        """
        Write the residual and state horizon error results to a pkl file

        :@param pkl_path: Path to a non-existing pkl path to dump results to.
        :@param additional_attributes: Dict of additional information to
            include alongside the horizon error distribution data. If any of
            the keys match existing auxillary attributes the new ones provided
            here will replace them.
        """
        pkl.dump(self.get_results(), pkl_path.open("wb"))

    def from_dict(self, config_dict):
        p = config_dict
        self._counts = p["counts"]
        self._sum = p["avg"]
        self._var_sum = p["var"]
        self._static = p["static"]
        self._static_int = p.get("static_int")
        self._time = p["time"]
        self._store_static = not self._static is None
        self._store_time = not self._time is None
        self._axes = p["axes"]
        self._feat_args_unevaluated = p["feat_args"]
        self._feat_args,self._feat_is_func = zip(
                *map(self._validate_feat_arg,p["feat_args"]))
        self._batch_count = p["batch_count"]
        self._indeces = p["indeces"]
        self._pred_coarseness = p.get("pred_coarseness", 1)
        self._coarse_reduce_str = p["coarse_reduce_func"]
        self._attrs = p["attrs"]
        self._absolute_error = p["use_absolute_error"]
        try:
            self._crf = self._rfuncs[self._coarse_reduce_str]
        except:
            raise ValueError(f"coarse_reduce_func must be in: " + \
                    "{self._rfuncs.keys()}")
        return self

    def from_pkl(self, pkl_path:Path):
        """ """
        return self.from_dict(pkl.load(pkl_path.open("rb")))

class EvalHorizon(Evaluator):
    def __init__(self, pred_coarseness=1, attrs={}):
        """ """
        self._pred_coarseness = pred_coarseness
        self._counts = None ## Number of samples included in sums
        self._es_sum = None ## Sum of state error wrt horizon
        self._er_sum = None ## Sum of residual error wrt horizon
        self._es_var_sum = None ## State error partial variance sum
        self._er_var_sum = None ## Residual error partial variance sum
        self._attrs = attrs ## additional attributes
        self._indeces = None

    @property
    def attrs(self):
        return self._attrs

    def add_batch(self, inputs, true_state, predicted_residual, indeces=None):
        """ """
        if not indeces is None and self._indeces is None:
            self._indeces = indeces
        ys,pr = true_state,predicted_residual
        ## the predicted state time series
        ps = ys[:,0,:][:,np.newaxis,:] + np.cumsum(pr, axis=1)
        ## Calculate the label residual from labels
        yr = ys[:,1:]-ys[:,:-1]
        ## Calculate the absolute error in the residual and state predictions
        es_abs = np.abs(ps - ys[:,1:,:])
        er_abs = np.abs(pr - yr)

        if self._counts is None:
            self._counts = es_abs.shape[0]
            self._es_sum = np.sum(es_abs, axis=0, dtype=np.float64)
            self._er_sum = np.sum(er_abs, axis=0, dtype=np.float64)
            self._es_var_sum = np.sum(
                    (es_abs-self._es_sum/self._counts)**2,
                    axis=0, dtype=np.float64)
            self._er_var_sum = np.sum(
                    (er_abs-self._er_sum/self._counts)**2,
                    axis=0, dtype=np.float64)
        else:
            self._counts += es_abs.shape[0]
            self._es_sum += np.sum(es_abs, axis=0, dtype=np.float64)
            self._er_sum += np.sum(er_abs, axis=0, dtype=np.float64)
            self._es_var_sum += np.sum(
                    (es_abs - self._es_sum/self._counts)**2,
                    axis=0, dtype=np.float64)
            self._er_var_sum += np.sum(
                    (er_abs - self._er_sum/self._counts)**2,
                    axis=0, dtype=np.float64)
        return

    def add(self, other:"EvalHorizon"):
        """
        Add the state and residual error sums and counts of multiple
        EvalHorizon instances
        """
        hor1 = self.get_results()
        hor2 = other.get_results()
        ## Assume by default all config comes from this object
        new_data = deepcopy(hor1)
        sum_fields = [ "state_avg", "state_var", "residual_avg",
                "residual_var", "counts"]
        ## Update the added data with the summed field
        new_data.update({f:hor1[f]+hor2[f] for f in sum_fields})
        if all(not ix is None for ix in [hor1["indeces"], hor2["indeces"]]):
            new_data["indeces"] = np.concatenate(
                    [hor1["indeces"], hor2["indeces"]], axis=0)
        return EvalHorizon().from_dict(new_data)

    def get_results(self):
        """ """
        return {
                "state_avg":self._es_sum,
                "state_var":self._es_var_sum,
                "residual_avg":self._er_sum,
                "residual_var":self._er_var_sum,
                "counts":self._counts,
                "indeces":self._indeces,
                "pred_coarseness":self._pred_coarseness,
                "attrs":self._attrs,
                }

    def to_pkl(self, pkl_path:Path, additional_attributes:dict={}):
        """
        Write the residual and state horizon error results to a pkl file

        :@param pkl_path: Path to a non-existing pkl path to dump results to.
        :@param additional_attributes: Dict of additional information to
            include alongside the horizon error distribution data. If any of
            the keys match existing auxillary attributes the new ones provided
            here will replace them.
        """
        pkl.dump(self.get_results(), pkl_path.open("wb"))

    def from_dict(self, config_dict):
        """ """
        p = config_dict
        self._counts = p["counts"]
        self._es_sum = p["state_avg"]
        self._er_sum = p["residual_avg"]
        self._es_var_sum = p["state_var"]
        self._er_var_sum = p["residual_var"]
        self._indeces = p["indeces"]
        self._pred_coarseness = p.get("pred_coarseness", 1)
        self._attrs = p["attrs"]
        return self

    def from_pkl(self, pkl_path:Path):
        """ """
        return self.from_dict(pkl.load(pkl_path.open("rb")))

'''
class EvalTemporal(Evaluator):
    def __init__(self, use_absolute_error=False, horizon_limit=None, attrs={}):
        """ """
        self._doy_r = None ## day of year residual error
        self._doy_s = None ## day of year static error
        self._doy_c = None ## day of year counts
        self._tod_r = None ## time of day residual error
        self._tod_s = None ## time of day static error
        self._tod_c = None ## time of day counts
        self._indeces = None
        self.absolute_error = use_absolute_error
        self.horizon_limit = horizon_limit
        self._attrs = attrs

    @property
    def attrs(self):
        return self._attrs

    def add_batch(self, inputs, true_state, predicted_residual, indeces=None):
        if not indeces is None and self._indeces is None:
            self._indeces = indeces
        (_,_,_,_,th),ys,pr = inputs,true_state,predicted_residual
        ps = ys[:,0,:][:,np.newaxis,:] + np.cumsum(pr, axis=1)
        yr = ys[:,1:]-ys[:,:-1]

        ## Once the prediction shape is known declare the derived arrays
        if self._doy_r is None:
            self._doy_r = np.zeros((366, pr.shape[-1]))
            self._doy_s = np.zeros((366, pr.shape[-1]))
            self._doy_c = np.zeros((366, pr.shape[-1]), dtype=np.uint)
            self._tod_r = np.zeros((24, pr.shape[-1]))
            self._tod_s = np.zeros((24, pr.shape[-1]))
            self._tod_c = np.zeros((24, pr.shape[-1]), dtype=np.uint)

        times = list(map(
            datetime.fromtimestamp,
            th.astype(np.uint)[:,:self.horizon_limit].reshape((-1,))
            ))
        ## Times are reported exactly on the hour, but float rounding can cause
        ## some to be above or below. Add a conditional to account for this.
        tmp_tods = np.array([
            (t.hour+1 if t.minute >= 30 else t.hour)%24 for t in times
            ])
        tmp_doys = np.array([t.timetuple().tm_yday-1 for t in times])

        es = ps - ys[:,1:]
        er = pr - yr
        if self.absolute_error:
            es,er = map(np.abs,(es,er))
        es = es[:,:self.horizon_limit].reshape((-1, es.shape[-1]))
        er = er[:,:self.horizon_limit].reshape((-1, er.shape[-1]))

        for i in range(len(times)):
            self._doy_s[tmp_doys[i]] += es[i]
            self._doy_r[tmp_doys[i]] += er[i]
            self._doy_c[tmp_doys[i]] += 1
            self._tod_s[tmp_tods[i]] += es[i]
            self._tod_r[tmp_tods[i]] += er[i]
            self._tod_c[tmp_tods[i]] += 1

    def add(self, other:"EvalTemporal"):
        """
        Add the state and residual error sums and counts of multiple
        EvalTemporal instances
        """
        hor1 = self.get_results()
        hor2 = other.get_results()
        ## Assume by default all config comes from this object
        new_data = deepcopy(hor1)
        sum_fields = [ "doy_state", "doy_residual", "doy_counts",
                "tod_state", "tod_residual", "tod_counts", ]
        ## Update the added data with the summed field
        new_data.update({f:hor1[f]+hor2[f] for f in sum_fields})
        if all(not ix is None for ix in [hor1["indeces"], hor2["indeces"]]):
            new_data["indeces"] = np.concatenate(
                    [hor1["indeces"], hor2["indeces"]], axis=0)
        return EvalTemporal().from_dict(new_data)

    def get_results(self):
        return {
                "doy_state":self._doy_s,
                "doy_residual":self._doy_r,
                "doy_counts":self._doy_c,
                "tod_state":self._tod_s,
                "tod_residual":self._tod_r,
                "tod_counts":self._tod_c,
                #"feats":pred_dict["pred_feats"],
                "indeces":self._indeces,
                "absolute_error":self.absolute_error,
                "horizon_limit":self.horizon_limit,
                "attrs":self._attrs,
                }

    def to_pkl(self, pkl_path:Path, additional_attributes:dict={}):
        """
        Write the residual and state horizon error results to a pkl file

        :@param pkl_path: Path to a non-existing pkl path to dump results to.
        :@param additional_attributes: Dict of additional information to
            include alongside the horizon error distribution data. If any of
            the keys match existing auxillary attributes the new ones provided
            here will replace them.
        """
        pkl.dump(self.get_results(), pkl_path.open("wb"))

    def from_dict(self, config_dict):
        """ """
        p = config_dict
        self._doy_s = p["doy_state"]
        self._doy_r = p["doy_residual"]
        self._doy_c = p["doy_counts"]
        self._tod_s = p["tod_state"]
        self._indeces = p["indeces"]
        self._tod_r = p["tod_residual"]
        self._tod_c = p["tod_counts"]
        self.absolute_error = p["absolute_error"]
        self.horizon_limit = p["horizon_limit"]
        self._attrs = p["attrs"]
        return self

    def from_pkl(self, pkl_path:Path):
        """ """
        return self.from_dict(pkl.load(pkl_path.open("rb")))
'''

class EvalStatic(Evaluator):
    def __init__(self, soil_idxs=None, use_absolute_error=False, attrs={}):
        """"
        Extracts a combination matrix of surface types and soil textures
        for state and residual bias or residual error

        :@param soil_idxs: feature indeces for the (sand, silt, clay)
            components of the static array (in the above order of decreasing
            particle size).
        """
        ## Soil components to index mapping. Scuffed and slow, I know, but
        ## unfortunately I didn't store integer types alongside sequences,
        ## and it's too late to turn back now :(
        self._soil_mapping = list(map(
            lambda a:np.array(a, dtype=np.float32),
            [
                [0.,   0.,   0.  ],
                [0.92, 0.05, 0.03],
                [0.82, 0.12, 0.06],
                [0.58, 0.32, 0.1 ],
                [0.17, 0.7 , 0.13],
                [0.1 , 0.85, 0.05],
                [0.43, 0.39, 0.18],
                [0.58, 0.15, 0.27],
                [0.1 , 0.56, 0.34],
                [0.32, 0.34, 0.34],
                [0.52, 0.06, 0.42],
                [0.06, 0.47, 0.47],
                [0.22, 0.2 , 0.58],
                ]
            ))
        self._counts = np.zeros((14,13))
        self._err_res = None
        self._err_state = None
        self.absolute_error = use_absolute_error
        self._indeces = None
        self.soil_idxs = soil_idxs
        self._attrs = attrs

    @property
    def attrs(self):
        return self._attrs

    def add_batch(self, inputs, true_state, predicted_residual, indeces=None):
        """ """
        (_,_,s,si,_),ys,pr = inputs,true_state,predicted_residual
        if not indeces is None and self._indeces is None:
            self._indeces = indeces
        if self._err_res is None:
            self._err_res = np.zeros((14,13,pr.shape[-1]))
            self._err_state = np.zeros((14,13,pr.shape[-1]))

        ## the predicted state time series
        ps = ys[:,0,:][:,np.newaxis,:] + np.cumsum(pr, axis=1)
        ## Calculate the label residual from labels
        yr = ys[:,1:]-ys[:,:-1]
        ## Calculate the error in the residual and state predictions
        es = ps - ys[:,1:,:]
        er = pr - yr

        ## Average the error over the full horizon
        if self.absolute_error:
            es = np.abs(es)
            er = np.abs(er)
        es_avg = np.average(es, axis=1)
        er_avg = np.average(er, axis=1)

        soil_texture = s[...,self.soil_idxs]
        for i,soil_array in enumerate(self._soil_mapping):
            ## Get a boolean mask
            m_this_soil = np.isclose(soil_texture, soil_array).all(axis=1)
            if not np.any(m_this_soil):
                continue
            es_avg_subset = es_avg[m_this_soil]
            er_avg_subset = er_avg[m_this_soil]
            si_subset = si[m_this_soil]
            ## Convert the one-hot encoded vegetation vectors to indeces
            si_idxs = np.argwhere(si_subset)[:,1]
            for j in range(si_idxs.shape[0]):
                self._err_res[si_idxs[j],i] += er_avg_subset[j]
                self._err_state[si_idxs[j],i] += es_avg_subset[j]
                self._counts[si_idxs[j],i] += 1

    def add(self, other:"EvalStatic"):
        """
        Add the state and residual error sums and counts of multiple
        EvalHorizon instances
        """
        stat1 = self.get_results()
        stat2 = other.get_results()
        ## Assume by default all config comes from this object
        new_data = deepcopy(stat1)
        sum_fields = ["err_state", "err_residual", "counts"]
        ## Update the added data with the summed field
        new_data.update({f:stat1[f]+stat2[f] for f in sum_fields})
        if all(not ix is None for ix in [stat1["indeces"], stat2["indeces"]]):
            new_data["indeces"] = np.concatenate(
                    [stat1["indeces"], stat2["indeces"]], axis=0)
        return EvalStatic().from_dict(new_data)

    def get_results(self):
        """ Collect data from batches into a dict """
        return {
            "err_state":self._err_state,
            "err_residual":self._err_res,
            "counts":self._counts,
            "soil_idxs":self.soil_idxs,
            "indeces":self._indeces,
            "use_absolute_error":self.absolute_error,
            "attrs":self._attrs,
            #"feats":pred_dict["pred_feats"],
            }
    def to_pkl(self, pkl_path:Path, additional_attributes:dict={}):
        """
        Serialize the bulk data and attributes of this instance as a pkl
        """
        pkl.dump(self.get_results(), pkl_path.open("wb"))

    def from_dict(self, config_dict):
        """ """
        p = config_dict
        self._err_state = p["err_state"]
        self._err_res = p["err_residual"]
        self._counts = p["counts"]
        self._indeces = p["indeces"]
        self.soil_idxs = p["soil_idxs"]
        self.absolute_error = p["use_absolute_error"]
        self._attrs = p["attrs"]
        return self

    def from_pkl(self, pkl_path:Path):
        """
        Load the bulk data and attributes of a EvalStatic instance from a pkl
        file that has already been generated
        """
        return self.from_dict(pkl.load(pkl_path.open("rb")))

class EvalJointHist(Evaluator):
    def __init__(self, ax1:tuple, ax2:tuple, cov:tuple, dataset_feats:dict,
                 hist_bounds:dict, hist_resolution:int=512, derived_feats={},
                 use_absolute_error=False, ignore_nan=False, pred_coarseness=1,
                 coarse_reduce_func="mean", attrs={}):
        """
        Initialize a histogram evaluator with 2 axes defined by tuples

        Specify a feature axis with a 2-tuple
        (
            (data_source, feat_idx),
            (val_min, val_max, num_bins)
        )

        Or specify a functional axis with 3-tuple (args, func, bounds) like:
        (
            ((data_source, feat_idx), (data_source, feat_idx), ...),
            func_or_lambda_str,
            (val_min, val_max, num_bins),
        )
        such that the first sub-tuple of (data_source, feat_idx) pairs provides
        the arguments for func_or_lambda_str

        data_source must be one of:
        {"horizon", "static", "true_res", "pred_res", "err_res",
         "true_state", "pred_state", "err_state"}

        :@param ax1_args: First (vertical) axis arguments as specified above
        :@param ax2_args: Second (horizontal) axis arguments as specified above
        :@param covariate_feature: Optional 2-tuple identifying
            (data_source, feat_idx) feature for which to capture an average
            value corresponding to each 2D value bin described by the axes
        :@param use_absolute_error: If True, calculate histograms based on
            the absolute value of error rather than the actual magnitude
        :@param ignore_nan: If True, NaN values encountered after a derived
            axis feature calculation will be ignored when histograms are binned
        :@param pred_coarseness: Include the model's coarseness argument so
            that arguments to derived axis feat calculations have the same
            number of elements along the sequence axis.
        :@param coarse_reduce_func: If functions output coarsened predictions,
            and axis arguments implement a function that uses horizon input
            data, a function must be used to reduce the inputs to the coarser
            resolution. Current choices include "min", "mean", and "max"
        """
        self._ax1 = ax1
        self._ax2 = ax2
        self._cov = cov
        self._ds_feats = dataset_feats
        self._derived = derived_feats

        ## if the feat is stored, get its index. Set to None if derived
        self._ax1_isfunc,self._ax1_ix,self._ax1_args,self._ax1_func = \
                EvalJointHist._validate_axis_args(self._ax1)
        self._ax2_isfunc,self._ax2_ix,self._ax2_args,self._ax2_func = \
                EvalJointHist._validate_axis_args(self._ax2)
        if not self._cov is None:
            self._cov_isfunc,self._cov_ix,self._cov_args,self._cov_func = \
                    EvalJointHist._validate_axis_args(self._rov)
        else:
            self._cov_is_func = None
            self._cov_ix = None

        ## Validate axis arguments and evaluate any string lambda functions
        self.ignore_nan = ignore_nan
        self.absolute_error = use_absolute_error
        self._attrs = attrs
        self._counts = None
        self._cov_sum = None
        self._indeces = None
        self._coarse_reduce_str = coarse_reduce_func
        self._pred_coarseness = pred_coarseness
        self._rfuncs = {"min":np.amin, "mean":np.average, "max":np.amax}
        try:
            self._crf = self._rfuncs[coarse_reduce_func]
        except:
            raise ValueError(f"coarse_reduce_func must be in: " + \
                    "{self._rfuncs.keys()}")

    @property
    def attrs(self):
        return self._attrs

    @staticmethod
    def _validate_axis_args(axis_args):
        """
        """
        is_func = axis_args[0]=="derived"
        if is_func:
            axis_ix = None
            axis_args = self._derived[axis_args[1]][0]
            axis_func = eval(self._derived[axis_args[1]][1])
        else:
            axis_ix = self._ds_feats[axis_args[0]].index(axis_args[1])
            axis_args = None
            axis_func = None
        return is_func,axis_ix,axis_args,axis_func

    def add_batch(self, inputs, true_state, predicted_residual, indeces=None):
        """ Update the partial evaluation data with a new batch of samples """
        (_,h,s,_,_),ys,pr = inputs,true_state,predicted_residual
        if not indeces is None and self._indeces is None:
            self._indeces = indeces
        ## the predicted state time series
        ps = ys[:,0,:][:,np.newaxis,:] + np.cumsum(pr, axis=1)
        ## Calculate the label residual from labels
        yr = ys[:,1:]-ys[:,:-1]
        ## Calculate the error in the residual and state predictions
        es = ps - ys[:,1:,:]
        er = pr - yr
        if self.absolute_error:
            es = np.abs(es)
            er = np.abs(er)
        ## Make a dict of the data arrays to make extraction easier
        if self._pred_coarseness != 1:
            b,_,f = h.shape
            h = h.reshape(h.shape[0],-1,self._pred_coarseness,h.shape[-1])
            h = self._crf(h, axis=2)
        data = {
                "horizon":h, "static":s,
                "true_res":yr, "pred_res":pr, "err_res":er,
                "true_state":ys[:,1:], "pred_state":ps, "err_state":es
                }
        ## Collect arguments and evaluate the method if ax1 is functional
        if self._ax1_is_func:
            args = [data[s][...,ix] for s,ix in self._ax1_args[0]]
            ax1 = self._ax1_args[1](*args)
        ## Otherwise just extract the data from the proper source array
        else:
            s,ix = self._ax1_args[0]
            ax1 = data[s][...,ix]
        ## Collect arguments and evaluate the method if ax2 is functional
        if self._ax2_is_func:
            args = [data[s][...,ix] for s,ix in self._ax2_args[0]]
            ax2 = self._ax2_args[1](*args)
        ## Otherwise just extract the data from the proper source array
        else:
            s,ix = self._ax2_args[0]
            ax2 = data[s][...,ix]
        if self._cov_feat != None:
            s,ix = self._cov_feat
            cov = data[s][...,ix]
        else:
            cov = None

        ## extract bounds from the axis arguments
        ax1_min,ax1_max,ax1_bins = self._ax1_args[-1]
        ax2_min,ax2_max,ax2_bins = self._ax2_args[-1]

        ## declare the counts array if it hasn't already been declared
        if self._counts is None:
            self._counts = np.zeros((ax1_bins,ax2_bins), dtype=np.uint64)
            if self._cov_feat != None:
                self._cov_sum = np.zeros((ax1_bins,ax2_bins), dtype=np.float64)
        ## Cast the (batch,sequence) arrays for this feature as integer indeces
        ## corresponding to their value bin, and flatten them into a 1d array.
        ax1_idxs = np.reshape(
                self._norm_to_idxs(ax1, ax1_min, ax1_max, ax1_bins), (-1,))
        ax2_idxs = np.reshape(
                self._norm_to_idxs(ax2, ax2_min, ax2_max, ax2_bins), (-1,))

        m_valid = None
        if self.ignore_nan:
            m_valid = np.logical_and(
                    np.isfinite(ax1_idxs),
                    np.isfinite(ax2_idxs)
                    )
            ax1_idxs = ax1_idxs[m_valid]
            ax2_idxs = ax2_idxs[m_valid]
        if self._cov_feat != None:
            cov = np.reshape(cov, (-1,))
            if self.ignore_nan:
                cov = cov[m_valid]
        ## Loop since fancy indexing doesn't accumulate repetitions
        for i in range(ax1_idxs.size):
            self._counts[ax1_idxs[i],ax2_idxs[i]] += 1
            if self._cov_feat != None:
                self._cov_sum[ax1_idxs[i],ax2_idxs[i]] += cov[i]

    @staticmethod
    def _norm_to_idxs(A:np.array, mins, maxs, num_bins):
        A = (np.clip(A, mins, maxs) - mins) / (maxs - mins)
        A = np.clip(np.floor(A * num_bins).astype(int), 0, num_bins-1)
        return A

    def add(self, other:"EvalJointHist"):
        """
        Add the state and residual error sums and counts of multiple
        EvalHorizon instances
        """
        ejh1 = self.get_results()
        ejh2 = other.get_results()
        ## Assume by default all config comes from this object
        new_data = deepcopy(ejh1)
        sum_fields = ["covariate_sum", "counts"]
        ## Update the added data with the summed field
        new_data.update(
                {f:ejh1[f]+ejh2[f] \
                        if (not ejh1[f] is None and not ejh2[f] is None) \
                        else None for f in sum_fields}
                )
        if all(not ix is None for ix in [ejh1["indeces"], ejh2["indeces"]]):
            new_data["indeces"] = np.concatenate(
                    [ejh1["indeces"], ejh2["indeces"]], axis=0)
        return EvalJointHist().from_dict(new_data)

    def get_results(self):
        """
        Collect the partial data from supplied batches into a dict of results
        formatted as the complete evaluation data this class produces.
        """
        return {
                "ax1_args":self._ax1_args_unevaluated,
                "ax2_args":self._ax2_args_unevaluated,
                "covariate_feature":self._cov_feat,
                "covariate_sum":self._cov_sum,
                "counts":self._counts,
                "use_absolute_error":self.absolute_error,
                "indeces":self._indeces,
                "ignore_nan":self.ignore_nan,
                "pred_coarseness":self._pred_coarseness,
                "coarse_reduce_func":self._coarse_reduce_str,
                "attrs":self._attrs,
                }

    def to_pkl(self, pkl_path:Path, additional_attributes:dict={}):
        """
        Serialize the bulk data and attributes of this instance as a pkl
        """
        pkl.dump(self.get_results(), pkl_path.open("wb"))

    def from_dict(self, config_dict):
        """ """
        p = config_dict
        self._ax1_args_unevaluated = p["ax1_args"]
        self._ax2_args_unevaluated = p["ax2_args"]
        self._ax1_args,self._ax1_is_func = self._validate_axis_args(
                self._ax1_args_unevaluated)
        self._ax2_args,self._ax2_is_func = self._validate_axis_args(
                self._ax2_args_unevaluated)
        self.absolute_error = p["use_absolute_error"]
        self.ignore_nan = p["ignore_nan"]
        self._counts = p["counts"]
        self._indeces = p.get("indeces", None)
        self._cov_sum = p["covariate_sum"]
        self._pred_coarseness = p["pred_coarseness"]
        self._coarse_reduce_str = p["coarse_reduce_func"]
        self._cov_feat = p["covariate_feature"]
        self._attrs = p["attrs"]
        try:
            self._crf = self._rfuncs[self._coarse_reduce_str]
        except:
            raise ValueError(f"coarse_reduce_func must be in: " + \
                    "{self._rfuncs.keys()}")
        return self

    def from_pkl(self, pkl_path:Path):
        """ """
        return self.from_dict(pkl.load(pkl_path.open("rb")))

EVALUATORS = {
        #"EvalHorizon":EvalHorizon,
        "EvalTemporal":EvalTemporal,
        #"EvalStatic":EvalStatic,
        #"EvalEfficiency":EvalEfficiency,
        #"EvalJointHist":EvalJointHist,
        #"EvalGridAxes":EvalGridAxes,
        }
REDUCE_FUNCS = {"min":np.amin, "mean":np.average, "max":np.amax, "sum":np.sum}

if __name__=="__main__":
    pass
