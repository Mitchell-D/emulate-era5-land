import numpy as np
import json
import torch
import h5py
from pathlib import Path
from time import perf_counter

from emulate_era5_land.models import get_model_from_config
from emulate_era5_land.helpers import _parse_feat_idxs,_calc_feat_array
from emulate_era5_land.helpers import move_to_device,take_index_from_nested

class PredictionDataset(torch.utils.data.IterableDataset):
    """
    Dataset generator that loads a model based on its training configuration,
    runs the model, and yields the inference results alongside the inputs.
    """
    def __init__(self, model_path:Path, use_dataset:str=None,
            normalized_outputs=None, config_override:dict=None, device=None,
            include_integrated_outputs:False, output_device=None, debug=False):
        """
        :@param model_path: Path to the model weights, which is assumed to be
            inside the 'model dir' containing the configuration json
        :@param use_dataset: String key of one of the datasets configured under
            'data' indicating which of the datasets in the config to base the
            configuration after
        :@param normalized_outputs: If True, outputs remain normalized as they
            were during training. Otherwise, they are rescaled to normal data
            coordinates before being returned.
        :@param config_override: dict containing a tree of substitutions to
            the original configuration. Only substitutions to the 'feats',
            'data', and 'seed' fields are supported; it doesn't make sense to
            modify any of the others after training. Use the feats to add
            auxiliary features or change the window/horizon size, and use the
            data field to configure an evaluation generator.
        :@param device: Device on which to compute the model results
        :@param include_integrated_outputs: If True, features in the prediction
            and target arrays that are differentiated and don't have an
            integrated feature explicitly included will be integrated and
            concatenated to the returned array, and the integrated feat labels
            will be reflected in the feat dict.
        :@param output_device: If defined, tensors are sent to this device
            before being returned. If None, the device is used
        """
        if device is None:
            self._device = torch.device(
                    "cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device
        if output_device is None:
            self._out_device = device
        else:
            self._out_device = output_device
        self._signature = {
            "model_path":model_path,
            "use_dataset":use_dataset,
            "normalized_outputs":normalized_outputs,
            "config_override":config_override,
            "device":device,
            "include_integrated_outputs":include_integrated_outputs,
            "output_device":output_device,
            "debug":debug,
            }
        self._norm_out = normalized_outputs
        self._model_dir = model_path.parent
        self._model_path = model_path
        og_config_path = self._model_dir.joinpath(
                f"{self._model_dir.name}_config.json")
        assert self._model_path.exists(), self._model_path
        assert og_config_path.exists(), og_config_path
        self._og_config = json.load(og_config_path.open("r"))

        co = config_override

        ## update the config with the overrides
        self._config = {
            "feats":{
                **self._og_config["feats"],
                **co.get("feats", {})
                },
            "data":{
                use_dataset:{
                    **self._og_config["data"].get(use_dataset, {}),
                    **co["data"].get(use_dataset, {})
                    },
                },
            "metrics":self._og_config.get("metrics"),
            "model":{
                "type":self._og_config["model"]["type"],
                "args":{
                    **self._og_config["model"]["args"],
                    ## Determines whether model preds will be normalized
                    ## inside the model code; doesn't affect other data.
                    **({"normalized_outputs":self._norm_out}
                       if not self._norm_out is None else {})
                    },
                },
            "setup":self._og_config.get("setup"),
            "seed":co.get("seed", self._og_config.get("seed")),
            "name":self._og_config.get("name"),
            "notes":self._og_config.get("notes")
            }

        assert self._config["data"][use_dataset],f"{use_dataset} not found!"

        self._ds = get_datasets_from_config(self._config)[use_dataset]
        self._model = get_model_from_config(self._config)
        self._model.load_state_dict(
                torch.load(
                    self._model_path,
                    weights_only=True,
                    map_location=torch.device(self._device),
                    )
                )
        self._model = self._model.to(self._device)
        self._model.eval()

        self._dl = iter(torch.utils.data.DataLoader(
                dataset=self._ds,
                batch_size=self._config["data"][use_dataset]["batch_size"],
                num_workers=self._config["data"][use_dataset]["num_workers"],
                prefetch_factor= \
                        self._config["data"][use_dataset]["prefetch_factor"],
                worker_init_fn=stsd_worker_init_fn,
                ))
        self._cur_ix = None

        pf_args = {
                "w":self._ds.feats["window"],
                "h":self._ds.feats["horizon"],
                "y":self._ds.feats["target"],
                ## ew yea ik
                "yinit":[fl.split(" ")[-1]
                    for fl in self._ds.feats["target"]]
                }

        self._norms = {}
        for k,v in pf_args.items():
            n = []
            for fl in v:
                if fl not in self._og_config["feats"]["norm_coeffs"].keys():
                    print(f"NORM COEFF NOT FOUND FOR {fl}")
                n.append(self._og_config["feats"]["norm_coeffs"].get(fl,(0,1)))
            self._norms[k] = torch.Tensor(n, device=self._out_device).T

        self._norms["s"] = torch.Tensor([
            self._ds.signature["norm_coeffs"].get(fl,(0,1))
            for fl in self._ds.fefats["static"]
            ], device=self._out_device).T

        self._integ_out_feats = []
        self._integ_out_ixs = []
        if self._signature["include_integrated_outputs"]:
            for ix,f in enumerate(self._ds.feats["target"]):
                if f.split(" ")[0]=="diff":
                    assert len(f.split(" "))==2
                    base_feat = f.split(" ")[-1]
                    if base_feat in self._ds.feats["target"]:
                        continue
                    self._integ_out_feats.append(base_feat)
                    self._integ_out_ixs.append(ix)

        self._debug = debug

    @property
    def feats(self):
        """
        Return a dict mapping array names to their feature labels, which is
        organized in the same way as the returned data.
        """
        return {
            **self._ds.feats,
            "target":self._ds_feats["target"] + self._integ_out_feats,
            "prediction":self._ds.feats["target"] + self._integ_out_feats,
            }

    @property
    def signature(self):
        """ expose dict of all input parameters for serialization / re-init """
        return self._signature


    def _replenish_batch(self):
        """ """
        #inputs,(y,),aux = next(self._dl)
        batch = next(self._dl)
        #w,h,s,si,init = inputs
        #p = self._model(w, h, s, si, target=y, device=self._device)
        batch["prediction"] = self._model(
            batch["window"], batch["horizon"], batch["static"],
            batch["static-int"], target=batch["target"], device=self._device
            )
        self._cur_ix = 0

        batch = move_to_device(batch, self._out_device)

        ## if the model expects normalized inputs, but de-normalized outputs
        ## are wanted, rescale the input values after running the model.
        ## There are other cases here that prob should be handled but nah
        norm_in = self._og_config["model"]["args"]["normalized_inputs"]
        if not self._norm_out and norm_in:
            batch["window"] = batch["window"] \
                    * self._norms["w"][1] + self._norms["w"][0]
            batch["horizon"] = batch["horizon"] \
                    * self._norms["h"][1] + self._norms["h"][0]
            batch["static"] = batch["static"] \
                    * self._norms["s"][1] + self._norms["s"][0]
            batch["target-init"] = batch["target-init"] \
                    * self._norms["yinit"][1] + self._norms["yinit"][0]
            batch["target"] = batch["target"] \
                    * self._norms["y"][1] + self._norms["y"][0]

        ## integrate the outputs if requested.
        if self._signature["include_integrated_outputs"]:
            if self._norm_out:
                raise ValueError(
                    "output integration is not currently supported for " + \
                    "normalized outputs. Have fun implementing it!")
            batch["target"] = torch.cat([
                batch["target"],
                batch["target-init"] + torch.cumsum(
                    batch["target"][...,self._integ_out_ixs], axis=1)
                ], axis=-1)
            batch["prediction"] = torch.cat([
                batch["prediction"],
                batch["target-init"] + torch.cumsum(
                    batch["prediction"][...,self._integ_out_ixs], axis=1)
                ], axis=-1)

        #self._cur_sample = ((w,h,s,si,init), (y,), aux, (p,))
        self._cur_sample = batch
        self._cur_bs = batch["window"].shape[0]

    def __next__(self):
        """ """
        if self._cur_ix is None or self._cur_ix==self._cur_bs-1:
            if self._debug:
                _t0 = perf_counter()
            self._replenish_batch()
            if self._debug:
                _tf = perf_counter()
                print(f"Batch replenish: {_tf-_t0}")
        cur = take_index_from_nested(self._cur_sample, self._cur_ix, axis=0)
        self._cur_ix += 1
        return cur

    def __iter__(self):
        """ """
        return self


class SparseTimegridSampleDataset(torch.utils.data.IterableDataset):
    """
    PyTorch IterableDataset implementation with the ability to multiprocess
    over sparse sampling of 1d time series
    """
    def __init__(self, timegrids:list, window_feats, horizon_feats,
            target_feats, static_feats, static_int_feats, derived_feats={},
            aux_dynamic_feats=[], aux_static_feats=[], static_embed_maps={},
            window_size=24, horizon_size=24, norm_coeffs={}, shuffle=True,
            seed=None, sample_cutoff=1, sample_across_files=True,
            sample_under_cutoff=True, sample_separation=1, random_offset=True,
            chunk_pool_count=4, buf_size_mb=1024, buf_slots=128, buf_policy=0,
            out_dtype="f4", debug=False):
        """
        The initializer method is responsible for:

        1. open pointers to timegrid datasets
        2. make a list of all chunks in the file, and determine which of them
           cross over into an adjacent timegrid. Ignore file-spanning chunks
           if the needed file isn't provided, or not sampling across files.
        3. shuffle chunks within each file and randomly drop a fraction of
           them if sample\_cutoff is provided.
        4. globally shuffle all chunks across files
        5. If requested, calculate random temporal offsets for each pixel
        6. group the chunks into pools that will be extracted, shuffled,
           and returned together during sample generation.
        7. get info needed to extract stored and derived features, and
           identify differentiated features in the window, horizon, target,
           or auxiliary dynamic outputs
        8. organize normalization coefficients into vectors.


         o  o  o  o  o  o  o  o  o  o  o  o  o  o  o  o  o  o  o  o  o  o
           | |-1--2--3--4--5--6--7--8--9||1--2--3--4--5--6--7--8--9|
            |              |                          |
          buffer         window                     horizon


        :@param timegrids: Timegrid files, MUST be ordered chronologically so
            that contiguous samples can be collected across files.
        :@param *_feats: Ordered lists of strings corresponding to the stored
            or derived feature labels included in the sample outputs.
        :@param derived_feats: dict mapping unique derived feature labels to a
            3-tuple (dynamic_args, static_args, lambda_str) where the args are
            each tuples of existing dynamic/static labels, and lambda_str
            contains a string-encoded function taking 2 arguments
            (dynamic,static) of tuples containing the corresponding arrays, and
            returns the subsequent new feature after calculating it based on
            the arguments. These will be invoked if the new derived feature
            label appears in one of the feature lists.
        :@param aux_dynamic_feats: Optional ordered list of strings
            corresponding to stored or derived feature labels that will be
            returned separately from the model inputs/targets. The auxilliary
            array will span the entire sample time period including the window
            and horizon. Use these data variables for analysis, closure, etc.
        :@param aux_static_feats: Optional ordered list of strings
            corresponding to stored static feature labels that will be returned
            separately from model inputs. You can use these for analysis,
            returning indices, biasing the loss function, etc.
        :@param static_embed_maps: Dict mapping a stored static feature label
            to a list of possible integer values. The order of the listed
            values determines the one-hot encoding vector position of each.
        :@param window_size: Number of input steps prior to the first
            prediction step, used to initialize the model.
        :@param horizon_size: Number of covariate input and prediction steps.
        :@param shuffle: If True, samples will be shuffled by chunk rather than
            being returned chronologically as stored.
        :@param seed: Random number generator seed
        :@param norm_coeffs: Dict mapping feature names to a 2-tuple
            (mean, stddev) for normalizing input variables to a unit gaussian.
        :@param sample_across_files: If True, samples that overlap the final
            time step in a timegrid file will be completed by reading the first
            timesteps for the same pixel in the subsequent timegrid in the
            provided list, unless it is the last one. Otherwise if False, the
            overlapping samples are discarded. If this option is True, it is
            assumed that the provided timegrids are strictly chronological.
        :@param sample_cutoff: Cutoff ratio of chunks per timegrid over or
            under which samples will be extracted. The subset is drawn after
            shuffling is (optionally) applied
        :@param sample_under_cutoff: If True, samples are drawn "under" the
            cutoff threshold. For example, if the threshold is .7 and this
            value is True, the first 70% of chunks will be selected, and if it
            is False, the final 30% will be selected.
        :@param sample_separation: Number of timesteps between sample
            extractions for any particular spatial pixel
        :@param random_offset: If True, each spatial pixel is assigned a
            random initial time step offset between 0 and sample_separation,
            which prevents overfitting to certain sequences/diurnal patterns.
        :@param chunk_pool_count: Sets the number of chunks that are
            simultaneously loaded and shuffled together by each STSD
        :@param buf_size_mb: Buffer size to use for caching hdf5 chunks.
        :@param buf_slots: Number of individual chunks that can be stored in
            the cache at once.
        :@param buf_policy: Closer to zero, the least recently accessed chunk
            will be released when needed, and closer to one the least recently
            *fully utilized* chunk will be released. Closer to one is more
            efficient if it is rare that the same chunk will be accessed again.
        :@param out_dtype: string-serialized dtype of sample outputs, ex "f2"
        :@param debug: If True, print general info and processing time data.

        --( to implement )--

        :@param static_conditions:
        """
        super(SparseTimegridSampleDataset).__init__()
        self._tgs = {}
        self._chunks = []
        self._tgs_ordered = timegrids
        self._cpc = chunk_pool_count
        self._rng = None
        self._shuffle = shuffle
        self._roffset = random_offset
        self._smp_sep = sample_separation

        self._w_feats = window_feats
        self._h_feats = horizon_feats
        self._y_feats = target_feats
        self._s_feats = static_feats
        self._si_feats = static_int_feats
        self._ad_feats = aux_dynamic_feats
        self._as_feats = aux_static_feats
        self._derived_feats = derived_feats
        self._w_size = window_size
        self._h_size = horizon_size
        self._si_embed_maps = {
                si_label:{b:a for a,b in enumerate(si_map)}
                for si_label,si_map in static_embed_maps.items()}
        self._coeffs = norm_coeffs

        self._cur_samples = None
        self._cur_sample_ix = None
        self._cur_sample_count = None
        self._seed = seed
        self._sample_across_files = sample_across_files
        self._sample_cutoff = sample_cutoff
        self._sample_under_cutoff = sample_under_cutoff
        self._buf_size_mb = buf_size_mb
        self._buf_slots = buf_slots
        self._buf_policy = buf_policy
        self._out_dtype = np.dtype(out_dtype)
        self._debug = debug

        ## include an additional element in case window diffs requested.
        self._smp_size = self._w_size + self._h_size + 1

        ## collect open file pointers, chunks, and labels for each timegrid
        for i,tg in enumerate(timegrids):
            tg_open = h5py.File(tg, "r", rdcc_nbytes=self._buf_size_mb*1024**2,
                    rdcc_nslots=self._buf_slots, rdcc_w0=self._buf_policy)
            self._tgs[tg] = {
                    "dynamic":tg_open["/data/dynamic"],
                    "static":tg_open["/data/static"],
                    "time":tg_open["/data/time"],
                    "dlabels":json.loads(
                        tg_open["data"].attrs["dynamic"]
                        )["flabels"],
                    "slabels":json.loads(
                        tg_open["data"].attrs["static"]
                        )["flabels"],
                    }

            ## make list of (tg_path, (tslice,pslice), across, next_tg_path)
            ## representing chunks within this timegrid. The boolean 'across'
            ## indicates whether chunk pixels overlap the subsequent timegrid
            ntimes = self._tgs[tg]["dynamic"].shape[0]
            cur_chunks = [
                    (tg, c[:2], ntimes-c[0].stop<self._smp_size,
                        None if i==len(timegrids)-1 else timegrids[i+1])
                    for c in tg_open["/data/dynamic"].iter_chunks()
                    ]

            ## never include overlapping chunks for the last file.
            cur_chunks = list(filter(
                    lambda c:not (c[2] and c[3] is None), cur_chunks))
            ## drop overlapping chunks if not sampling across files
            if not self._sample_across_files:
                cur_chunks = list(filter(lambda c:not c[2], cur_chunks))

            ## Shuffle the chunks for all files and subset them as requested
            if self._shuffle:
                if self._rng is None:
                    self._rng = np.random.default_rng(seed=self._seed)
                self._rng.shuffle(cur_chunks)
            else:
                self._rng = None

            ## subset above or below the provided threshold
            nc_under = int(len(cur_chunks)*self._sample_cutoff)
            if self._sample_under_cutoff:
                cur_chunks = cur_chunks[:nc_under]
            else:
                cur_chunks = cur_chunks[nc_under:]
            self._chunks += cur_chunks
            if self._debug:
                print(f"Extracting {len(cur_chunks)} chunks from {tg.name}")

        ## globally shuffle chunks
        if self._shuffle:
            self._rng.shuffle(self._chunks)

        ## calculate pixel offsets if requested
        npx = None
        for tg in self._tgs_ordered:
            if npx is None:
                npx = self._tgs[tg]["dynamic"].shape[1]
            else:
                assert npx == self._tgs[tg]["dynamic"].shape[1], tg
        ## offsets must be between zero and the total length of samples
        if self._roffset:
            if self._rng is None:
                self._rng = np.random.default_rng(seed=self._seed)
            self._offsets = self._rng.integers(0, self._smp_sep, npx)
        else:
            self._offsets = np.zeros(npx)

        ## group chunks into pools that are extracted and returned together
        nslices = len(self._chunks) // self._cpc + \
                int(bool(len(self._chunks) % self._cpc))
        self._pool_slices = [slice(i*self._cpc,(i+1)*self._cpc)
                for i in range(nslices)]

        ## make sure all timegrids have the same feats and ordering
        for tg in self._tgs_ordered[1:]:
            assert tuple(self._tgs[self._tgs_ordered[0]]["dlabels"]) \
                    == tuple(self._tgs[tg]["dlabels"]), tg
            assert tuple(self._tgs[self._tgs_ordered[0]]["slabels"]) \
                    == tuple(self._tgs[tg]["slabels"]), tg

        ## get idxs/funcs of the stored and derived dynamic feats of each type
        ## yinit is always the undifferentiated version of features.
        pf_args = {"w":self._w_feats, "h":self._h_feats,
                "y":self._y_feats, "ad":self._ad_feats,
                "yinit":[fl.split(" ")[-1] for fl in self._y_feats]}
        self._feat_info = {}
        self._diff_feats = {"w":[], "h":[], "y":[], "ad":[]}
        for k,v in pf_args.items():
            ## want to make a full grammar out of the feat notation eventually
            for fl in v:
                fl_components = fl.split(" ")
                if len(fl_components)>1 and fl_components[0] == "diff":
                    self._diff_feats[k].append(fl)
            self._feat_info[k] = _parse_feat_idxs(
                    out_feats=[l.split(" ")[-1] for l in v],
                    src_feats=self._tgs[self._tgs_ordered[0]]["dlabels"],
                    static_feats=self._tgs[self._tgs_ordered[0]]["slabels"],
                    derived_feats=self._derived_feats,
                    )[:2]

        ## establish vectors for normalizing outputs. keep any modifier parts
        self._norms = {
                k:np.array([self._coeffs.get(fl,(0,1)) for fl in v]).T
                for k,v in pf_args.items()
                }
        self._norms["s"] = np.array([
            self._coeffs.get(fl,(0,1)) for fl in self._s_feats
            ]).T

        self._feat_info["s"] = tuple(zip(*[
                (self._tgs[self._tgs_ordered[0]]["slabels"].index(fl),None)
                for fl in self._s_feats
                ]))
        self._feat_info["as"] = tuple(zip(*[
                (self._tgs[self._tgs_ordered[0]]["slabels"].index(fl),None)
                for fl in self._as_feats
                ]))
        self._feat_info["si"] = tuple(zip(*[
                (self._tgs[self._tgs_ordered[0]]["slabels"].index(fl),None)
                for fl in self._si_feats
                ]))
        if self._debug:
            print(f"Total chunks over {len(self._tgs.keys())} timegrids:",
                    len(self._chunks))

        ## probably no need to clutter up the namespace. just use the signature
        '''
        for k,v in self.signature.items():
            if k in dir(self):
                print(f"DUPLICATE PARAM: {k}. Not setting.")
            else:
                setattr(self, k, v)
        '''

    @property
    def feats(self):
        """
        Return a dict mapping array names to their feature labels, which is
        organized in the same way as the returned data.
        """
        return {
            "window":self._w_feats,
            "horizon":self._h_feats,
            "static":self._s_feats
            "static-int":self._si_feats,
            "target-init":[f.split(" ")[-1] for f in self._y_feats],
            "target":self._y_feats,
            "auxd-h":self._ad_feats,
            "auxd-w":self._ad_feats,
            "auxs":self._as_feats,
            "time":["epoch"],
            }

    @property
    def signature(self):
        """ expose dict of all input parameters for serialization / re-init """
        return {
                "timegrids":self._tgs_ordered,
                "window_feats":self._w_feats,
                "horizon_feats":self._h_feats,
                "target_feats":self._y_feats,
                "static_feats":self._s_feats,
                "static_int_feats":self._si_feats,
                "derived_feats":self._derived_feats,
                "aux_dynamic_feats":self._ad_feats,
                "aux_static_feats":self._as_feats,
                "static_embed_maps":self._si_embed_maps,
                "window_size":self._w_size,
                "horizon_size":self._h_size,
                "norm_coeffs":self._coeffs,
                "shuffle":self._shuffle,
                "seed":self._seed,
                "sample_cutoff":self._sample_cutoff,
                "sample_across_files":self._sample_across_files,
                "sample_under_cutoff":self._sample_under_cutoff,
                "sample_separation":self._smp_sep,
                "random_offset":self._roffset,
                "chunk_pool_count":self._cpc,
                "buf_size_mb":self._buf_size_mb,
                "buf_slots":self._buf_slots,
                "buf_policy":self._buf_policy,
                "debug":self._debug
                }

    def _replenish_chunk_pool(self, pool_slice:slice):
        """
        Load a series of chunks from the timegrid files, identify valid start
        times, separate samples into time series components, calculate derived
        features, normalize data, and return the new series of samples.
        """
        dsamples = []
        ssamples = []
        tsamples = []
        ## start time
        if self._debug:
            rcp_stime = perf_counter()
        ## extract samples from all the chunks in the provided pool
        for tg,(ts,ps),across,next_tg in self._chunks[pool_slice]:
            ts_extended = slice(ts.start, ts.stop+self._smp_size)
            dsample = self._tgs[tg]["dynamic"][ts_extended,ps,:]
            ssample = self._tgs[tg]["static"][ps,:]
            tsample = self._tgs[tg]["time"][ts_extended][1:]
            dsample = dsample.astype(np.float32)
            ssample = ssample.astype(np.float32)
            tsample = tsample.astype(np.float32)
            #print(dsample.dtype,ssample.dtype)
            ## If this chunk overlaps the next timegrid, extract the
            ## corresponding first sample size in the next one
            if across and not next_tg is None:
                ## valid start positions are drawn from within the current tg
                ## chunk so will need no more than an additional sample's size
                dsample = np.concatenate([
                    dsample,self._tgs[next_tg]["dynamic"][:self._smp_size,ps,:]
                    ], axis=0)
                tsample = np.concatenate([
                    tsample,self._tgs[next_tg]["time"][:self._smp_size]
                    ], axis=0)
                if self._debug:
                    print(f"Loading spanning chunk!")
            ## apply pixel-wise offsets and extract all samples from valid
            ## starting positions that fall within this temporal slice
            for pix,offset in enumerate(self._offsets[ps]):
                slice_range = np.arange(ts.start,ts.stop)
                sixs = slice_range[(slice_range-offset)%self._smp_sep == 0]
                for ix in sixs-ts.start:
                    ## replace this with extracted (w,h,s,si,t,y)
                    dsamples.append(dsample[ix:ix+self._smp_size,pix,:])
                    ssamples.append(ssample[pix,:])
                    tsamples.append(tsample[ix:ix+self._smp_size])
        ## extract completion time
        if self._debug:
            rcp_etime = perf_counter()

        ## separate, reorder, calculate derived features for each data category
        ## also if feats are differentiated calculate the forward-difference.
        dsamples = np.stack(dsamples, axis=0)
        ssamples = np.stack(ssamples, axis=0)
        ## add singleton feat dimension for consistency and remove the initial
        ## time step that was retained in _smp_size for difference calculations
        tsamples = np.stack(tsamples, axis=0)[:,1:,None]

        tmp_w = _calc_feat_array(
                src_array=dsamples[:,:self._w_size+1],
                static_array=ssamples,
                stored_feat_idxs=self._feat_info["w"][0],
                derived_data=self._feat_info["w"][1],
                )
        for fl in self._diff_feats["w"]:
            ix_fl = self._w_feats.index(fl)
            tmp_w[:,1:,ix_fl] = np.diff(tmp_w[...,ix_fl], axis=1)
        tmp_w = (tmp_w[:,1:]-self._norms["w"][0])/self._norms["w"][1]
        tmp_w = tmp_w.astype(self._out_dtype)

        tmp_h = _calc_feat_array(
                src_array=dsamples[:,-self._h_size-1:],
                static_array=ssamples,
                stored_feat_idxs=self._feat_info["h"][0],
                derived_data=self._feat_info["h"][1],
                )
        for fl in self._diff_feats["h"]:
            ix_fl = self._h_feats.index(fl)
            tmp_h[:,1:,ix_fl] = np.diff(tmp_h[...,ix_fl], axis=1)
        tmp_h = (tmp_h[:,1:]-self._norms["h"][0])/self._norms["h"][1]
        tmp_h = tmp_h.astype(self._out_dtype)

        tmp_y = _calc_feat_array(
                src_array=dsamples[:,-self._h_size-1:],
                static_array=ssamples,
                stored_feat_idxs=self._feat_info["y"][0],
                derived_data=self._feat_info["y"][1],
                )
        for fl in self._diff_feats["y"]:
            ix_fl = self._y_feats.index(fl)
            tmp_y[:,1:,ix_fl] = np.diff(tmp_y[...,ix_fl], axis=1)
        ## extract undifferentiated initial vector
        tmp_init = tmp_y[:,0][:,None]
        tmp_init = (tmp_init-self._norms["yinit"][0])/self._norms["yinit"][1]
        tmp_init = tmp_init.astype(self._out_dtype)
        tmp_y = (tmp_y[:,1:]-self._norms["y"][0])/self._norms["y"][1]
        tmp_y = tmp_y.astype(self._out_dtype)

        tmp_ad = _calc_feat_array(
                src_array=dsamples,
                static_array=ssamples,
                stored_feat_idxs=self._feat_info["ad"][0],
                derived_data=self._feat_info["ad"][1],
                )
        for fl in self._diff_feats["ad"]:
            ix_fl = self._ad_feats.index(fl)
            tmp_ad[1:,:,ix_fl] = np.diff(tmp_ad[...,ix_fl], axis=0)
        tmp_ad = tmp_ad[:,1:] ## don't normalize auxiliary data
        tmp_ad = tmp_ad.astype(self._out_dtype)
        tmp_adw = tmp_ad[:,:self._w_size]
        tmp_adh = tmp_ad[:,-self._h_size-1:]

        ## extract float-style static data for each sample
        tmp_s = np.stack([
            ssamples[:,fix] for fix in self._feat_info["s"][0]
            ], axis=-1)
        tmp_s = (tmp_s-self._norms["s"][0])/self._norms["s"][1]
        tmp_s = tmp_s.astype(self._out_dtype)
        tmp_as = np.stack([
            ssamples[:,fix] for fix in self._feat_info["as"][0]
            ], axis=-1)
        tmp_as = tmp_as.astype(self._out_dtype)

        ## extract and one-hot encode int-style static data
        tmp_si = [
                np.zeros(
                    (ssamples.shape[0], len(self._si_embed_maps[fl])),
                    dtype=np.float32)
                for fl in self._si_feats
                ]
        try:
            for i,fl in enumerate(self._si_feats):
                tmpx = ssamples[:,self._feat_info["si"][0][i]]
                si_ixs = np.vectorize(
                        self._si_embed_maps[fl].get,
                        otypes="i")(tmpx.astype(int))
                tmp_si[i][np.arange(ssamples.shape[0]),si_ixs] = 1
        except TypeError as te:
            print(f"Not all values in {fl} hae an embed map element")
            print(f"Unique values: {np.unique(tmpx)}")
            print(f"Embed map: {self._si_embed_maps[fl]}")
            raise te

        ## if requested, shuffle the samples between the chunks
        self._cur_sample_ix = 0
        self._cur_sample_count = tmp_w.shape[0]
        sixs = np.arange(self._cur_sample_count)
        if self._shuffle:
            self._rng.shuffle(sixs)

        ## format the outputs as a tuple (inputs, outputs, auxiliary_
        '''
        self._cur_samples = (
                (tmp_w[sixs],tmp_h[sixs],tmp_s[sixs],tmp_si,tmp_init[sixs]),
                (tmp_y[sixs],), ## outputs
                (tmp_ad[sixs], tmp_as[sixs], tsamples[sixs]) ## auxiliary
                )
        '''
        self._cur_samples = {
            "window":tmp_w[sixs],
            "horizon":tmp_h[sixs],
            "static":tmp_s[sixs],
            "static-int":tuple(v[sixs] for v in tmp_si),
            "target-init":tmp_init[sixs],
            "target":tmp_y[sixs],
            "auxd-h":tmp_adh[sixs],
            "auxd-w":tmp_adw[sixs],
            "auxs":tmp_as[sixs],
            "time":tsamples[sixs],
            }
        ## finish time
        if self._debug:
            rcp_ftime = perf_counter()
            print(f"Extract: {rcp_etime-rcp_stime:.2f} " + \
                    f"Compute: {rcp_ftime-rcp_etime:.2f}")

    def __next__(self):
        """
        Step through the currently-loaded samples in the chunk pool,
        replenishing the pool with the next set of chunks when necessary.

        Samples are returned as 3-tuples like: (input, target, auxiliary)

        such that...

        input:      (window, horizon, static, static_int, target_init)
        target:     (targets,)
        auxiliary:  (aux_dynamic, aux_static, time)

        and each array has a structure like...

        window:         (B, Sw, Fw)
        horizon:        (B, Sh, Fh)
        static:         (B, Fs)
        static_int:     ((B, Fsi_k) for k in static_feats)
        target_init:    (B, 1, Fy)
        targets:        (B, Sh, Fy)
        aux_dynamic:    (B, Sw+Sh, Fad)
        aux_static:     (B, Fas)
        time:           (B, Sw+Sh)

        where...

        B       := samples per batch
        Sw      := window sequence elements
        Sh      := horizon sequence elements
        Fw      := window features
        Fh      := horizon features
        Fs      := static features
        Fsi_k   := Static int embedding elements
        Fy      := target features
        Fad     := auxiliary dynamic features
        Fas     := auxiliary static features
        """
        if self._cur_sample_ix == self._cur_sample_count:
            if len(self._pool_slices) == 0:
                raise StopIteration(f"No More Chunks Available")
            self._replenish_chunk_pool(self._pool_slices.pop(0))
        cur_sample = take_index_from_nested(
                self._cur_samples, self._cur_sample_ix, axis=0)
        self._cur_sample_ix += 1
        return cur_sample

    def __iter__(self):
        return self

def stsd_worker_init_fn(worker_id):
    """
    Modify the datasets so that they each reference a mutually exclusive
    subset of all available chunks when multiprocessing is used.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    dataset._chunks = dataset._chunks[worker_info.id::worker_info.num_workers]
    ## re-calculate the pool slices given the subset of chunks for this worker
    nslices = len(dataset._chunks) // dataset._cpc + \
            int(bool(len(dataset._chunks) % dataset._cpc))
    dataset._pool_slices = [slice(i*dataset._cpc,(i+1)*dataset._cpc)
            for i in range(nslices)]

dataset_options = {
    "stsd":SparseTimegridSampleDataset,
    }

def get_datasets_from_config(config):
    """
    Instantiate dataset for each of the datasets under 'data' in the config,
    and return the result as a dictionary of Dataset objects

    TODO: generalize so that the dataset type is determined dynamically
        rather than assuming SparseTimegridSampleDataset
    """
    out_dss = {}
    for dk,dc in config["data"].items():
        out_dss[dk] = SparseTimegridSampleDataset(
            timegrids=dc["timegrids"],

            ## feature configuration
            window_feats=config["feats"]["window_feats"],
            horizon_feats=config["feats"]["horizon_feats"],
            target_feats=config["feats"]["target_feats"],
            static_feats=config["feats"]["static_feats"],
            static_int_feats=config["feats"]["static_int_feats"],
            aux_dynamic_feats=config["feats"]["aux_dynamic_feats"],
            aux_static_feats=config["feats"]["aux_static_feats"],
            derived_feats=config["feats"]["derived_feats"],
            static_embed_maps=config["feats"]["static_embed_maps"],
            window_size=config["feats"]["window_size"],
            horizon_size=config["feats"]["horizon_size"],
            norm_coeffs=config["feats"]["norm_coeffs"],

            ## training data specific configuration
            shuffle=dc["shuffle"],
            sample_cutoff=dc["sample_cutoff"],
            sample_across_files=dc["sample_across_files"],
            sample_under_cutoff=dc["sample_under_cutoff"],
            sample_separation=dc["sample_separation"],
            random_offset=dc["random_offset"],
            chunk_pool_count=dc["chunk_pool_count"],
            buf_size_mb=dc["buf_size_mb"],
            buf_slots=dc["buf_slots"],
            buf_policy=dc["buf_policy"],

            seed=config["seed"],
            )
    return out_dss

if __name__=="__main__":
    proj_dir = Path("/rhome/mdodson/emulate-era5-land/")
    data_dir = proj_dir.joinpath("data/timegrids")
    tg_paths = [tg for tg in data_dir.iterdir()
            if int(tg.stem.split("_")[2]) in range(2012,2018)]
