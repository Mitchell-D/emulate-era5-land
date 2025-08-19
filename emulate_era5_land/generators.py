import numpy as np
import json
import torch
import h5py
from pathlib import Path

class SparseTimegridSampleDataset(torch.utils.data.IterableDataset):
    """
    PyTorch IterableDataset implementation with the ability to multiprocess
    over sparse sampling of 1d time series over
    """
    def __init__(self, timegrids:list, target_feats, window_feats,
            horizon_feats, static_feats, static_int_feats, derived_feats={},
            static_embed_sizes={}, window_size=24, horizon_size=24,
            dynamic_norm_coeffs={}, static_norm_coeffs={}, shuffle=True,
            seed=None, sample_across_files=True, sample_cutoff=1,
            sample_under_cutoff=True, sample_separation=1, shuffle_offset=True,
            chunk_pool_count=4, buf_size_mb=1024, buf_slots=128, buf_policy=0,
            debug=False):
        """

         o  o  o  o  o  o  o  o  o  o  o  o  o  o  o  o  o  o  o  o  o  o
        | |-1--2--3--4--5--6--7--8--9||1--2--3--4--5--6--7--8--9|
         |              |                          |
        buffer       window                     horizon

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
        :@param static_embed_sizes: Dict mapping a stored static feature label
            to an integer referring to the size of its embedding vector.
        :@param window_size: Number of input steps prior to the first
            prediction step, used to initialize the model.
        :@param horizon_size: Number of covariate input and prediction steps.
        :@param shuffle: If True, samples will be shuffled by chunk rather than
            being returned chronologically as stored.
        :@param seed: Random number generator seed
        :@param dynamic_norm_coeffs: Dict mapping feature names to a 2-tuple
            (mean, stddev) for normalizing input variables to a unit gaussian.
        :@param static_norm_coeffs:
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
        :@param shuffle_offset: If True, each spatial pixel is assigned a
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

        --( to implement )--

        :@param static_conditions:
        """
        super(SparseTimegridSampleDataset).__init__()
        self._tg_info = {}
        self._chunks = []
        self._tgs_ordered = timegrids
        self._cpc = chunk_pool_count
        self._rng = None

        self._w_feats = window_feats
        self._h_feats = horizon_feats
        self._s_feats = static_feats
        self._si_feats = static_int_feats
        ## include an additional element in case window diffs requested.
        self._ssize = self._w_size + self_h_size + 1

        ## collect open file pointers, chunks, and labels for each timegrid
        for i,tg in enumerate(timegrids):
            tg_open = h5py.File(tg, "r", rdcc_nbytes=buf_size_mb*1024**2,
                    rdcc_nslots=buf_slots, rdcc_w0=buf_policy)
            self._tgs[tg] = {
                    "dynamic":tg_open["/data/dynamic"],
                    "static":tg_open["/data/static"],
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
                    (tg, c[:2], ntimes-c[0].stop<self._ssize,
                        None if i==len(timegrids)-1 else timegrids[i+1])
                    for c in tg_open["/data/dynamic"].chunks
                    ]

            ## never include overlapping chunks for the last file.
            cur_chunks = filter(
                    lambda c:not (c[2] and c[3] is None), cur_chunks)
            ## drop overlapping chunks if not sampling across files
            if not sample_across_files:
                cur_chunks = filter(lambda c:not c[2], cur_chunks)

            ## Shuffle the chunks for all files and subset them as requested
            if shuffle:
                if self._rng is None:
                    self._rng = np.random.default_rng(seed=seed)
                self._rng.shuffle(self._chunks)
            else:
                self._rng = None

            ## subset above or below the provided threshold
            nc_under = int(len(self._chunks)*sample_cutoff)
            if sample_under_cutoff:
                cur_chunks = cur_chunks[:nc_under]
            else:
                cur_chunks = cur_chunks[-(len(self._chunks)-nc_under):]

            self._chunks += cur_chunks

        ## calculate pixel offsets if requested
        npx = None
        for tg in self._tgs_ordered:
            if npx is None:
                npx = self._tgs[tg]["dynamic"].shape[1]
            else:
                assert npx == self._tgs[tg]["dynamic"].shape[1], tg
        ## offsets must be between zero and the total length of samples
        if shuffle_offset:
            self._offsets = rng.integers(0, sample_separation, npx)
        else:
            self._offsets = np.zeros(npx)
        self._preload_count = preload_count

        self._w_size = window_size
        self._h_size = horizon_size

        ## determine
        nslices = len(self._chunks) // self._cpc + \
                int(bool(len(self._chunks) % self._cpc))
        self._pool_slices = [slice(i*self._cpc,(i+1)*self._cpc)
                for i in range(nslices)]
        self._ssize = self._w_size + self._h_size

        self._cur_samples = []

        ## make sure all timegrids have the same feats and ordering
        for tg in self._tgs_ordered[1:]:
            assert tuple(self._tgs[self._tgs_ordered[0]]["dlabels"]) \
                    == tuple(self._tgs[tg]["dlabels"]), tg
            assert tuple(self._tgs[self._tgs_ordered[0]]["slabels"]) \
                    == tuple(self._tgs[tg]["slabels"]), tg

        ## get idxs/funcs of the stored and derived dynamic feats of each type
        pf_args = {"w":self._w_feats, "h":self._h_feats, "y":self._y_feats}
        self._feat_info = {}
        self._diff_feats = {"w":[],"h":[],"y":[]}
        for k,v in pf_args.values:
            ## want to make a full grammar out of the feat notation eventually
            base_feats = []
            for fl in v:
                fl_components = fl.split(" ")
                if len(fl_components)>1 and fl_components[0] == "diff":
                    self._diff_feats[k].append(fl)
                base_feats.append()
            self._feat_info[k] = _parse_feat_idxs(
                    out_feats=[l.split(" ")[-1] for l in v],
                    src_feats=self._tgs[self._tgs_ordered[0]]["dlabels"],
                    static_feats=self._tgs[self._tgs_ordered[0]]["slabels"],
                    derived_feats=derived_feats,
                    )[:2]

        ## establish vectors for normalizing outputs. keep any modifier parts
        self._norms = {
                k:np.array([dynamic_norm_coeffs.get(fl,(0,1)) for fl in v]).T
                for k,v in pf_args.items()
                }
        self._norms["s"] = np.array([
            static_norm_coeffs.get(fl,(0,1)) for fl in self._s_feats
            ]).T

    def _replenish_chunk_pool(self, pool_slice:slice):
        """ """
        dsamples = []
        ssamples = []
        for tg,(ts,ps),across,next_tg in self._chunks[cpool]:
            dsample = self._tgs[tg]["dynamic"][ts,ps,:]
            ssample = self._tgs[tg]["static"][ps,:]
            if across and not next_tg is None:
                dsample = np.concatenate([
                    dsample, self._tgs[next_tg]["dynamic"][:ssize,ps,:]
                    ], axis=0)
            for pix,offset in enumerate(self._offsets[ps]):
                ## replace this with extracted (w,h,s,si,t),y
                dsamples.append(dsample[offset:offset+ssize,pix,:])
                ssamples.append(ssample[pix,:])

        ## separate, reorder, calculate derived features for each data category
        dsamples = np.stack(dsamples, axis=0)
        ssamples = np.stack(ssamples, axis=0)
        tmp_w = _calc_feat_array(
                src_array=dsamples[:self._w_size+1],
                static_array=ssamples,
                stored_feat_idxs=self._feat_info["w"][0],
                derived_data=self._feat_info["w"][1],
                )
        for fl in diff_feats["w"]:
            ix_fl = self._w_feats.index(fl)
            tmp_w[1:,:,ix_fl] = np.diff(tmp_w[...,ix_fl], axis=0)
        tmp_w = (tmp_w[1:]-self._norms["w"][0])/self._norms["w"][1]

        tmp_h = _calc_feat_array(
                src_array=dsamples[-self._h_size-1:],
                static_array=ssamples,
                stored_feat_idxs=self._feat_info["h"][0],
                derived_data=self._feat_info["h"][1],
                )
        for fl in diff_feats["h"]:
            ix_fl = self._h_feats.index(fl)
            tmp_h[1:,:,ix_fl] = np.diff(tmp_h[...,ix_fl], axis=0)
        tmp_h = (tmp_h[1:]-self._norms["h"][0])/self._norms["h"][1]

        tmp_y = _calc_feat_array(
                src_array=dsamples[-self._h_size-1:],
                static_array=ssamples,
                stored_feat_idxs=self._feat_info["y"][0],
                derived_data=self._feat_info["y"][1],
                )
        for fl in diff_feats["y"]:
            ix_fl = self._y_feats.index(fl)
            tmp_y[1:,:,ix_fl] = np.diff(tmp_y[...,ix_fl], axis=0)
        tmp_y = (tmp_y[1:]-self._norms["y"][0])/self._norms["y"][1]

        tmp_s = np.stack([ssamples[...,fl] for fl in self._s_feats], axis=-1)


    def __iter__(self):
        """ """
        if len(self._cur_samples)==0:
            if len(self._pool_slices) == 0:
                raise StopIteration(f"No More Chunks Available")
            self._replenish_chunk_pool(self._pool_slices.pop(0))
        return self._cur_samples.pop(0)

def worker_init_fn(worker_id):
    """
    Modify the datasets so that they only reference a subset of all available
    chunks
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    dataset._chunks = dataset._chunks[worker_info.id::worker_info.num_workers]

if __name__=="__main__":
    proj_dir = Path("/rhome/mdodson/emulate-era5-land/")
    data_dir = proj_dir.joinpath("data/timegrids")
    tg_paths = [tg for tg in data_dir.iterdir()
            if int(tg.stem.split("_")[2]) in range(2012,2018)]

