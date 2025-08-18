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
    def __init__(self, timegrids:list, shuffle=True, seed=None, window_size=24,
            horizon_size=24, dynamic_norm_coeffs={}, static_norm_coeffs={},
            derived_feats={}, sample_across_files=True, sample_cutoff=1,
            sample_under_cutoff=True, sample_separation=1, shuffle_offset=True,
            chunk_pool_count=4, buf_size_mb=1024, buf_slots=128, buf_policy=0,
            debug=False):
        """
        :@param timegrids: Timegrid files, MUST be ordered chronologically so
            that contiguous samples can be collected across files.
        :@param shuffle: If True, samples will be shuffled by chunk rather than
            being returned chronologically as stored.
        :@param seed: Random number generator seed
        :@param window_size: Number of input steps prior to the first
            prediction step, used to initialize the model.
        :@param horizon_size: Number of covariate input and prediction steps.
        :@param dynamic_norm_coeffs: Dict mapping feature names to a 2-tuple
            (mean, stddev) for normalizing input variables to a unit gaussian.
        :@param static_norm_coeffs:
        :@param derived_feats:
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
            ssize = self._w_size + self_h_size
            ntimes = self._tgs[tg]["dynamic"].shape[0]
            cur_chunks = [
                    (tg, c[:2], ntimes-c[0].stop<ssize,
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
        self._cur_sample_ix = 0

    def _replenish_chunk_pool(self, pool_slice:slice):
        """ """
        samples = []
        for tg,(ts,ps),across,next_tg in self._chunks[cpool]:
            tmp_d = self._tgs[tg]["dynamic"][ts,ps,:]
            tmp_s = self._tgs[tg]["static"][ps,:]
            if across and not next_tg is None:
                tmp_d = np.concatenate([
                    tmpd, self._tgs[next_tg]["dynamic"][:ssize,ps,:]
                    ], axis=0)
            for pix,offset in enumerate(self._offsets[ps]):
                ## replace this with extracted (w,h,s,si,t),y
                samples.append(tmp_d[offset:offset+ssize,pix])

    def __iter__(self):
        if self._cur_sample_ix == len(self._cur_samples):
            if len(self._pool_slices) == 0:
                raise StopIteration(f"No More Chunks Available")
            self._replenish_chunk_pool(self._pool_slices.pop(0))

        for cpool in pool_slices:
        return None

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

