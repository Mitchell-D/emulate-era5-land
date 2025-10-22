import torch
import numpy as np
from pathlib import Path

from collections.abc import Mapping, Sequence

def move_to_device(data, device):
    """ Move all tensors in a nested data structure to a particular device """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return type(data)([move_to_device(item, device) for item in data])
    elif isinstance(data, dict):
        return {key: move_to_device(value, device)
                for key, value in data.items()}
    else:
        return data # Return non-tensor data as is

def np_collate_fn(batch):
    """
    Pytorch collate_fn implementation that can handle nested dicts
    and tuples, but returns numpy arrays instead of pytorch Tensors
    """
    elem = batch[0]
    if isinstance(elem, np.ndarray):
        return np.stack(batch, axis=0)
    elif isinstance(elem, Mapping):
        return {k:np_collate_fn([d[k] for d in batch]) for k in elem}
    elif isinstance(elem, tuple):
        return tuple(np_collate_fn(s) for s in zip(*batch))
    elif isinstance(elem, Sequence) and not isinstance(elem, str):
        return [np_collate_fn(samples) for samples in zip(*batch)]
    elif isinstance(elem, torch.Tensor):
        batch = torch.stack(batch, axis=0)
        if elem.requires_grad:
            batch = batch.detach()
        return batch.numpy()
    else:
        raise ValueError(f"Make another condition for this: {type(elem)}")

def _parse_feat_idxs(out_feats, src_feats, static_feats, derived_feats,
        alt_feats:list=[]):
    """
    Helper for determining the Sequence indeces of stored features,
    and the output array indeces of derived features.

    :@param out_feats: Full ordered list of output features including
        dynamic stored and dynamic derived features
    :@param src_feats: Full ordered list of the features available in
        the main source array, which will be reordered and sub-set
        as needed to supply the ingredients for derived feats
    :@param static_feats: List of labels for static array features
    :@param alt_feats: If stored features can be retrieved from a
        different source array, provide a list of that array's feat
        labels here, and a third element will be included in the
        returned tuple listing the indeces of stored features with
        respect to alt_feats. These indeces will correspond in order
        to the None values in the stored feature index list
    :@return: 2-tuple (stored_feature_idxs, derived_data) where
        stored_feature_idxs is a list of integers indexing the
        array corresponding to src_feats, and derived_data is a
        4-tuple (out_idx,dynamic_arg_idxs,static_arg_idxs,lambda_func).
        If alt_feats are provided, a 3-tuple is returned instead
        with the third element being the indeces of features available
        only in the alternative array wrt the alternative feature list.
    """
    tmp_sf_idxs = [] ## stored feature idxs wrt src feats
    tmp_derived_data = [] ## derived feature idxs wrt output arrays
    tmp_alt_sf_idxs = [] ## alt stored feature idxs wrt alt_feats
    tmp_alt_out_idxs = [] ## alt stored feature idxs wrt out array
    for ix,l in enumerate(out_feats):
        if l not in src_feats:
            if l in alt_feats:
                tmp_alt_sf_idxs.append(alt_feats.index(l))
                tmp_alt_out_idxs.append(ix)
                tmp_sf_idxs.append(0)
            elif l in derived_feats.keys():
                assert l in derived_feats.keys()
                ## make a place for the derived features in the output
                ## array by temporarily indexing the first feature,
                ## to be overwritten when derived values are calc'd.
                tmp_sf_idxs.append(0)
                ## parse the derived feat arguments and function
                tmp_in_flabels,tmp_in_slabels,tmp_func = \
                        derived_feats[l]
                ## get derived func arg idxs wrt stored static/dynamic
                ## data; cannot yet support nested derived feats
                tmp_in_fidxs = tuple(
                    src_feats.index(q) for q in tmp_in_flabels)
                tmp_in_sidxs = tuple(
                        static_feats.index(q) for q in tmp_in_slabels)
                ## store (output_idx, dynamic_input_idxs,
                ##          static_input_idxs, derived_lambda_func)
                ## as 4-tuple corresponding to this single derived feat
                tmp_derived_data.append(
                        (ix,tmp_in_fidxs,tmp_in_sidxs,eval(tmp_func)))
            else:
                raise ValueError(
                        f"{l} not a stored, derived, or alt feature")
        else:
            tmp_sf_idxs.append(src_feats.index(l))

    alt_info = (tmp_alt_sf_idxs, tmp_alt_out_idxs)
    return tuple(tmp_sf_idxs),tmp_derived_data,alt_info

def _calc_feat_array(src_array, static_array,
        stored_feat_idxs:tuple, derived_data:list,
        alt_info=None, alt_array=None,
        alt_to_src_shape_slices:tuple=tuple()):
    """
    Compute a feature array including derived features and stored
    features from an alternative source array. This includes
    extracting and re-ordering a subset of source and alternative
    data features, as well as extracting ingredients for and
    computing derived data.

    Both stored_feat_idxs and derived_data, and optionally
    alt_info are outputs of _parse_feat_idxs

    stored_feat_idxs must include placeholder indeces where derived
    or alternative data is substituted. derived_data is a list
    of 4-tuples: (out_idx, dynamic_arg_idxs, static_arg_idxs, func)
    where out_idx specifies each derived output's location in the
    output array, *_arg_idxs are the indeces of the function inputs
    with respect to the source array, and func is the initialized
    lambda object associated with the transform.

    The optional alternative array of dynamic features may be the
    same shape or larger than the source array, As long as it can
    be adapted to the proper size.

    The alternative array ability is mainly used to provide a
    feature stored in the "pred" array of a sequence time series
    as an output in the "horizon" sequence array. The prediction
    array has samples covering the same time range as the horizon
    array, but including the timestep just prior to the first
    output. With the alternative functionality, features predicted
    by a different model (ie snow, runoff, canopy evaporation) may
    be substituted for the actual outputs.

    :@param src_array: Array-like main source of input data for
        the derived feature. The output shape will match this
        array's shape, except for the final (feature) axis.
    :@param static_array: Array like source of static data for
        derived features, which must contain a superset of all
        their ingredient features.
    :@param stored_feat_idxs: Ordered indeces of stored feats with
        respect to the source array, including placeholder values
        (typically 0) where derived/alternative feats are placed.
        This is an output of _parse_feat_idxs
    :@param derived_data: List of 4-tuples (see above) containing
        derived feature info and functions. This is an output of
        _parse_feat_idxs
    :@param alt_info: Optional 2-tuple of lists for alt feature
        indeces wrt the alt array and output array, respectively.
        This is also an output of _parse_feat_idxs.
    :@param alt_array: Alternative source array containing a
        superset of any alt feats requested in the output array.
    :@param alt_to_src_shape_slices: tuple of slice objects that
        correspond to the axes of alt_array, which reshape
        alt_array to the shape of src_array (except the feat dim).
    """
    ## Extract a numpy array around stored feature indeces, which
    ## should include placeholders for alt and derived feats
    sf_subset = src_array[...,stored_feat_idxs]

    ## Extract and substitute alternative features
    if not alt_array is None:
        ## 2 empty lists will extracts zero-element arrays that
        ## don't affect the stored feature subset
        if alt_info is None:
            alt_info = ([], [])
        ## alt array slc should be a tuple of slices
        slc = alt_to_src_shape_slices
        if type(slc) is slice:
            slc = (slc,)
        alt_sf_idxs,alt_out_idxs = alt_info
        ## slice the alt array to match the source array,
        ## and replace features sourced from alt data
        sf_subset[...,alt_out_idxs] = alt_array[*slc][...,alt_sf_idxs]

    ## Calculate and substitute derived features
    for (ix,dd_idxs,sd_idxs,fun) in derived_data:
        try:
            sf_subset[...,ix] = fun(
                    tuple(src_array[...,f] for f in dd_idxs),
                    tuple(static_array[...,f] for f in sd_idxs),
                    )
        except Exception as e:
            print(f"Error getting derived feat in position {ix}:")
            print(e)
            raise e
    return sf_subset

def get_permutation_inverse(perm:np.array):
    """
    Get a random permutation of the provided number of elements and its inverse

    :@param perm: (N,) integer array mapping original positions to new indeces
    """
    return np.asarray(tuple(zip(*sorted(zip(
        list(perm), range(len(perm))
        ), key=lambda v:v[0])))[1])

def get_permutation(coord_array, initial_perm=None, target_avg_dist=3,
        roll_threshold=.66, threshold_diminish=.01, recycle_count=2,
        dynamic_roll_threshold=False, max_iterations=None, seed=None,
        return_stats=False, debug=False):
    """
    Method for iteratively discovering a semi-random permutation that
    balances preserving the approximate spatial locality of coordinates while
    randomly shuffling indeces.

    The goal is for the permuted data to be partitioned into contiguous chunks
    in a way that simultaneously (1) minimizes the number of chunks needed
    to extract a global random subset of points and (2) minimizes the number
    of chunks needed to extract a subset of points within local bounds.

    This algorithm accomplishes that goal by ranking the distance of each
    permutation, and re-shuffling the most and least distant. Both are mutually
    shuffled together because otherwise the closest would converge on their
    original positions.

    :@param coord_array: (N,C) array of N data points located by C cartesian
        coordinate dimensions. These are the original positions of each point.
    :@param initial_perm: (N,) integer array capturing the first-guess
        permutation of the N points from their original position.
    :@param target_avg_distance: Mean cartesian distance in coordinate space
        below which the search stops. Set this to a level that maintains
        reasonable locality without restoring the original positions.
    :@param roll_threshold: Initial ratio of furthest points to reshuffle.
    :@param threshold_diminish: Decrease in ratio of reshuffled points / iter.
    :@param recycle_count: Number of closest points to include in reshuffling.
    :@param dynamic_roll_threshold: If True, the threshold of far-distance
        points to reshuffle is calculated as the ratio of the current mean
        distance to that given the initial_perm, with roll_threshold as an
        upper bound. Theoretically this should solve more throroughly, but
        is typically much slower to converge.
    :@param max_iterations: Max iterations allowed to discover a permutation
    :@param seed: Random seed for initial permutation and reshuffling.
    :@param return_stats: If True, a list of 2-tuples (mean_dist, stdev_dist)
        corresponding to the permutation from each iteration is returned
    :@param debug: If True, prints mean and stdev of distance each iteration.

    :@return: Array of the permutation, or 2-tuple (permutation, stats) if
        return_stats is True
    """
    ## establish the random number generator and initial index permutation
    rng = np.random.default_rng(seed=seed)
    if initial_perm is None:
        initial_perm = np.arange(coord_array.shape[0])
        rng.shuffle(initial_perm)
    tmp_perm = initial_perm

    init_avg_dist = None
    iter_count=0
    stats = []
    while True:
        ## determine the euclidean distance in coordinate space of each
        ## point's destination from its origin given the current permutation
        tmp_ond = np.asarray([
            (ixo,ixn,np.sum((coord_array[ixn]-coord_array[ixo])**2)**(1/2))
            for ixo,ixn in enumerate(tmp_perm)
            ])
        ## sort permutations by distance
        dsort_old_ix,dsort_new_ix,dsort_dist = map(
            np.asarray,zip(*sorted(tmp_ond, key=lambda ond:ond[-1])))
        avg_dist = np.average(dsort_dist)
        if debug:
            print(f"Distance Avg: {avg_dist:<6.3f} " + \
                f"Stdev: {np.std(dsort_dist):<6.3f}")
        if return_stats:
            stats.append((avg_dist, np.std(dsort_dist)))
        if init_avg_dist is None:
            init_avg_dist = avg_dist

        ## If dynamic roll threshold is requested, calculate it, using the
        ## user-provided roll_threshold as an upper bound
        if dynamic_roll_threshold:
            roll_threshold = min([avg_dist / init_avg_dist, roll_threshold])
        else:
            roll_threshold -= threshold_diminish

        ## return the current permutation if the roll threshold is fully
        ## diminished, the target average has been met, or out of iterations.
        if int(tmp_perm.size * roll_threshold) <= 0:
            break
        if avg_dist <= target_avg_dist:
            break
        if max_iterations != None:
            if iter_count > max_iterations:
                break
            else:
                iter_count += 1

        ## determine the threshold of most distant points to re=roll
        roll_range =  int(dsort_dist.size * roll_threshold)
        ## get and shuffle the indeces of the most and least distant points
        reperm_idxs = np.concatenate([
            np.arange(recycle_count), ## least-distant points to recycle
            np.arange(dsort_dist.size)[-roll_range:], ## most-distant points
            ], axis=0)
        rng.shuffle(reperm_idxs)

        ## develop the new permutation by remapping nearest and furthest points
        reperm = np.arange(dsort_dist.size)
        reperm[:recycle_count] = reperm_idxs[:recycle_count]
        reperm[-roll_range:] = reperm_idxs[-roll_range:]
        dsort_new_ix = dsort_new_ix[reperm]

        ## restore the modified permutation by sorting by initial indeces
        dsort_old_ix,tmp_perm = zip(*sorted(
            zip(dsort_old_ix,dsort_new_ix),
            key=lambda d:d[0]
            ))
        tmp_perm = np.asarray(tmp_perm, dtype=int)
    if return_stats:
        return tmp_perm, stats
    return tmp_perm

def get_permutation_conv(coord_array, dist_threshold, reperm_cap, shuffle_frac,
        seed=None, return_stats=False, debug=False):
    """
    Alt method for generating a locality preserving semi-random permutation
    """
    rng = np.random.default_rng(seed)
    jump_count = np.zeros(coord_array.shape[0])
    ## index order wrt original array for mixing steps
    conv_order = np.arange(coord_array.shape[0])
    rng.shuffle(conv_order)

    stats = []
    ## start with identity permutation
    cur_perm = np.arange(coord_array.shape[0])
    for ix in conv_order:
        ## unpermuted distances wrt chosen pixel and mix candidate mask
        dists = (np.sum((coord_array-coord_array[ix])**2,axis=1))**(1/2)
        m_mix = (dists < dist_threshold) & (jump_count <= reperm_cap)
        num_mix = int(np.count_nonzero(m_mix)*shuffle_frac)
        if not num_mix > 1:
            continue
        ## unpermuted indeces of permutation destinations to mix
        ix_nomix = rng.choice(np.where(m_mix)[0], size=num_mix, replace=False)
        ## shuffled permutation destinations
        ix_mix = np.copy(ix_nomix)
        rng.shuffle(ix_mix)

        ## shuffle the selected mix indeces
        cur_perm[ix_nomix] = cur_perm[ix_mix]
        #jump_count[ix_nomix] += 1 ## disabled and modified at 171
        ## carry over the number of jumps for this pixel from prev position
        new_jump_count = np.copy(jump_count)
        new_jump_count[ix_nomix] = jump_count[ix_mix] + 1
        jump_count = new_jump_count

        if debug:
            tmpd = np.sum((coord_array-coord_array[cur_perm])**2, axis=1)**(.5)
            print(f"Distance Avg: {np.average(tmpd):<6.3f} " + \
                f"Stdev: {np.std(tmpd):<6.3f}")
            if return_stats:
                stats.append((np.average(tmpd), np.std(tmpd)))

    if return_stats:
        return cur_perm,stats
    return cur_perm

def mp_get_permutation(args):
    return args,get_permutation(**args)

def mp_get_permutation_conv(args):
    return args,get_permutation_conv(**args)

def sector_slices(sector_shape, bounding_box,
        iteration_order=None, separate_sparse=True):
    """
    get an ordered list of tuples corresponding to "slices" of a grid

    Along each dimension, sector inclusion can be specified according to:

        None        : all elements along axis at once
        integer     : maximum size of slice in number of elements
        list[int]   : collection of individual indeces to extract along axis

    Along each dimension the bounding box to break into sectors is defined per:

        integer     : number of elements along the axis indexed from zero
        list[int]   : 2-element list [min, max] range of indeces

    :@param sector_shape: Maximum shape along each dimension for sectors that
        are returned. The actual shape of returned sectors may be smaller if
        a border of bounding_box is encountered.
    :@param bounding_box: Size of the grid within which sectors are
        partitioned. This may be a tuple of integers specifying the number
        of elements along each axis, or a tuple of 2-tuples of integers
        contianing a range of index values [start, stop) along each axis.
    :@param iteration_order: Optional list of integers specifying the order in
        which each axis will be incremented during iteration. The first item in
        the list will be varied and all others held constant until the last
        element along the first item's axis is reached, at which point
        the next element in the second item's axis will be selected, and
        iteration will proceed along the first item's axis, and so forth.
    """
    assert len(bounding_box)==len(sector_shape)

    ## establish the dimensions of the bounding box we will be iterating within
    bbox = [] ## normalized shape of bounding box
    boffset = [] ## integer offset number of elements per dimension
    for bdim in bounding_box:
        try:
            if isinstance(bdim, tuple):
                assert len(bdim)==2
                assert all(isinstance(v,int) and v>0 for v in bdim)
                assert bdim[0]<bdim[1]
                bbox.append(bdim[1]-bdim[0])
                boffset.append(bdim[0])
                continue
            elif isinstance(bdim, int):
                assert bdim>0
                bbox.append(bdim)
                boffset.append(0)
            else:
                raise ValueError("unrecognized bbox type")
        except:
            raise ValueError(
                    "All axes in the boundary box definition must " + \
                    "be either an integer maximum size or a 2-tuple " + \
                    "range of integers like [start, stop)")

    ## determine the type and number of iterations per axis
    iclasses = [] ## iteration type classes: [span, tile, sparse]
    icounts = [] ## number of iterations along the axis
    islices = [[] for i in range(len(sector_shape))]
    for ax,s in enumerate(sector_shape):
        if s is None:
            iclasses.append(0)
            icounts.append(1)
            islices[ax].append(slice(boffset[ax],boffset[ax]+bbox[ax]))
        ## tiled range of elements
        elif isinstance(s, int):
            assert s <= bbox[ax], "slice along {ax = } is too large"
            iclasses.append(1)
            nslc = bbox[ax]//s + bool(bbox[ax]%s)
            icounts.append(nslc)
            for c in range(nslc):
                ci = boffset[ax] + c*s
                cf = min([ci+s, boffset[ax]+bbox[ax]])
                islices[ax].append(slice(ci, cf))
        ## sparsely defined elements
        elif isinstance(s, (list,tuple,np.ndarray)):
            s = np.asarray(s)
            assert len(s.shape)==1, "use 1d array of indeces"
            assert np.issubdtype(s.dtype, np.integer)
            for v in s:
                assert boffset[ax] <= v < (boffset[ax] + bbox[ax])
            iclasses.append(2)
            if separate_sparse:
                icounts.append(s.size)
                for v in s:
                    islices[ax].append(slice(v,v+1))
            else:
                icounts.append(1)
                islices[ax].append(s)
        else:
            raise ValueError(f"Unrecognized sector argument: {type(s) = } " + \
                    "must be None, an integer, or an iterable.")


    if iteration_order is None:
        iteration_order = list(range(len(bounding_box)))
    else:
        iteration_order = list(iteration_order)
        if len(iteration_order)!=len(bounding_box):
            excluded = list(set(range(len(bounding_box)))-set(iteration_order))
            for ex in excluded:
                assert icounts[ex]==1,"Axes with multiple iteration steps " + \
                        "must also be provided in the iteration order list!"

                iteration_order.append(ex)

    ## reverse permutation of iteration order
    r_order,_ = zip(*sorted(
        enumerate(iteration_order[::-1]),
        key=lambda p:p[1]))

    ## develop indeces of slices in the requested order of iteration
    sidxs = np.stack(np.meshgrid(
        *[np.arange(icounts[ix]) for ix in iteration_order[::-1]],
        indexing="ij"
        ), axis=-1).reshape(-1, len(icounts))[...,r_order]

    return [[islices[j][sidxs[i,j]] for j in range(sidxs.shape[1])]
            for i in range(sidxs.shape[0])]
