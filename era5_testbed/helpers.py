import numpy as np
from pathlib import Path

def get_permutation_pair(length, seed=None):
    """
    Get a random permutation of the provided number of elements and its inverse
    """
    rng = np.random.default_rng()
    perm = np.arange(length)
    rng.shuffle(perm)
    r_perm = np.asarray(tuple(zip(*sorted(zip(
        list(perm), range(len(perm))
        ), key=lambda v:v[0])))[1])
    return perm,r_perm

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
