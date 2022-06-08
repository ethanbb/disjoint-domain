import numpy as np
from scipy.linalg import block_diag, svd
from scipy.cluster import hierarchy
from scipy.spatial import distance
import matplotlib.pyplot as plt
import torch
from typing import Dict, Any
import re

import util
import problem_analysis as pa

ITEMS_PER_DOMAIN = 8

# make some util functions available under this namespace
choose_k_inds = util.choose_k_inds
choose_k = util.choose_k
choose_k_set_bits = util.choose_k_set_bits


def get_cluster_sizes(clusters):
    """Parse a 'clusters' string such as '4-2-2_suffix' to get the size of each cluster"""
    try:
        sizes = [int(sz) for sz in clusters.partition('_')[0].split('-')]
    except ValueError:
        return [ITEMS_PER_DOMAIN]
    if sum(sizes) != ITEMS_PER_DOMAIN:
        raise ValueError('Invalid clusters specification')

    return sizes


def _make_n_dist_d_attr_vecs(centroid, n=4, d=4):
    """
    Useful for making 'circles' and similar structures in other attr cluster settings
    Here d is the *Hamming* distance between vectors, which is the square of Euclidean distance
    (and each vector is half the Hamming distance from the centroid as to each other vector)
    """
    if n == 1:  # don't overcomplicate things, just use the centroid itself
        return centroid.copy()

    if (n * d) % 4 != 0:
        raise ValueError('Distance times n must be a multiple of 4')

    try:
        all_set_bits = choose_k_set_bits(1 - centroid, round(n * d // 4))
        all_unset_bits = choose_k_set_bits(centroid, round(n * d // 4))
    except ValueError:
        raise ValueError(f'Not enough attributes to make {n} vectors of dist {d} from each other')

    similar_vecs = np.tile(centroid.copy(), (n, 1))

    for similar_vec, set_bits, unset_bits in zip(similar_vecs, np.array_split(all_set_bits, n),
                                                 np.array_split(all_unset_bits, n)):
        similar_vec[set_bits] = 1
        similar_vec[unset_bits] = 0

    return similar_vecs


def _make_3_group_attr_vecs(ctx_per_domain, attrs_per_context, attrs_set_per_item,
                            clusters='4-2-2', intragroup_dists=None, intergroup_dist=40,
                            organized=False, **_extra):
    """
    Make some attribute vectors that conform to the Euclidean distance plot (Figure R3, bottom).
    There are 8 items. Outputs a list of ctx_per_domain 8 x attrs_per_context matrices.
    These attributes are simply repeated for each domain.

    attrs_per_context must be at least 50 (approximately) in order for the distances to be correct.
    If not None (for default), intragroup_dists should be a list of length 4, specifying Hamming distances:
        - between each pair of circles (default=4)
        - between square and star centroids (default=12)
        - between squares (default=2)
        - between stars (default=10)
    Each must be even and must be a multiple of 4 if the corresponding group has an odd number of items.
    intergroup_dist refers to the distance between the circle and "other item" centroids.
    
    clust_sizes: 3-item list of # of circles, squares, and stars. Currently # of stars must be 2.
    
    If 'organized' is true, permute the final matrix to put cluster centroids toward the beginning, to help
    with visualization.

    """
    # Validate everything first
    max_disjoint_bits = max(attrs_set_per_item, attrs_per_context - attrs_set_per_item)

    if intergroup_dist % 2 != 0 or intergroup_dist // 2 > max_disjoint_bits:
        raise ValueError(f'Invalid intergroup distance - must be even and <= {max_disjoint_bits * 2}')

    clust_sizes = get_cluster_sizes(clusters)
    if len(clust_sizes) != 3 or sum(clust_sizes) != 8:
        raise ValueError('Invalid clust_sizes')
    n_circles, n_squares, n_stars = clust_sizes

    if intragroup_dists is None:
        default_sq_dist = 4 * (n_squares - 1) / n_squares  # hack to allow all but 1 to be 2 away from the centroid
        intragroup_dists = [4, 12, default_sq_dist, 10]

    circ_dist, sqst_dist, square_dist, star_dist = intragroup_dists

    attrs = [np.empty((ITEMS_PER_DOMAIN, attrs_per_context)) for _ in range(ctx_per_domain)]

    for attr_mat in attrs:  
        # first, handle circles
        # choose centroid randomly
        circ_centroid = np.zeros(attrs_per_context)
        circ_centroid_set = choose_k(np.arange(attrs_per_context), attrs_set_per_item)        
        circ_centroid_unset = np.setdiff1d(range(attrs_per_context), circ_centroid_set, assume_unique=True)
        circ_centroid[circ_centroid_set] = 1
        if organized:
            perm_vec = circ_centroid_set

        # now choose bits to flip for each of the 'circle' vectors, keeping total # set the same
        attr_mat[:n_circles] = _make_n_dist_d_attr_vecs(circ_centroid, n_circles, circ_dist)
        if organized:
            circle_attrs = np.flatnonzero(np.any(attr_mat[:n_circles], axis=0))
            circle_new_attrs = np.setdiff1d(circle_attrs, perm_vec, assume_unique=True)
            perm_vec = np.concatenate((perm_vec, circle_new_attrs))

        # pick centroid for other items, which should be 40 bits away from this centroid.
        # (or overridden by setting intergroup_dist)
        other_centroid = circ_centroid.copy()
        other_centroid_new_attrs = choose_k(circ_centroid_unset, intergroup_dist // 2)
        other_centroid[other_centroid_new_attrs] = 1
        other_centroid[choose_k(circ_centroid_set, intergroup_dist // 2)] = 0
        if organized:
            other_new_attrs = np.setdiff1d(other_centroid_new_attrs, perm_vec, assume_unique=True)
            perm_vec = np.concatenate((perm_vec, other_new_attrs))

        # now square and star centroids, which are centered on other_centroid and differ by 12 bits (by default)
        square_centroid, star_centroid = _make_n_dist_d_attr_vecs(other_centroid, 2, sqst_dist)
        if organized:
            square_new_attrs = np.setdiff1d(np.flatnonzero(square_centroid), perm_vec, assume_unique=True)
            perm_vec = np.concatenate((perm_vec, square_new_attrs))

        # squares differ by just 2 bits (by default). be a little imprecise and let one of them be the centroid.
        square_range = n_circles + np.arange(n_squares)
        attr_mat[square_range] = _make_n_dist_d_attr_vecs(square_centroid, n_squares, square_dist)
        if organized:
            square_attrs = np.flatnonzero(np.any(attr_mat[square_range], axis=0))
            square_new_attrs = np.setdiff1d(square_attrs, perm_vec, assume_unique=True)
            perm_vec = np.concatenate((perm_vec, square_new_attrs))

        # stars differ by 10 bits (by default).
        # again be a little imprecise, let one differ from centroid by 4 and the other by 6 (all unique)
        attr_mat[-n_stars:] = _make_n_dist_d_attr_vecs(star_centroid, n_stars, star_dist)
        if organized:
            star_new_attrs = np.setdiff1d(np.flatnonzero(star_centroid), perm_vec, assume_unique=True)
            perm_vec = np.concatenate((perm_vec, star_new_attrs))
            star_attrs = np.flatnonzero(np.any(attr_mat[-n_stars:], axis=0))
            star_new_attrs = np.setdiff1d(star_attrs, perm_vec, assume_unique=True)
            perm_vec = np.concatenate((perm_vec, star_new_attrs))
            
            remaining_inds = np.setdiff1d(np.arange(attrs_per_context), perm_vec)
            perm_vec = np.concatenate((perm_vec, remaining_inds))
            
            attr_mat[:] = attr_mat[:, perm_vec]

    return attrs


def _make_2_group_attr_vecs(ctx_per_domain, attrs_per_context, attrs_set_per_item,
                            clusters='4-4', intragroup_dists=None, intergroup_dist=40, **_extra):
    """
    Make attribute vectors with 2 clusters in a systematic way. All distances are Hamming and
    should be divisible by 4 (2 in the case of intergroup)
    """
    if intragroup_dists is None:
        intragroup_dists = [4, 12]

    clust_sizes = get_cluster_sizes(clusters)
    max_disjoint_bits = max(attrs_set_per_item, attrs_per_context - attrs_set_per_item)

    # Validate everything first
    if intergroup_dist % 2 != 0 or intergroup_dist // 2 > max_disjoint_bits:
        raise ValueError(f'Invalid intergroup distance - must be even and <= {max_disjoint_bits * 2}')

    if any([dist % 4 != 0 for dist in intragroup_dists]):
        raise ValueError('Invalid intragroup distances - must all be multiples of 4')
    if any([dist // 4 * n > max_disjoint_bits for dist, n in zip(intragroup_dists, clust_sizes)]):
        raise ValueError('Not enough attributes per cluster for these sizes and intragroup distances')

    attrs = [np.empty((ITEMS_PER_DOMAIN, attrs_per_context)) for _ in range(ctx_per_domain)]

    for attr_mat in attrs:
        circ_centroid = np.zeros(attrs_per_context)
        circ_centroid_set = choose_k(np.arange(attrs_per_context), attrs_set_per_item)
        circ_centroid_unset = np.setdiff1d(range(attrs_per_context), circ_centroid_set, assume_unique=True)
        circ_centroid[circ_centroid_set] = 1

        # make cirlces
        n_circles, n_squares = clust_sizes
        circ_dists, square_dists = intragroup_dists
        attr_mat[:n_circles] = _make_n_dist_d_attr_vecs(circ_centroid, n_circles, circ_dists)

        # make square centroid
        square_centroid = circ_centroid.copy()
        square_centroid[choose_k(circ_centroid_unset, intergroup_dist // 2)] = 1
        square_centroid[choose_k(circ_centroid_set, intergroup_dist // 2)] = 0

        # make squares
        attr_mat[n_circles:] = _make_n_dist_d_attr_vecs(square_centroid, n_squares, square_dists)

    return attrs


def _make_ring_attr_vecs(ctx_per_domain, attrs_per_context, attrs_set_per_item,
                         intragroup_dists=None, rotating_overlap=0, **_extra):
    """
    Make attribute vectors by activating a rotating set within a subset of attribute units
    (ITEMS_PER_DOMAIN * intragroup_dists[0]/2). Neighboring items (with wraparound) overlap
    by rotating_overlap within this subset, plus potentially by a fixed set of active units outside it.
    If rotating_overlap is 0, all items are equidistant.
    """
    if intragroup_dists is None:
        dist = 10
    else:
        dist = intragroup_dists[0]

    if dist % 2 != 0:
        raise ValueError('dist must be even')
    step = dist // 2
    
    rotating_section_size = step * ITEMS_PER_DOMAIN
    rotating_block_size = step + rotating_overlap
    assert rotating_block_size <= attrs_set_per_item, 'Not enough set attributes for desired step and overlap'
    fixed_section_size = attrs_set_per_item - rotating_block_size
    zero_section_size = attrs_per_context - rotating_section_size - fixed_section_size
    assert zero_section_size >= 0, 'Not enough total attributes for desired set # and rotation step and overlap'
    
    attrs = [np.zeros((ITEMS_PER_DOMAIN, attrs_per_context)) for _ in range(ctx_per_domain)]
    
    for attr_mat in attrs:
        # pick fixed set and rotating indices
        fixed_set_and_rot_inds = util.choose_k_inds(attrs_per_context, fixed_section_size + rotating_section_size)
        fixed_set_inds, rot_inds = fixed_set_and_rot_inds.split([fixed_section_size, rotating_section_size])

        # set fixed indices for all items and rotating indices for each individually
        attr_mat[:, fixed_set_inds] = 1
        for i, attr_vec in enumerate(attr_mat):
            rel_inds = (np.arange(rotating_block_size) + (i * step)) % rotating_section_size
            attr_vec[rot_inds[rel_inds]] = 1

    return attrs


def _make_eq_freq_attr_vecs(ctx_per_domain, attrs_per_context, attrs_set_per_item,
                            intergroup_dist=22, **_extra):
    """
    Make special 4-4-2 cluster attribute vectors such that each item has
    the same mean attribute frequency. Requires at least 53 attributes per context.
    Here, 'intergroup_dist' is interpreted as "excess" distance above the baseline of
    20.5 (between circles and squares/stars) and must be even.
    attrs_set_per_item should be between 12 + intergroup_dist/2 and
    (attrs_per_context - 53) + 12 - intergroup_dist/2.
    """
    base_attrs_needed = 53
    excess_attrs = attrs_per_context - base_attrs_needed
    ig_dist_attrs_per_item = intergroup_dist // 2
    uniform_attrs = excess_attrs - intergroup_dist  # all on or off
    nonuniform_attrs_per_item = 12 + ig_dist_attrs_per_item
    all_on_attrs = attrs_set_per_item - nonuniform_attrs_per_item
    all_off_attrs = uniform_attrs - all_on_attrs

    if excess_attrs < 0:
        raise ValueError(f'Eq freq requires >= {base_attrs_needed} attributes')

    if (intergroup_dist < 0 or intergroup_dist > excess_attrs or
            intergroup_dist % 2 != 0):
        raise ValueError('Intergroup dist not compatible with # of attrs for eq freq')

    if all_on_attrs < 0 or all_on_attrs > uniform_attrs:
        raise ValueError('Too many or few attrs per item for eq freq')

    # each matrix will be a copy of the template
    attr_template = np.zeros((ITEMS_PER_DOMAIN, attrs_per_context))
    curr_ind = 0

    # common for "circles", plus extras for other items to make it work
    attr_template[:4, curr_ind + np.arange(2 + ig_dist_attrs_per_item)] = 1
    attr_template[[6, 7], curr_ind + np.arange(2)] = 1
    curr_ind += 2 + ig_dist_attrs_per_item

    # common for "squares" and "stars"
    attr_template[4:, curr_ind + np.arange(3 + ig_dist_attrs_per_item)] = 1
    attr_template[[6, 7], curr_ind + np.arange(2)] = 0
    curr_ind += 3 + ig_dist_attrs_per_item

    # circle individual attributes, with some extra things for squares and stars
    for i in range(4):
        attr_template[i, curr_ind + np.arange(7)] = 1
        attr_template[4:6, curr_ind] = 1
        attr_template[[6, 6 + (i % 2), 7], curr_ind + 1 + np.arange(3)] = 1
        curr_ind += 7

    # square common and individual attributes
    attr_template[4:6, curr_ind] = 1
    attr_template[4, curr_ind + 1 + np.arange(4)] = 1
    attr_template[5, curr_ind + 5 + np.arange(4)] = 1
    curr_ind += 9

    # star common and individual attributes
    attr_template[6:8, curr_ind] = 1
    attr_template[6, curr_ind + 1 + np.arange(2)] = 1
    attr_template[7, curr_ind + 3 + np.arange(2)] = 1
    curr_ind += 5

    # circle round robin
    for i in range(3):
        for j in range(i + 1, 4):
            attr_template[[i, j], curr_ind] = 1
            curr_ind += 1

    # attributes common to all
    attr_template[:, curr_ind + np.arange(all_on_attrs)] = 1
    curr_ind += all_on_attrs

    assert attrs_per_context - curr_ind == all_off_attrs, "Something's wrong - attributes don't add up!"

    return [attr_template.copy() for _ in range(ctx_per_domain)]


def _make_ordering_attr_vecs(ctx_per_domain, attrs_per_context, attrs_set_per_item,
                             dist_accel=1, dist_offset=0, organized=True, **_extra):
    """
    Make a set of attribute vectors to implement the "ordering" structure as depicted in
    Figure 9 of Saxe et al., 2019.
    'dist_accel' is the amount by which the number of unique attributes assigned to each item
    increases as we iterate through the items.
    'dist_offset' is the extra number of unique attributes assigned to each item.
    If 'organized' is false, shuffles the attributes of each context independently.
    """
    n_accel_attrs = dist_accel * ITEMS_PER_DOMAIN * (ITEMS_PER_DOMAIN-1) // 2
    n_offset_attrs = dist_offset * ITEMS_PER_DOMAIN
    n_base_attrs = attrs_set_per_item - dist_offset
    total_attrs_used = n_base_attrs + n_accel_attrs + n_offset_attrs
    if total_attrs_used > attrs_per_context:
        raise ValueError('Not enough attributes for these "ordering" settings')
    
    attr_template = np.zeros((ITEMS_PER_DOMAIN, attrs_per_context))
    attr_template[:, :n_base_attrs] = 1
    
    # offset section
    for i in range(ITEMS_PER_DOMAIN):
        attr_template[i, n_base_attrs + i*dist_offset:n_base_attrs + (i+1)*dist_offset] = 1
    
    # accelerating section
    accel_offset = n_base_attrs + n_offset_attrs
    for i in range(1, ITEMS_PER_DOMAIN):
        attr_template[i, :i*dist_accel] = 0
        attr_template[i, accel_offset:accel_offset + i*dist_accel] = 1
        accel_offset += i*dist_accel
    
    attr_vecs = [attr_template.copy() for _ in range(ctx_per_domain)]
    if not organized:
        attr_vecs = [av[:, torch.randperm(attrs_per_context, device='cpu')] for av in attr_vecs]
    
    return attr_vecs


def _make_saxe_ordering_attr_vecs(ctx_per_domain, attrs_per_context, attrs_set_per_item,
                                  n_set_step=1, organized=True, **_extra):
    """
    Make "ordering" attribute vectors the way Saxe et al. do it, as in equation S63 in the supplement.
    This method treats 'attrs_set_per_item' as a mean, but each item will have a different # of set attributes.
    2022-04-27: changed so that attrs_set_per_item is the mean rather than the maximum.
    If 'organized' is false, shuffles the attributes of each context independently.
    """
    n_set_range = n_set_step * (ITEMS_PER_DOMAIN-1)
    max_n_set = attrs_set_per_item + n_set_range // 2
    each_n_set = max_n_set - n_set_step * np.arange(ITEMS_PER_DOMAIN)
    
    if each_n_set[0] > attrs_per_context or each_n_set[-1] <= 0:
        raise ValueError('Step in attributes set too large')
        
    attr_template = np.zeros((ITEMS_PER_DOMAIN, attrs_per_context))
    for n_set, row in zip(each_n_set, attr_template):
        row[:n_set] = 1
        
    attr_vecs = [attr_template.copy() for _ in range(ctx_per_domain)]
    if not organized:
        attr_vecs = [av[:, torch.randperm(attrs_per_context, device='cpu')] for av in attr_vecs]
    
    return attr_vecs
                                  

def _shuffle_attr_vec_mat(attr_vecs):
    """
    Given a matrix of attr_vecs (y), shuffles them in such a way
    as to destroy the group hierarchy but preserve the mean frequency per item.
    The algorithm is as follows:
        - For each n, 1 <= n <= ITEMS_PER_DOMAIN, we consider the set of
          attributes that are on for exactly n items. This set is an invariant.
        - The number of attributes on for each item within each set is also an invariant.
          Keep track of how many attributes are remaining to be assigned for each item.
        - Iterate through the attributes in each set and turn on the attribute for a
          random combination of n of the items that are not yet fully assigned.
    """
    new_attr_vecs = np.zeros(attr_vecs.shape)
    for num_items in range(1, ITEMS_PER_DOMAIN + 1):
        set_attrs = np.isclose(np.sum(attr_vecs, axis=0), num_items)
        attr_vecs_n = attr_vecs[:, set_attrs]
        num_attrs = attr_vecs_n.shape[1]
        num_to_assign = np.round(np.sum(attr_vecs_n, axis=1)).astype(int)
        skips_left = num_attrs - num_to_assign  # any with skips_left == 0 get set from there on
        attr_vecs_n[...] = 0

        # iterate through and assign attributes to items randomly
        for vec in attr_vecs_n.T:
            # include all items that must be included to not run out of attributes
            items_to_set = np.flatnonzero(skips_left == 0)
            assert len(items_to_set) <= num_items, "Oops, something's wrong"

            # choose remainder from items that may or may not be included here
            n_to_choose = num_items - len(items_to_set)
            items_to_choose_from = np.flatnonzero((num_to_assign > 0) & (skips_left > 0))
            items_to_set = np.append(items_to_set, choose_k(items_to_choose_from, n_to_choose))

            vec[items_to_set] = 1
            num_to_assign[items_to_set] -= 1
            skips_left[np.setdiff1d(range(ITEMS_PER_DOMAIN), items_to_set)] -= 1

        assert np.all(num_to_assign == 0), "Oops, something's wrong"
        new_attr_vecs[:, set_attrs] = attr_vecs_n

    return new_attr_vecs


def _resample_attr_vec_mat(attr_vecs, item_weights=None):
    """
    Given a matrix of attr vecs (y), make a new one that preserves the same distribution of
    attribute frequencies. Proceeding from most to least frequent attribute in the original matrix,
    randomly chooses items to assign to the corresponding attribute in the new matrix. The probability of
    choosing an item can be controlled through item_weights (a list of length ITEMS_PER_DOMAIN; defaults to all equal),
    and goes to zero once the item has been assigned the number of attributes it had in the original matrix.
    """
    if item_weights is None:
        item_weights = np.repeat(1 / ITEMS_PER_DOMAIN, ITEMS_PER_DOMAIN)

    if len(item_weights) != ITEMS_PER_DOMAIN:
        raise ValueError(f'item_p must be a list/tuple of length {ITEMS_PER_DOMAIN}')

    if sum(item_weights) <= 0 or any([wt < 0 for wt in item_weights]):
        raise ValueError('Invalid item weights - must be nonnegative and have positive sum')

    item_weights = torch.tensor(item_weights, dtype=torch.float, device='cpu')

    # info about the y matrix we're basing this on
    item_remaining_attrs = torch.tensor(np.sum(attr_vecs, axis=1), device='cpu')
    attr_freqs = np.sum(attr_vecs, axis=0)
    n_item_seq = np.arange(ITEMS_PER_DOMAIN, 0, -1)  # 8 down to 1
    freq_freqs = [np.sum(attr_freqs == n) for n in n_item_seq]
    attr_starts = np.cumsum(np.concatenate([[0], freq_freqs[:-1]]))

    new_attr_vecs = torch.zeros(attr_vecs.shape, device='cpu')

    for n_items, n_attrs, start_i in zip(n_item_seq, freq_freqs, attr_starts):
        for i_attr in range(start_i, start_i + n_attrs):
            item_weights[item_remaining_attrs == 0] = 0
            these_items = torch.multinomial(item_weights, n_items)
            new_attr_vecs[these_items, i_attr] = 1.
            item_remaining_attrs[these_items] -= 1

    return new_attr_vecs.numpy()


def _scramble_attr_vec_mat(attr_vecs):
    """
    Like _resample_attr_vec_mat with all equal weights, except that it
    does not care about keeping the same # of attributes per item.
    In other words, it just shuffles the items for each attribute.
    """
    attr_freqs = np.sum(attr_vecs, axis=0)
    nonuniform_attr_inds = np.flatnonzero((attr_freqs > 0) & (attr_freqs < ITEMS_PER_DOMAIN))
    attr_vec_tensor = torch.from_numpy(attr_vecs)
    
    for ind in nonuniform_attr_inds:
        perm = torch.randperm(ITEMS_PER_DOMAIN, device='cpu')
        attr_vec_tensor[:, ind] = attr_vec_tensor[perm, ind]
    
    return attr_vecs


def _scramble_attr_vecs_lengthwise(attr_vecs):
    n_attrs = attr_vecs.shape[1]
    for i in range(len(attr_vecs)):
        attr_vecs[i, :] = attr_vecs[i, torch.randperm(n_attrs, device='cpu')]
    
    return attr_vecs


def _take_n_svd_modes_of_attr_vecs(attr_vecs, n_modes):
    u, s, vd = svd(attr_vecs, full_matrices=False)
    n_mode_mat = u[:, :n_modes] @ np.diag(s[:n_modes]) @ vd[:n_modes, :]
    # compress to range [0, 1]
    n_mode_mat -= np.min(n_mode_mat)
    n_mode_mat /= np.max(n_mode_mat)
    return n_mode_mat


def normalize_cluster_info(cluster_info):
    """
    Get a dict that specifies information about the attribute clusters.
    Input is either a string or a dict with keys 'clusters',
    'intragroup_dists', 'intergroup_dist', and 'special'.
    The latter three keys are optional and default to None.
    If input is a string, the part before the first underscore is interpreted as 'clusters'
    and each underscore-separated string after the the first (if any)
    becomes an element of the list in 'special'.
    """
    if isinstance(cluster_info, list):
        return [normalize_cluster_info(clst) for clst in cluster_info]
        
    if isinstance(cluster_info, str):
        clusters, *special = cluster_info.split('_')
        return {'clusters': clusters, 'special': special}
    cluster_info: Dict[str, Any] = {'special': [], **cluster_info}
    return cluster_info


def make_attr_vecs(ctx_per_domain, attrs_per_context, attrs_set_per_item, cluster_info, padding_attrs=0):
    """
    Wrapper to make any set of attr vectors (returns a list of one matrix per context)
    If cluster_info is a list, should be of length ctx_per_domain. In this case the corrresponding
    cluster info is used to generate each context's attributes.
    """    
    if isinstance(cluster_info, list):
        assert len(cluster_info) == ctx_per_domain, (
            f'Length of cluster_info {len(cluster_info)} does not match number of contexts {ctx_per_domain}')
        # Generate attribute vectors for each context individually and concatenate
        return sum([make_attr_vecs(1, attrs_per_context, attrs_set_per_item, cifo) for cifo in cluster_info], [])
    
    attrs_used_per_context = attrs_per_context - padding_attrs
    if attrs_set_per_item > attrs_used_per_context:
        raise ValueError('Cannot set more attrs per item than # allocated to each context')

    cluster_info = normalize_cluster_info(cluster_info)

    # special case for equalized attribute frequency
    if 'eq-freq' in cluster_info['special']:
        if cluster_info['clusters'] != '4-2-2':
            raise ValueError('eq-freq attributes are only defined for 4-2-2 clusters')

        attr_vec_fn = _make_eq_freq_attr_vecs
    elif cluster_info['clusters'] == 'ordering':
        attr_vec_fn = _make_ordering_attr_vecs
    elif cluster_info['clusters'] == 'saxe-ordering':
        attr_vec_fn = _make_saxe_ordering_attr_vecs
    else:
        n_clusts = len(get_cluster_sizes(cluster_info['clusters']))
        try:
            attr_vec_fn = {
                1: _make_ring_attr_vecs,
                2: _make_2_group_attr_vecs,
                3: _make_3_group_attr_vecs
            }[n_clusts]
        except KeyError:
            raise ValueError('Invalid clusters specification')

    attr_vecs = attr_vec_fn(ctx_per_domain, attrs_used_per_context, attrs_set_per_item, **cluster_info)
    
    if match := re.search(r'(\d+)svdmode', '_'.join(cluster_info['special'])):
        n_modes = int(match.group(1))
        attr_vecs = [_take_n_svd_modes_of_attr_vecs(mat, n_modes) for mat in attr_vecs]

    # special case for shuffled attribute-item assignments keeping same mean
    # attr frequency for each item
    if 'shuffled' in cluster_info['special']:
        attr_vecs = [_shuffle_attr_vec_mat(mat) for mat in attr_vecs]

    # "resample" special case, to disrupt hierarchical structure
    if 'resample' in cluster_info['special']:
        try:
            item_weights = cluster_info['resample_weights']
        except KeyError:
            item_weights = None
        attr_vecs = [_resample_attr_vec_mat(mat, item_weights) for mat in attr_vecs]
        
    if 'scramble' in cluster_info['special']:
        attr_vecs = [_scramble_attr_vec_mat(mat) for mat in attr_vecs]
        
    if 'scramble-attrs' in cluster_info['special']:
        attr_vecs = [_scramble_attr_vecs_lengthwise(mat) for mat in attr_vecs]
        
    if 'item_permutation' in cluster_info:
        attr_vecs = [mat[cluster_info['item_permutation'], :] for mat in attr_vecs]
        
    if padding_attrs != 0:
        attr_vecs = [np.concatenate([mat, np.zeros((mat.shape[0], padding_attrs))], axis=1) for mat in attr_vecs]

    return attr_vecs


def make_io_mats(ctx_per_domain=4, attrs_per_context=50, attrs_set_per_item=25, padding_attrs=0,
                 n_domains=4, cluster_info='4-2-2', last_domain_cluster_info=None,
                 repeat_attrs_over_domains=False, share_ctx=False, share_attr_units_in_domain=False,
                 **_extra):
    """
    Make the actual item, context, and attribute matrices, across a given number of domains.
    If one_equidistant is true, replaces the last domain's attrs with equidistant attr vectors.
    Cluster_info and last_domain_cluster_info should be valid inputs to normalize_cluster_info.
    By default (when None), last_domain_clusters is the same as clusters.
    repeat_attrs_over_domains - if True, don't regenerate attrs for each domain, just repeat them.
    share_ctx - if True, use one bank of context units for all domains.
    share_attr_units_in_domain - if True, all contexts in a domain use the same attr units; if not they are disjoint.
    """

    # First make it for a single domain, then use block_diag to replicate.
    item_mat_1 = np.tile(np.eye(ITEMS_PER_DOMAIN), (ctx_per_domain, 1))
    item_mat = block_diag(*[item_mat_1 for _ in range(n_domains)])

    context_mat_1 = np.repeat(np.eye(ctx_per_domain), ITEMS_PER_DOMAIN, axis=0)
    if share_ctx:
        context_mat = np.tile(context_mat_1, (n_domains, 1))
    else:
        context_mat = block_diag(*[context_mat_1 for _ in range(n_domains)])

    if last_domain_cluster_info is None:
        last_domain_cluster_info = ()
    elif not isinstance(last_domain_cluster_info, tuple):
        # have to use tuple to distinguish from different clusters in different domains (list)
        last_domain_cluster_info = (last_domain_cluster_info,)
    
    n_last = len(last_domain_cluster_info)

    if share_attr_units_in_domain:
        def make_domain_attrs(attrs_per_ctx):
            return np.concatenate(attrs_per_ctx, axis=0)
    else:
        def make_domain_attrs(attrs_per_ctx):
            return block_diag(*attrs_per_ctx)

    if repeat_attrs_over_domains:
        uniform_attr_list = make_attr_vecs(ctx_per_domain, attrs_per_context,
                                          attrs_set_per_item, cluster_info, padding_attrs)     
        uniform_attr_mats = [make_domain_attrs(uniform_attr_list)] * (n_domains - n_last)
        
        last_attr_lists = [make_attr_vecs(ctx_per_domain, attrs_per_context,
                                          attrs_set_per_item, ifo, padding_attrs)
                           for ifo in last_domain_cluster_info]
        last_attr_mats = [make_domain_attrs(attr_list) for attr_list in last_attr_lists]
        
        attr_mat = block_diag(*uniform_attr_mats, *last_attr_mats)

    else:
        # New behavior: generate a new set of attr vecs for each domain.
        domain_attr_list = [make_attr_vecs(ctx_per_domain, attrs_per_context,
                                           attrs_set_per_item, cluster_info, padding_attrs)
                            for _ in range(n_domains - n_last)]
        domain_attr_list.extend([make_attr_vecs(ctx_per_domain, attrs_per_context,
                                                attrs_set_per_item, ifo, padding_attrs)
                                 for ifo in last_domain_cluster_info])
        attr_mat = block_diag(*[make_domain_attrs(attrs) for attrs in domain_attr_list])

    return item_mat, context_mat, attr_mat


def plot_item_attributes(ctx_per_domain=4, attrs_per_context=50,
                         attrs_set_per_item=25, cluster_info='4-2-2', io_mats=None, figsize=(12, 6)):
    """Item and context inputs and attribute outputs for each input combination (regardless of domain)"""

    if io_mats is None:
        item_mat, context_mat, attr_mat = make_io_mats(ctx_per_domain, attrs_per_context, attrs_set_per_item,
                                                       n_domains=1, cluster_info=cluster_info)
    else:
        item_mat, context_mat, attr_mat = io_mats

    fig = plt.figure(figsize=figsize)

    # plot 1: items
    ax1 = fig.add_subplot(1, 20, (1, 3))
    ax1.set_title('Items')
    ax1.imshow(item_mat, aspect='auto', interpolation='nearest')
    ax1.set_yticks([])
    ax1.set_xticks(range(0, ITEMS_PER_DOMAIN, 2))

    # plot 2: contexts
    ax2 = fig.add_subplot(1, 20, 4)
    ax2.set_title('Contexts')
    ax2.imshow(context_mat, aspect='auto', interpolation='nearest')
    ax2.set_yticks([])
    ax2.set_xticks(range(0, context_mat.shape[1], 2))

    # plot 3: attributes
    ax3 = fig.add_subplot(1, 20, (5, 20))
    ax3.set_title('Attributes')
    ax3.imshow(attr_mat, aspect='auto', interpolation='nearest')
    ax3.set_yticks([])
    ax3.set_xticks(range(0, attr_mat.shape[1], 10))

    return fig, (ax1, ax2, ax3)


def get_item_attribute_rdm(ctx_per_domain=4, attrs_per_context=50, attrs_set_per_item=25,
                           cluster_info='4-2-2', metric='cityblock'):
    """Make RDM of similarities between the items' attributes, collapsed across contexts"""
    attrs = make_attr_vecs(ctx_per_domain, attrs_per_context, attrs_set_per_item, cluster_info)
    return pa.get_attribute_rdm(attrs, metric=metric)


def plot_item_attribute_dendrogram(ax=None, ctx_per_domain=4, attrs_per_context=50, attrs_set_per_item=25,
                                   cluster_info='4-2-2', method='single', metric='euclidean', rel_color_thresh=0.6, **_extra):
    """Dendrogram of similarities between the items' attributes, collapsed across contexts"""

    dist_mat = get_item_attribute_rdm(ctx_per_domain, attrs_per_context,
                                      attrs_set_per_item, cluster_info, metric=metric)
    condensed_dist = distance.squareform(dist_mat)

    item_names = get_items(n_domains=1, cluster_info=cluster_info)[1]

    if ax is None:
        fig, ax = plt.subplots()

    z = hierarchy.linkage(condensed_dist, method=method)
    with plt.rc_context({'lines.linewidth': 2.5}):
        dgram = hierarchy.dendrogram(z, ax=ax, orientation='right', distance_sort='ascending', labels=np.array(item_names),
                                     color_threshold=rel_color_thresh * max(z[:, 2]), above_threshold_color='black')
    ax.set_title('Item attribute dissimilarity, collapsed across contexts')
    ax.set_xlabel(f'{metric.capitalize()} distance')
    ax.set_ylabel('Input #')
    return dgram, ax


def item_group(n=slice(None), clusters='4-2-2', **_extra):
    """Equivalent of circle/square/star, to identify 'types' of items"""
    if isinstance(clusters, dict):
        clusters = clusters['clusters']
    cluster_sizes = get_cluster_sizes(clusters)
    groups = [i for i, ksize in enumerate(cluster_sizes) for _ in range(ksize)]
    return groups[n]


def item_group_symbol(n, clusters='4-2-2'):
    if isinstance(clusters, dict):
        clusters = clusters['clusters']
    if len(get_cluster_sizes(clusters)) == 1:
        return ''

    symbol_array = np.array(['\u26ab',   # circle
                             '\u25aa',   # square
                             '\u2605'])  # star
    return symbol_array[item_group(n, clusters)]


def domain_name(n):
    return chr(ord('A') + n)


def get_domain_colors():
    """For plotting MDS trajectories. From 'saltwater taffy' palette in Illustrator."""
    return [
        '#aadff1',
        '#fbb57f',
        '#f59db2',
        '#80caa5',
        '#fbed91'
    ]


def get_items(train_only=False, n_domains=4, n_train_domains=None,
              cluster_info='4-2-2', last_domain_cluster_info=None,
              device=None, **_extra):
    """Get item tensors (without repetitions) and their corresponding names"""
    if last_domain_cluster_info is None:
        last_domain_cluster_info = ()
    elif not isinstance(last_domain_cluster_info, tuple):
        last_domain_cluster_info = (last_domain_cluster_info,)
        
    n_last = len(last_domain_cluster_info)
    n_prelast = n_domains - n_last
    
    cluster_info = normalize_cluster_info(cluster_info)
    last_domain_cluster_info = [normalize_cluster_info(ldci) for ldci in last_domain_cluster_info]
    
    if isinstance(cluster_info, list):
        # Can't assign symbols because the clusters vary over contexts
        cluster_info = {'clusters': '8'}
        
    for i, ldci in enumerate(last_domain_cluster_info):
        if isinstance(ldci, list):
            last_domain_cluster_info[i] = {'clusters': '8'}
    
    if train_only and n_train_domains is not None:
        n_domains = n_train_domains

    items = torch.eye(ITEMS_PER_DOMAIN * n_domains, device=device)
    all_clusters = [cluster_info['clusters']] * n_domains
    for i, ldci in zip(range(n_prelast, n_domains), last_domain_cluster_info):
        all_clusters[i] = ldci['clusters']

    item_names = [domain_name(d) + str(n + 1) + item_group_symbol(n, clst)
                  for d, clst in enumerate(all_clusters)
                  for n in range(ITEMS_PER_DOMAIN)]
    return items, item_names


def get_contexts(train_only=False, n_domains=4, n_train_domains=None,
                 ctx_per_domain=4, share_ctx=False, device=None, **_extra):
    """Get context tensors (without repetitions) and their corresponding names"""
    if train_only and n_train_domains is not None:
        n_domains = n_train_domains
    
    if share_ctx:
        contexts = torch.eye(ctx_per_domain, device=device)
        context_names = [str(n + 1) for n in range(ctx_per_domain)]
    else:
        contexts = torch.eye(ctx_per_domain * n_domains, device=device)
        context_names = [domain_name(d) + str(n + 1) for d in range(n_domains) for n in range(ctx_per_domain)]
    return contexts, context_names
