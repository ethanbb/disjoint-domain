import numpy as np
from scipy.linalg import block_diag, svd
from scipy.cluster import hierarchy
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import torch
from typing import Dict, Any

ITEMS_PER_DOMAIN = 8


def choose_k_inds(n, k=1):
    """Get random permutation of k indices in range(n)"""
    if k > n:
        raise ValueError(f'Cannot pick {k} unique indices from range({n})')
    return torch.randperm(n, device='cpu')[:k]


def choose_k(a, k=1):
    """
    Get permutation of k items from a using PyTorch
    (want to make sure we use the same rng for everything so it's deterministic given the seed)
    """
    # need to use .numpy() here, or else a vector of length 1 will become a scalar
    return a[choose_k_inds(len(a), k).numpy()]


def choose_k_set_bits(a, k=1):
    """Get permutation of k indices of a that are set (a should be a boolean array)"""
    return choose_k(np.flatnonzero(a), k)


def get_cluster_sizes(clusters):
    """Parse a 'clusters' string such as '4-2-2_suffix' to get the size of each cluster"""
    sizes = [int(sz) for sz in clusters.partition('_')[0].split('-')]
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
                            clusters='4-2-2', intragroup_dists=None, intergroup_dist=40, **_extra):
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

        # now choose bits to flip for each of the 'circle' vectors, keeping total # set the same
        attr_mat[:n_circles] = _make_n_dist_d_attr_vecs(circ_centroid, n_circles, circ_dist)

        # pick centroid for other items, which should be 40 bits away from this centroid.
        # (or overridden by setting intergroup_dist)
        other_centroid = circ_centroid.copy()
        other_centroid[choose_k(circ_centroid_unset, intergroup_dist // 2)] = 1
        other_centroid[choose_k(circ_centroid_set, intergroup_dist // 2)] = 0

        # now square and star centroids, which are centered on other_centroid and differ by 12 bits (by default)
        square_centroid, star_centroid = _make_n_dist_d_attr_vecs(other_centroid, 2, sqst_dist)

        # squares differ by just 2 bits (by default). be a little imprecise and let one of them be the centroid.
        attr_mat[n_circles + np.arange(n_squares)] = _make_n_dist_d_attr_vecs(square_centroid, n_squares, square_dist)

        # stars differ by 10 bits (by default).
        # again be a little imprecise, let one differ from centroid by 4 and the other by 6 (all unique)
        attr_mat[-n_stars:] = _make_n_dist_d_attr_vecs(star_centroid, n_stars, star_dist)

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


def _make_equidistant_attr_vecs(ctx_per_domain, attrs_per_context, attrs_set_per_item,
                                intragroup_dists=None, **_extra):
    """
    Make attribute vectors that are all equidistant from each other with Hamming distance `dist`. 
    attrs_per_context must be at least attrs_set_per_item + (dist/2) * (ITEMS_PER_DOMAIN-1)
    (by default, at least 60). Also, dist must be even.
    """
    if intragroup_dists is None:
        dist = 10
    else:
        dist = intragroup_dists[0]

    if dist % 2 != 0:
        raise ValueError('dist must be even')
    half_dist = dist // 2

    if attrs_per_context < attrs_set_per_item + half_dist * (ITEMS_PER_DOMAIN - 1):
        raise ValueError(f'Need more attrs to get equidistant vecs with distance {dist}')

    n_rot = half_dist * ITEMS_PER_DOMAIN  # portion of vector that rotates for each item
    n_fixed_set = attrs_set_per_item - half_dist

    attrs = [np.zeros((ITEMS_PER_DOMAIN, attrs_per_context)) for _ in range(ctx_per_domain)]

    for attr_mat in attrs:
        # pick fixed set and rotating indices
        fixed_set_and_rot_inds = choose_k_inds(attrs_per_context, n_fixed_set + n_rot)
        fixed_set_inds, rot_inds = fixed_set_and_rot_inds.split([n_fixed_set, n_rot])
        rot_inds_each = rot_inds.split(half_dist)

        # set fixed indices for all items and rotating indices for each individually
        attr_mat[:, fixed_set_inds] = 1
        for attr_vec, inds in zip(attr_mat, rot_inds_each):
            attr_vec[inds] = 1

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
    if isinstance(cluster_info, str):
        clusters, *special = cluster_info.split('_')
        return {'clusters': clusters, 'special': special}
    cluster_info: Dict[str, Any] = {'special': [], **cluster_info}
    return cluster_info


def make_attr_vecs(ctx_per_domain, attrs_per_context, attrs_set_per_item, cluster_info):
    """Wrapper to make any set of attr vectors (returns a list of one matrix per context)"""
    if attrs_set_per_item > attrs_per_context:
        raise ValueError('Cannot set more attrs per item than # allocated to each context')

    cluster_info = normalize_cluster_info(cluster_info)

    # special case for equalized attribute frequency
    if 'eq-freq' in cluster_info['special']:
        if cluster_info['clusters'] != '4-2-2':
            raise ValueError('eq-freq attributes are only defined for 4-2-2 clusters')

        attr_vec_fn = _make_eq_freq_attr_vecs
    else:
        n_clusts = len(get_cluster_sizes(cluster_info['clusters']))
        try:
            attr_vec_fn = {
                1: _make_equidistant_attr_vecs,
                2: _make_2_group_attr_vecs,
                3: _make_3_group_attr_vecs
            }[n_clusts]
        except KeyError:
            raise ValueError('Invalid clusters specification')

    attr_vecs = attr_vec_fn(ctx_per_domain, attrs_per_context, attrs_set_per_item, **cluster_info)

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

    return attr_vecs


def make_io_mats(ctx_per_domain=4, attrs_per_context=50, attrs_set_per_item=25,
                 n_domains=4, cluster_info='4-2-2', last_domain_cluster_info=None,
                 repeat_attrs_over_domains=False, share_ctx=False, **_extra):
    """
    Make the actual item, context, and attribute matrices, across a given number of domains.
    If one_equidistant is true, replaces the last domain's attrs with equidistant attr vectors.
    Cluster_info and last_domain_cluster_info should be valid inputs to normalize_cluster_info.
    By default (when None), last_domain_clusters is the same as clusters.
    repeat_attrs_over_domains - if True, don't regenerate attrs for each domain, just repeat them.
    share_ctx - if True, use one bank of context units for all domains.
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
        last_domain_cluster_info = cluster_info
        last_is_same = True
    else:
        last_is_same = False

    cluster_info = normalize_cluster_info(cluster_info)
    last_domain_cluster_info = normalize_cluster_info(last_domain_cluster_info)

    if repeat_attrs_over_domains:
        domain_attrs = make_attr_vecs(ctx_per_domain, attrs_per_context,
                                      attrs_set_per_item, cluster_info)
        if last_is_same:
            attr_mat = block_diag(*([block_diag(*domain_attrs)] * n_domains))
        else:
            domain_attrs_last = make_attr_vecs(ctx_per_domain, attrs_per_context,
                                               attrs_set_per_item, last_domain_cluster_info)
            attr_mat = block_diag(*([block_diag(*domain_attrs)] * (n_domains - 1) + [block_diag(*domain_attrs_last)]))
    else:
        # New behavior: generate a new set of attr vecs for each domain.
        domain_attrs = [make_attr_vecs(ctx_per_domain, attrs_per_context,
                                       attrs_set_per_item, cluster_info)
                        for _ in range(n_domains - 1)]
        domain_attrs.append(make_attr_vecs(ctx_per_domain, attrs_per_context,
                                           attrs_set_per_item, last_domain_cluster_info))
        attr_mat = block_diag(*[block_diag(*attrs) for attrs in domain_attrs])

    return item_mat, context_mat, attr_mat


def get_mean_attr_freqs(item_mat, attr_mat):
    # collapse attribute matrix over contexts
    y_collapsed = item_mat.T @ attr_mat
    
    # for each item, combine attributes with frequency of each attribute
    attr_freq = np.sum(y_collapsed, axis=0)
    return (y_collapsed @ attr_freq) / np.sum(y_collapsed, axis=1)


def get_attr_freq_dist_mat(item_mat, attr_mat):
    item_mean_attr_freqs = get_mean_attr_freqs(item_mat, attr_mat)
    return np.abs(item_mean_attr_freqs[np.newaxis, :] - item_mean_attr_freqs[:, np.newaxis])


def get_io_corr_matrix(item_mat, attr_mat, n_domains):
    """
    Computes the input-output correlation matrix for each domain (defined in Saxe paper)
    The result is a list of attrs_per_context x ITEMS_PER_DOMAIN matrices.
    """
    corr_mats = []

    for i, (item_slab, attr_slab) in enumerate(zip(np.split(item_mat, n_domains), np.split(attr_mat, n_domains))):
        item_submat = np.split(item_slab, n_domains, axis=1)[i]
        attr_submat = np.split(attr_slab, n_domains, axis=1)[i]
        corr_mats.append(attr_submat.T @ (item_submat / item_submat.shape[0]))

    return corr_mats


def get_nth_signflip_mat(size, n):
    """
    Utility to map integers in [0, 2^size-1] onto matrices with 1s and -1s along the diagonal.
    """
    assert 0 <= n < 2 ** size, 'n out of range'
    bit_array = 1 << np.arange(size)
    b_flip_entry = (n & bit_array) > 0
    return np.eye(size) - 2 * np.diag(b_flip_entry)


def get_item_svd_loadings(item_mat, attr_mat, n_domains):
    """
    Computes SVD V-matrices for each item in each domain and concatenates them along the item dimension (each row is an
    item). The rows of the resulting matrix can be compared, e.g. with cityblock distance, to quantify differences in
    'hierarchical role'.
    """
    corr_mats = get_io_corr_matrix(item_mat, attr_mat, n_domains)
    svd_loading_list = []
    for i, corr_mat in enumerate(corr_mats):
        # randomly permute items when doing SVD to prevent bias
        item_perm = torch.randperm(ITEMS_PER_DOMAIN, device='cpu')
        # noinspection PyTupleAssignmentBalance
        _, s, vh = svd(corr_mat[:, item_perm], full_matrices=False)
        vh_scaled = np.empty_like(vh)
        vh_scaled[:, item_perm] = np.diag(s) @ vh

        if i == 0:
            signflip_mat = np.eye(ITEMS_PER_DOMAIN)
        else:
            # resolve sign ambiguity based on item correlation up to item permutation... brute force technique
            first_domain_v = svd_loading_list[0]
            best_signflip_mat_n = -1
            best_total_item_corr = -1  # at least half the max total corrs must be >= 0, so this is safe

            for n in range(2 ** ITEMS_PER_DOMAIN):
                curr_signflip_mat = get_nth_signflip_mat(ITEMS_PER_DOMAIN, n)
                item_corr_mat = first_domain_v @ curr_signflip_mat @ vh_scaled
                # find permutation of columns (2nd items)
                row_ind, col_ind = linear_sum_assignment(item_corr_mat, maximize=True)
                total_item_corr = item_corr_mat[row_ind, col_ind].sum()
                if total_item_corr > best_total_item_corr:
                    best_total_item_corr = total_item_corr
                    best_signflip_mat_n = n

            signflip_mat = get_nth_signflip_mat(ITEMS_PER_DOMAIN, best_signflip_mat_n)
            # end non-first-domain case
        svd_loading_list.append(vh_scaled.T @ signflip_mat)
        # end loop over domains
    return np.concatenate(svd_loading_list, axis=0)


def get_item_svd_dist_mat(item_mat, attr_mat, n_domains):
    loading_mat = get_item_svd_loadings(item_mat, attr_mat, n_domains)
    return distance.squareform(distance.pdist(loading_mat, metric='cityblock'))


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

    return ax1, ax2, ax3

def get_item_attribute_rdm(ctx_per_domain=4, attrs_per_context=50, attrs_set_per_item=25,
                           cluster_info='4-2-2', metric='euclidean'):
    """Make RDM of similarities between the items' attributes, collapsed across contexts"""

    attrs = make_attr_vecs(ctx_per_domain, attrs_per_context, attrs_set_per_item, cluster_info)
    mean_dist = np.mean(np.stack([distance.pdist(a, metric=metric) for a in attrs]), axis=0)
    return distance.squareform(mean_dist)


def plot_item_attribute_dendrogram(ax=None, ctx_per_domain=4, attrs_per_context=50, attrs_set_per_item=25,
                                   cluster_info='4-2-2', method='single', metric='euclidean', **_extra):
    """Dendrogram of similarities between the items' attributes, collapsed across contexts"""

    dist_mat = get_item_attribute_rdm(ctx_per_domain, attrs_per_context,
                                      attrs_set_per_item, cluster_info, metric=metric)
    condensed_dist = distance.squareform(dist_mat)

    item_names = get_items(n_domains=1, cluster_info=cluster_info)[1]

    if ax is None:
        fig, ax = plt.subplots()

    z = hierarchy.linkage(condensed_dist, method=method)
    with plt.rc_context({'lines.linewidth': 2.5}):
        hierarchy.dendrogram(z, ax=ax, orientation='right', color_threshold=0.6 * max(z[:, 2]),
                             distance_sort='ascending', labels=np.array(item_names))
    ax.set_title('Item attribute dissimilarity, collapsed across contexts')
    ax.set_xlabel(f'{metric.capitalize()} distance')
    ax.set_ylabel('Input #')
    return ax


def init_torch(device=None, torchfp=None, use_cuda_if_possible=True):
    """Establish floating-point type and device to use with PyTorch"""

    if device is None:
        if use_cuda_if_possible and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)

    if device.type == 'cuda':
        ttype_ns = torch.cuda
    else:
        ttype_ns = torch

    if torchfp is None:
        torchfp = torch.float

    if torchfp == torch.float:
        torch.set_default_tensor_type(ttype_ns.FloatTensor)
    elif torchfp == torch.double:
        torch.set_default_tensor_type(ttype_ns.DoubleTensor)
    else:
        raise NotImplementedError(f'No tensor type known for dtype {torchfp}')

    return device, torchfp


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


def get_items(n_domains=4, cluster_info='4-2-2', last_domain_cluster_info=None, **_extra):
    """Get item tensors (without repetitions) and their corresponding names"""
    cluster_info = normalize_cluster_info(cluster_info)
    last_domain_cluster_info = (cluster_info if last_domain_cluster_info is None
                                else normalize_cluster_info(last_domain_cluster_info))

    items = torch.eye(ITEMS_PER_DOMAIN * n_domains)
    all_clusters = [cluster_info['clusters']] * n_domains
    if last_domain_cluster_info is not None:
        last_domain_cluster_info = normalize_cluster_info(last_domain_cluster_info)
        all_clusters[-1] = last_domain_cluster_info['clusters']

    item_names = [domain_name(d) + str(n + 1) + item_group_symbol(n, clst)
                  for d, clst in enumerate(all_clusters)
                  for n in range(ITEMS_PER_DOMAIN)]
    return items, item_names


def get_contexts(n_domains=4, ctx_per_domain=4, share_ctx=False, **_extra):
    """Get context tensors (without repetitions) and their corresponding names"""
    if share_ctx:
        contexts = torch.eye(ctx_per_domain)
        context_names = [str(n + 1) for n in range(ctx_per_domain)]
    else:
        contexts = torch.eye(ctx_per_domain * n_domains)
        context_names = [domain_name(d) + str(n + 1) for d in range(n_domains) for n in range(ctx_per_domain)]
    return contexts, context_names


def get_net_dims(n_domains=4, ctx_per_domain=4, attrs_per_context=50, attrs_set_per_item=25, **_extra):
    """Get some basic facts about the default architecture, possibly with overrides"""
    n_items = ITEMS_PER_DOMAIN * n_domains
    n_ctx = ctx_per_domain * n_domains

    return ctx_per_domain, n_domains, n_items, n_ctx, attrs_per_context, attrs_set_per_item


def calc_snap_epochs(snap_freq, snap_freq_scale, num_epochs):
    """Given the possibility of taking snapshots on a log scale, get the actual snapshot epochs"""
    if snap_freq_scale == 'log':
        snap_epochs = np.arange(0, np.log2(num_epochs), snap_freq)
        snap_epochs = np.exp2(snap_epochs)
    elif snap_freq_scale == 'lin':
        snap_epochs = np.arange(0, num_epochs, snap_freq)
    else:
        raise ValueError(f'Unkonwn snap_freq_scale {snap_freq_scale}')

    return list(np.unique(np.round(snap_epochs).astype(int)))
