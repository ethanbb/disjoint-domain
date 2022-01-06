"""
Functions for analyzing a problem domain to be learned by a network.
These functions deal with matrices of items, contexts, and attributtes
(for family tree, these are a.k.a. person1s, relations, and person2s).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.linalg import svd
from scipy.spatial import distance
import torch


def get_attribute_rdm(attr_mats, metric='cityblock'):
    """
    Computes a dissimilarity matrix of the given attribute vector matrix.
    This function does not know what item each vector belongs to - it computes dissimilarity between individual inputs.
    If a list of matrices are provided (e.g., one for each context), averages dissimilarities across the list.
    """
    if isinstance(attr_mats, np.ndarray):
        attr_mats = [attr_mats]
    # noinspection PyTypeChecker
    mean_dist = np.nanmean(np.stack([distance.pdist(a, metric=metric) for a in attr_mats]), axis=0)
    return distance.squareform(mean_dist)


def get_mean_attr_freqs(item_mat, attr_mat):
    """Make a vector of how frequent each item's attributes are on average"""
    # collapse attribute matrix over contexts
    y_collapsed = item_mat.T @ attr_mat

    # for each item, combine attributes with frequency of each attribute
    attr_freq = np.sum(y_collapsed, axis=0)
    return (y_collapsed @ attr_freq) / np.sum(y_collapsed, axis=1)


def get_attr_freq_dist_mat(item_mat, attr_mat):
    """Make an n_items x n_items matrix of how much pairs of items differ on mean atribute frequency"""
    item_mean_attr_freqs = get_mean_attr_freqs(item_mat, attr_mat)
    return np.abs(item_mean_attr_freqs[np.newaxis, :] - item_mean_attr_freqs[:, np.newaxis])


def get_contextfree_io_corr_matrix(item_mat, attr_mat):
    """
    Computes the input-output correlation matrix for given items and attributes (defined in Saxe paper)
    This version collapses relations across contexts (treats context as irrelevant).
    Output is n_attributes x n_items.
    """
    return (attr_mat.T @ item_mat) / item_mat.shape[0]


def get_contextual_io_corr_matrices(item_mat, ctx_mat, attr_mat):
    """
    Computes the I/O correlation matrix of the given items and attributes within each context.
    Output is n_contexts x n_attributes x n_items.
    """
    ctx_masks = ctx_mat.T > 0
    item_mat_per_ctx = [item_mat[mask, :] for mask in ctx_masks]
    attr_mat_per_ctx = [attr_mat[mask, :] for mask in ctx_masks]
    io_corrs_per_ctx = [get_contextfree_io_corr_matrix(i, a) for i, a in zip(item_mat_per_ctx, attr_mat_per_ctx)]
    return np.stack(io_corrs_per_ctx)


def split_and_trim_matrix(matrix, sizes_or_sections, axis=0):
    """
    Given a matrix, splits it into even (if given an int) or uneven sections along axis,
    then takes along the other axis only where at least one entry is nonzero.
    If sizes_or_sections is an iterable, it specifies the size of each section.
    """
    if axis == 1:
        matrix = matrix.T

    split_ax_len = len(matrix)
    if np.isscalar(sizes_or_sections):
        if split_ax_len % sizes_or_sections != 0:
            raise ValueError(f'Cannot evenly split matrix into {sizes_or_sections} parts')
        indices_or_sections = sizes_or_sections
    else:
        if sum(sizes_or_sections) != split_ax_len:
            raise ValueError('Given sections do not add up to size of axis to split')
        indices_or_sections = np.cumsum(sizes_or_sections)[:-1]

    submats_untrimmed = np.split(matrix, indices_or_sections)
    submats = [mat[:, mat.any(axis=0)] for mat in submats_untrimmed]

    if axis == 1:
        submats = [mat.T for mat in submats]
    return submats


def get_io_corr_matrices(full_item_mat, full_ctx_mat, full_attr_mat, domains=4, contextual=False):
    """
    Splits the given io matrices into domains, according to 'domains' which is either the number of domains with an
    equal number of items in each or a list of the number of items in each domain, and then computes the contextual or
    context-free I/O correlation matrix for each domain.
    """
    # First split the item matrix according to 'domains', then use that to split the other matrices
    item_mats = split_and_trim_matrix(full_item_mat, domains, axis=1)
    domain_x_lens = [len(mat) for mat in item_mats]
    attr_mats = split_and_trim_matrix(full_attr_mat, domain_x_lens, axis=0)

    if contextual:
        if full_ctx_mat is None:
            raise ValueError('Context matrix must be provided for contextual io corr matrices')
        ctx_mats = split_and_trim_matrix(full_ctx_mat, domain_x_lens, axis=0)
        return [get_contextual_io_corr_matrices(i, c, a) for i, c, a in zip(item_mats, ctx_mats, attr_mats)]
    else:
        return [get_contextfree_io_corr_matrix(i, a) for i, a in zip(item_mats, attr_mats)]


def get_nth_signflip_mat(size, n):
    """
    Utility to map integers in [0, 2^size-1] onto matrices with 1s and -1s along the diagonal.
    """
    assert 0 <= n < 2 ** size, 'n out of range'
    bit_array = 1 << np.arange(size)
    b_flip_entry = (n & bit_array) > 0
    return np.eye(size) - 2 * np.diag(b_flip_entry)


def get_item_loadings_from_corr_mat(io_corr_mat, center=False):
    """Returns an n_modes x n_items loading matrix"""
    # randomly permute items when doing SVD to prevent bias
    item_perm = torch.randperm(io_corr_mat.shape[1], device='cpu')

    if center:
        io_corr_mat = io_corr_mat - np.mean(io_corr_mat, axis=1, keepdims=True)

    # noinspection PyTupleAssignmentBalance
    _, s, vh = svd(io_corr_mat[:, item_perm], full_matrices=False)
    vh_scaled = np.empty_like(vh)
    vh_scaled[:, item_perm] = np.diag(s) @ vh
    return vh_scaled


def get_item_svd_loadings(item_mat, attr_mat, domains, center=False):
    """
    Computes SVD V-matrices for each item in each domain and concatenates them along the item dimension (each row is an
    item). The rows of the resulting matrix can be compared, e.g. with cityblock distance, to quantify differences in
    'hierarchical role'.

    Note that this does not involve computing SVD on the *full* (multi-domain) I/O matrix. Thus it does not reflect
    what a linear network would learn when trained on all domains simultaneously. It rather allows easy comparison of
    the individual SVD loadings of each domain.
    """
    corr_mats = get_io_corr_matrices(item_mat, None, attr_mat, domains, contextual=False)
    items_per = corr_mats[0].shape[1]
    svd_loading_list = []
    for i, corr_mat in enumerate(corr_mats):
        vh_scaled = get_item_loadings_from_corr_mat(corr_mat, center=center)

        if i == 0:
            signflip_mat = np.eye(items_per)
        else:
            # resolve sign ambiguity based on item correlation up to item permutation... brute force technique
            first_domain_v = svd_loading_list[0]
            best_signflip_mat_n = -1
            best_total_item_corr = -1  # at least half the max total corrs must be >= 0, so this is safe

            for n in range(2 ** items_per):
                curr_signflip_mat = get_nth_signflip_mat(items_per, n)
                item_corr_mat = first_domain_v @ curr_signflip_mat @ vh_scaled
                # find permutation of columns (2nd items)
                row_ind, col_ind = linear_sum_assignment(item_corr_mat, maximize=True)
                total_item_corr = item_corr_mat[row_ind, col_ind].sum()
                if total_item_corr > best_total_item_corr:
                    best_total_item_corr = total_item_corr
                    best_signflip_mat_n = n

            signflip_mat = get_nth_signflip_mat(items_per, best_signflip_mat_n)
            # end non-first-domain case
        svd_loading_list.append(vh_scaled.T @ signflip_mat)
        # end loop over domains
    return np.concatenate(svd_loading_list, axis=0)


def get_contextfree_item_svd_dist(item_mat, attr_mat, n_domains, modes_to_use=slice(None), metric='cityblock'):
    """
    Compares "hierarchical role" based on SVD loadings that have been aligned across domains using the previous function
    """
    loading_mat = get_item_svd_loadings(item_mat, attr_mat, n_domains, center=True)
    loading_mat = loading_mat[:, modes_to_use]
    if metric == 'corr':
        return np.corrcoef(loading_mat)
    else:
        return distance.squareform(distance.pdist(loading_mat, metric=metric))


def get_contextfree_item_svd_corr(item_mat, attr_mat, n_domains, modes_to_use=slice(None)):
    return get_contextfree_item_svd_dist(item_mat, attr_mat, n_domains, modes_to_use, metric='corr')


def get_1domain_contextual_item_svd_dist(item_mat, ctx_mat, attr_mat, modes_to_use=slice(None), metric='cityblock'):
    """
    With items, contexts and attributes from just 1 domain/tree, computes the distance between items
    averaged over contexts, optionally using only a subset of SVD modes.
    """
    contextual_io_corr_mats = get_io_corr_matrices(item_mat, ctx_mat, attr_mat, domains=1, contextual=True)[0]
    loading_mats = [get_item_loadings_from_corr_mat(mat, center=True).T for mat in contextual_io_corr_mats]
    loading_mats = [mat[:, modes_to_use] for mat in loading_mats]
    if metric == 'corr':
        loading_dists = [np.corrcoef(mat) for mat in loading_mats]
    else:
        loading_dists = [distance.squareform(distance.pdist(mat, metric=metric)) for mat in loading_mats]
    return np.mean(loading_dists, axis=0)


def get_1domain_contextual_item_svd_corr(item_mat, ctx_mat, attr_mat, modes_to_use=slice(None)):
    return get_1domain_contextual_item_svd_dist(item_mat, ctx_mat, attr_mat, modes_to_use, metric='corr')
