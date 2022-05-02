"""
Functions for analyzing a problem domain to be learned by a network.
These functions deal with matrices of items, contexts, and attributtes
(for family tree, these are a.k.a. person1s, relations, and person2s).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.linalg import svd
from scipy.spatial import distance
from scipy.special import k0
from scipy.constants import pi
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
    if metric == 'corr':
        def distfn(mat):
            return np.corrcoef(mat)
    elif metric == 'covar_centered':
        def distfn(mat):
            iomat_centered = (mat - np.mean(mat, axis=1, keepdims=True)) / len(mat)
            return iomat_centered @ iomat_centered.T
    else:
        def distfn(mat):
            return distance.squareform(distance.pdist(mat, metric=metric))
    mean_dist = np.nanmean(np.stack([distfn(a) for a in attr_mats]), axis=0)
    return mean_dist


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
    return (attr_mat.T @ item_mat) / len(item_mat)


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
        indices_or_sections = np.cumsum(sizes_or_sections)

    submats_untrimmed = np.split(matrix, indices_or_sections)
    if not np.isscalar(sizes_or_sections):
        # discard the last section, which is empty if the list of sizes adds up to the size of the dimension
        submats_untrimmed = submats_untrimmed[:-1]
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
    # Update: realized the way I was doing this was kind of broken since there are attributes
    # that are zero for all items. Instead let's depend on the fact that the # of attributes per
    # domain is not likely to ever be unequal.
    item_mats = split_and_trim_matrix(full_item_mat, domains, axis=1)
    attr_mats = split_and_trim_matrix(full_attr_mat, domains, axis=1)

    if contextual:
        if full_ctx_mat is None:
            raise ValueError('Context matrix must be provided for contextual io corr matrices')
        domain_x_lens = [len(item_mat) for item_mat in item_mats]
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


def corr_mat_svd(io_corr_mat, center=False):
    # randomly permute items when doing SVD to prevent bias
    item_perm = torch.randperm(io_corr_mat.shape[1], device='cpu')

    if center:
        io_corr_mat = io_corr_mat - np.mean(io_corr_mat, axis=0, keepdims=True)

    # noinspection PyTupleAssignmentBalance
    u, s, vh_permuted = svd(io_corr_mat[:, item_perm], full_matrices=False)
    vh = np.empty_like(vh_permuted)
    vh[:, item_perm] = vh_permuted
    return u, s, vh


def get_item_loadings_from_corr_mat(io_corr_mat, center=False):
    """Returns an n_modes x n_items scaled loading matrix"""
    _, s, vh = corr_mat_svd(io_corr_mat, center=center)
    return np.diag(s) @ vh


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
    center = False if metric == 'covar' else True
    loading_mat = get_item_svd_loadings(item_mat, attr_mat, n_domains, center=center)
    loading_mat = loading_mat[:, modes_to_use]
    if metric == 'corr':
        return np.corrcoef(loading_mat)
    elif metric == 'covar_centered' or metric == 'covar':
        return loading_mat @ loading_mat.T
    else:
        return distance.squareform(distance.pdist(loading_mat, metric=metric))


def get_contextfree_item_svd_corr(item_mat, attr_mat, n_domains, modes_to_use=slice(None)):
    return get_contextfree_item_svd_dist(item_mat, attr_mat, n_domains, modes_to_use, metric='corr')


def get_contextfree_item_svd_covar(item_mat, attr_mat, n_domains, modes_to_use=slice(None)):
    return get_contextfree_item_svd_dist(item_mat, attr_mat, n_domains, modes_to_use, metric='covar_centered')


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


def get_io_corr_basis_mats_and_svs(item_mat, attr_mat):
    """Do SVD to obtain a set of basis (rank-1) matrices for the input-output correlation with corresponding SVs."""
    io_corr_mat = get_contextfree_io_corr_matrix(item_mat, attr_mat)
    u, s, vh = corr_mat_svd(io_corr_mat, center=False)
    basis_mats = np.einsum('ij,jk->jik', u, vh)
    return basis_mats, s


def get_linearmodel_sv_trajectory(sv, tau, a0, t):
    """Formula from Saxe et al., 2019"""
    expterm = np.exp(2 * sv * t/tau)
    return (sv * expterm) / (expterm - 1 + sv/a0)


def get_effective_svs(io_corr_mat, basis_mats):
    """
    Project given empirical I/O correlation matrix onto each basis matrix to obtain the effective SVs a_1,...a_N1.
    """
    return np.sum(basis_mats * io_corr_mat, axis=(-2, -1))


def basis_and_svs_to_expected_svs_at_epochs(basis_mats, svs, a0s, tau, epochs):
    return np.stack([get_linearmodel_sv_trajectory(sv, tau, a0, epochs) for sv, a0 in zip(svs, a0s)])


def get_expected_svs_at_epochs(item_mat, attr_mat, a0s, tau, epochs, modes_to_use=slice(None)):
    basis_mats, svs = get_io_corr_basis_mats_and_svs(item_mat, attr_mat)
    basis_mats = basis_mats[modes_to_use]
    svs = svs[modes_to_use]
    return basis_and_svs_to_expected_svs_at_epochs(basis_mats, svs, a0s, tau, epochs)
    

def plot_sv_trajectories(ax, item_mat, attr_mat, a0s, tau, n_epochs, modes_to_use=slice(None)):
    """Make a plot of expected change in each singular value over epochs, like Saxe et al. figure 3C"""
    t = np.arange(n_epochs)
    trajs = get_expected_svs_at_epochs(item_mat, attr_mat, a0s, tau, t, modes_to_use=modes_to_use)
    mode_nums = np.arange(len(trajs))[modes_to_use] + 1
    
    for traj, num in zip(trajs, mode_nums):
        ax.plot(t, traj, label=f'Mode {num}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Effective SV')
    ax.legend()


def basis_and_svs_to_expected_iomats_at_epochs(basis_mats, svs, tau, epochs):
    trajs = basis_and_svs_to_expected_svs_at_epochs(basis_mats, svs, tau, epochs)
    return np.tensordot(trajs.T, basis_mats, axes=1)
    
    
def get_expected_iomats_at_epochs(item_mat, attr_mat, tau, epochs):
    """Use expected trajectories of each mode to compute expected input/output correlation matrices at specific epochs"""
    basis_mats, svs = get_io_corr_basis_mats_and_svs(item_mat, attr_mat)
    return basis_and_svs_to_expected_iomats_at_epochs(basis_mats, svs, tau, epochs)


def basis_and_svs_to_epected_attr_covar_at_epochs(basis_mats, svs, tau, epochs):
    expected_iomats = basis_and_svs_to_expected_iomats_at_epochs(basis_mats, svs, tau, epochs)
    centered_iomats = expected_iomats - np.mean(expected_iomats, axis=1, keepdims=True)
    return np.transpose(centered_iomats, (0, 2, 1)) @ centered_iomats


def get_expected_attr_covar_at_epochs(item_mat, attr_mat, tau, epochs):
    basis_mats, svs = get_io_corr_basis_mats_and_svs(item_mat, attr_mat)
    return basis_and_svs_to_epected_attr_covar_at_epochs(basis_mats, svs, tau, epochs)


def combine_single_sv_mode_group(u_group, vh_group):
    """Helper for below"""
    n_modes, n_items = vh_group.shape
    n_domains = n_modes
    items_per_domain = n_items // n_domains
    
    # Find which domain each item loading vec has the maximum variance in
    # reshape vh_group (item loadings) to be modes x domains x items
    vh_per_domain = np.reshape(vh_group, (n_modes, n_domains, items_per_domain))
    var_per_domain = np.var(vh_per_domain, axis=2)
    top_domains = np.argmax(var_per_domain, axis=1)
    
    # Now go over domains and decide whether to flip each mode (after the first)
    first_mode_vec = vh_per_domain[0, top_domains[0]]
    multipliers = np.ones(n_domains)
    for i, vh_mat in zip(range(1, n_modes), vh_per_domain[1:]):
        other_mode_vec = vh_per_domain[i, top_domains[i]]
        corr = np.dot(first_mode_vec, other_mode_vec)
        if corr < 0:
            multipliers[i] = -1
            
    # Finally flip the ones we want to flip and sum
    u_group_flipped = u_group * multipliers[np.newaxis, :]
    vh_group_flipped = vh_group * multipliers[:, np.newaxis]
    return np.sum(u_group_flipped, axis=1), np.sum(vh_group_flipped, axis=0)


def combine_sv_mode_groups_aligning_items(u, vh, n_domains=2):
    """
    Hackily add together groups of attr and item loadings, signflipping such that
    the item loadings in the domains they most load onto are maximally "aligned." 
    """
    u_groups = np.split(u, u.shape[1] // n_domains, axis=1)
    vh_groups = np.split(vh, vh.shape[0] // n_domains, axis=0)
    
    combined_u = np.empty((u.shape[0], len(u_groups)))
    combined_vh = np.empty((len(vh_groups), vh.shape[1]))
    for k, (u_group, vh_group) in enumerate(zip(u_groups, vh_groups)):
        combined_u[:, k], combined_vh[k, :] = combine_single_sv_mode_group(u_group, vh_group)
        
    return combined_u, combined_vh

