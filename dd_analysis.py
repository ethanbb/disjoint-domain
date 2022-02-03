"""Useful functions for analyzing results of disjoint-domain net runs"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # noqa
from scipy.cluster import hierarchy
from scipy.spatial import distance
from scipy.linalg import block_diag, svd, norm
from sklearn.manifold import MDS
from sklearn.decomposition import NMF
from statsmodels.regression.linear_model import OLS
from patsy import dmatrices

import ddnet
import disjoint_domain as dd
import util
import net_analysis
import problem_analysis as pa

report_titles = {
    'loss': 'Mean loss',
    'accuracy': 'Mean accuracy',
    'weighted_acc': 'Mean accuracy (weighted)',
    'weighted_acc_loose': 'Mean weighted sign accuracy',
    'weighted_acc_loose_indomain': 'Mean weighted sign accuracy (within domain)',
    'etg_item': 'Epochs to learn new items',
    'etg_context': 'Epochs to learn new contexts',
    'etg_domain': 'Epochs to learn new domain',
    'test_accuracy': 'Accuracy on novel item/context pairs',
    'test_weighted_acc': 'Mean generalization accuracy (weighted)',
    'test_weighted_acc_loose': 'Generalization sign accuracy (weighted)',
    'test_weighted_acc_loose_indomain': 'Generalization sign accuracy (weighted, within domain)'
}


# make some util functions available under this namespace since I have a ton of code using it
auto_subplots = util.auto_subplots
make_plot_grid = util.make_plot_grid
outside_legend = util.outside_legend
add_colorbar = util.add_colorbar


def get_result_means(res_path, **result_mean_opts):
    
    res = net_analysis.get_result_means(res_path, **result_mean_opts, extra_keys=['ys'])
    
    # expand any cluster info that was passed as a nullary function
    for possible_fn in ddnet.callable_net_params:
        if possible_fn in res['net_params'] and callable(res['net_params'][possible_fn]):
            res['net_params'][possible_fn] = res['net_params'][possible_fn]()
            
    # make aliases for "item_" snapshots if contexts were not used
    if not res['net_params']['use_ctx']:
        for field in {'snaps', 'repr_corr', 'repr_dists'} & set(res.keys()):
            snap_aliases = {'item_' + snap_type: val for snap_type, val in res[field].items() if 'item' not in snap_type}
            res[field].update(snap_aliases)
            
    # add full item and context repr dists
    item_snaps = [res['snaps'][stype] for stype in ['item', 'item_hidden'] if stype in res['snaps']]
    if len(item_snaps) > 0:
        item_full_snaps = np.concatenate(item_snaps, axis=3)
        res['repr_corr']['item_full'] = net_analysis.calc_mean_repr_corr(item_full_snaps, **result_mean_opts)
        if 'compute_full_rdms' in result_mean_opts and result_mean_opts['compute_full_rdms']:
            res['repr_dists']['item_full'] = net_analysis.calc_mean_repr_dists(item_full_snaps, **result_mean_opts)
    
    ctx_snaps = [res['snaps'][stype] for stype in ['context', 'context_hidden'] if stype in res['snaps']]
    if len(ctx_snaps) > 0:
        ctx_full_snaps = np.concatenate(ctx_snaps, axis=3)
        res['repr_corr']['context_full'] = net_analysis.calc_mean_repr_corr(ctx_full_snaps, **result_mean_opts)
        if 'compute_full_rdms' in result_mean_opts and result_mean_opts['compute_full_rdms']:
            res['repr_dists']['context_full'] = net_analysis.calc_mean_repr_dists(ctx_full_snaps, **result_mean_opts)
    return res    


def plot_report(ax, res, report_type, **kwargs):
    if 'title' not in kwargs:
        kwargs['title'] = report_titles[report_type]
    net_analysis.plot_report(ax, res, report_type, **kwargs)


def plot_individual_reports(ax, res, report_type, **kwargs):
    if 'title' not in kwargs:
        kwargs['title'] = report_titles[report_type]
    net_analysis.plot_individual_reports(ax, res, report_type, **kwargs)


def _get_names_for_snapshots(snap_type, **net_params):
    """Helper to retrieve input names depending on the type of snapshot"""
    if 'item' in snap_type or 'attr' in snap_type:
        names = dd.get_items(train_only=True, **net_params)[1]
    elif 'context' in snap_type:
        names = dd.get_contexts(train_only=True, **net_params)[1]
    else:
        raise ValueError('Unrecognized snapshot type')
    
    return names
    

def plot_matrix_with_input_labels(ax, mat, input_type, res=None, **plot_matrix_args):
    """Helper to plot matrix with labels corresponding to items or contexts"""
    if res is None:
        net_params = {}
    else:
        net_params = res['net_params']
    labels = _get_names_for_snapshots(input_type, **net_params)
    return util.plot_matrix_with_labels(ax, mat, labels, **plot_matrix_args)


def plot_repr_corr(ax, res, snap_type, snap_ind, **kwargs):
    """
    Plot an RSA matrix for the representation of items or contexts at a particular epoch
    If 'rsa_mat' is provided, overrides the matrix to plot.
    """
    input_names = _get_names_for_snapshots(snap_type, **res['net_params'])
    return net_analysis.plot_repr_corr(ax, res, snap_type, snap_ind, input_names, **kwargs)


def plot_rdm(ax, res, snap_type, snap_ind, **kwargs):
    input_names = _get_names_for_snapshots(snap_type, **res['net_params'])
    return net_analysis.plot_rdm(ax, res, snap_type, snap_ind, input_names, **kwargs)


def get_item_loadings_svs_and_scores(res, snap_ind, run_ind, n_modes=None, layer='item_attr', center=False):
    """
    Get SVD loading onto items of the empirical I/O matrix ('attr' snapshot) from a particular run.
    weighted controls whether to weight each mode by its singular value.
    n_modes allows returning only the first n modes (if not None)
    Output is an n_items x n_modes matrix.
    """
    mat = res['snaps'][layer][run_ind, snap_ind, ...]
    if center:
        mat = mat - np.mean(mat, axis=0)

    u, s, vd = svd(mat, full_matrices=False)
    if n_modes is not None:
        u = u[:, :n_modes]
        s = s[:n_modes]
        vd = vd[:n_modes, :]
    return u, s, vd
    

def plot_item_svd_loadings(ax, res, snap_ind, run_ind, weighted=True, n_modes=None, layer='item_attr', center=False,
                           title_addon=None, colorbar=True, tick_fontsize='x-small'):
    """
    Plot the SVD loadings onto items of the empirical I/O matrix from a particular run.
    """
    u, s = get_item_loadings_svs_and_scores(res, snap_ind, run_ind, n_modes=n_modes, layer=layer, center=center)[:2]
    if weighted:
        u = u @ np.diag(s)
    image = plot_matrix_with_input_labels(ax, u, 'item', res, colorbar=colorbar,
                                          label_cols=False, tick_fontsize=tick_fontsize)
#     image = ax.imshow(u)
#     add_colorbar(image)
    ax.set_xlabel('SVD component #')
    
    title = f'Epoch {res["snap_epochs"][snap_ind]}'
    if title_addon is not None:
        title += f'\n({title_addon})'
    ax.set_title(title)

    return image


def plot_item_svd_scores(ax, res, snap_ind, run_ind, weighted=False, n_modes=None, layer='item_attr', center=False,
                         title_addon=None, colorbar=True, tick_fontsize='x-small'):
    """
    Plot the SVD loadings onto attributes of the empirical I/O matrix from a particular run.
    """
    s, vd = get_item_loadings_svs_and_scores(res, snap_ind, run_ind, n_modes=n_modes, layer=layer, center=center)[1:]
    if weighted:
        vd = np.diag(s) @ vd

    image = util.imshow_centered_bipolar(ax, vd, aspect='auto')
    
    if colorbar:
        add_colorbar(image)

    for ticklabel in ax.get_xticklabels() + ax.get_yticklabels():
        ticklabel.set_fontsize(tick_fontsize)

    ax.set_ylabel('SVD component #')
    ax.set_xlabel('Attribute')
    
    title = f'Epoch {res["snap_epochs"][snap_ind]}'
    if title_addon is not None:
        title += f'\n({title_addon})'
    ax.set_title(title)

    return image


def get_item_nmf_loadings_and_scores(res, snap_ind, run_ind, n_modes=None, layer='item_attr'):
    """
    Do an NMF decomposition of the empirical I/O matrix
    """    
    model = NMF(n_components=n_modes, solver='mu', max_iter=500)
    mat = res['snaps'][layer][run_ind, snap_ind, ...]
    # a little hacky but I think it's reasonable to get all elements positive...
    mat_shifted = mat - np.min(mat)
    u = model.fit_transform(mat_shifted)
    vd = model.components_

    # normalize so that *attribute* loadings (scores) have unit norm
    norm_vec = norm(vd, axis=1)
    inv_norm_vec = norm_vec
    inv_norm_vec[np.flatnonzero(norm_vec)] **= -1
    vd = vd * inv_norm_vec[:, np.newaxis]
    u = u * norm_vec[np.newaxis, :]
    
    return u, vd


def get_domain_mixing_score(res, snap_ind, run_ind, layer='item_attr'):
    """
    A method of quantifying to what extent SVD modes of the empirical I/O matrix traverse
    multiple domains.
    - For each item loading, the L2 norm over each domain's items is computed.
    - These individual domain norms are normalized to sum to 1
    - The entropy of this "distribution" over domains is calculated
    - The result is the sum of each mode's entropy, weighted by its singular value.
    
    Update 10/1: this doesn't take into account mixing that can occur due to repeated singular values,
    that doesn't correspond to shared information when the modes are summed together.
    Use rank-based mixing score instead (rank compression)
    """
    item_loadings, svs = get_item_loadings_svs_and_scores(res, snap_ind, run_ind, layer=layer)[:2]
    weights = svs / sum(svs)
    
    # get the domains
    n_items = item_loadings.shape[0]
    dom_len = dd.ITEMS_PER_DOMAIN
    assert n_items % dom_len == 0, 'Huh? Odd number of items'
    n_domains = n_items // dom_len
    
    mode_entropies = np.zeros(len(svs))
    for i, loading in enumerate(item_loadings.T):
        domain_norms = np.array([
            norm(loading[dom_len*d:dom_len*(d+1)]) for d in range(n_domains)
        ])
        domain_norms /= sum(domain_norms)
        mode_entropies[i] = -sum(domain_norms * np.log2(domain_norms))
    
    return np.dot(weights, mode_entropies)


def plot_domain_mixing_scores(ax, res, epoch_range=None, layer='item_attr',
                              with_ci=True, label=None, **plot_params):
    """Make a plot of mean domain mixing scores over epochs"""
    
    iomats = res['snaps'][layer] # runs x snaps x items x attrs
    if epoch_range is None:
        epoch_range = range(iomats.shape[1])
        
    mixing_scores = np.array([
        [
            get_domain_mixing_score(res, snap_ind, run_ind, layer=layer)
            for snap_ind in epoch_range
        ]
        for run_ind in range(iomats.shape[0])
    ])

    score_mean, score_ci = util.get_mean_and_ci(mixing_scores)
    xaxis = [res['snap_epochs'][e] for e in epoch_range]
    ax.plot(xaxis, score_mean, label=label, **plot_params)
    if with_ci:
        ax.fill_between(xaxis, *score_ci, alpha=0.3)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weighted domain entropy (bits)')
    ax.set_title(f'Domain mixing in I/O SVD loadings onto items ({layer} layer)')
    

def get_io_corr_rank(res, snap_ind, run_ind, layer='attr', center=True):
    """Find the rank of an I/O correlation matrix up to 99% of the power (according to squared singular values)"""
    svs = get_item_loadings_svs_and_scores(res, snap_ind, run_ind, layer=layer, center=center)[1]
    cum_sv_power = np.cumsum(svs**2 / sum(svs**2))
    return np.sum(cum_sv_power <= 0.99) + 1


def plot_io_corr_ranks(ax, res, epoch_range=None, layer='item_attr', center=True,
                       with_ci=True, label=None, **plot_params):
    iomats = res['snaps'][layer] # runs x snaps x items x attrs
    if epoch_range is None:
        epoch_range = range(iomats.shape[1])
        
    corr_ranks = np.array([
        [
            get_io_corr_rank(res, snap_ind, run_ind, layer=layer, center=center)
            for snap_ind in epoch_range
        ]
        for run_ind in range(iomats.shape[0])
    ])

    rank_mean, rank_ci = util.get_mean_and_ci(corr_ranks)
    xaxis = [res['snap_epochs'][e] for e in epoch_range]
    ax.plot(xaxis, rank_mean, label=label, **plot_params)
    if with_ci:
        ax.fill_between(xaxis, *rank_ci, alpha=0.3)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Rank to reach 99% power')
    ax.set_title(f'Effective rank of I/O correlation matrix ({layer} layer)')
    
    
def get_full_vs_domain_rank_ratio(res, snap_ind, run_ind, layer, center=True):
    """
    Evaluate shared information across domains by looking at the SVD rank (up to 99% power)
    of the full I/O matrix vs. submatrices corresponding to individual domains.
    This can be done for the full end-to-end I/O matrix (in which case the ratio is expected to converge
    to the value it takes for the ground-truth I/O matrix) or for intermediate steps.
    This is selected with the layer argument, which should be 'repr', 'hidden', or 'attr'.
    """
    try:
        mat = res['snaps'][layer][run_ind, snap_ind, ...]
        if center:
            mat = mat - np.mean(mat, axis=0)
    except KeyError:
        raise ValueError('Given results do not have I/O matrix information for the requested layer.')
    
    full_mat_rank = get_io_corr_rank(res, snap_ind, run_ind, layer=layer, center=center)
    
    n_domains = res['net_params']['n_domains']
    sub_mats = np.split(mat, n_domains, axis=0)  # splitting by items
    sub_svs = [svd(sub_mat, full_matrices=False)[1] for sub_mat in sub_mats]
    sub_cum_powers = [np.cumsum(svs**2 / sum(svs**2)) for svs in sub_svs]
    sub_ranks = np.array([np.sum(cum_powers <= 0.99) + 1 for cum_powers in sub_cum_powers])
    
    return full_mat_rank / np.mean(sub_ranks)


def get_rank_domain_mixing_score(res, snap_ind, run_ind, layer):
    """
    Helper to remap mean domain rank ratio to a [0, 1] score of domain mixing
    (If individual domain ranks are n_domains times less than rank of full matrix, there's no mixing;
    if the two are equal, there's full mixing.)
    """
    n_domains = res['net_params']['n_domains']
    if n_domains < 2:
        raise ValueError('Cannot score domain mixing with just one domain')
    
    rank_ratio = get_full_vs_domain_rank_ratio(res, snap_ind, run_ind, layer, center=True)
    svs = get_item_loadings_svs_and_scores(res, snap_ind, run_ind, layer=layer, center=True)[1]
    total_var = sum((svs**2))
    rank_score = n_domains - rank_ratio
    rank_score /= (n_domains - 1)
    rank_score *= total_var
    return rank_score


def plot_rank_domain_mixing_scores(ax, res, epoch_range=None, layer='item_attr',
                                   with_ci=True, label=None, **plot_params):
    
    iomats = res['snaps'][layer] # runs x snaps x items x attrs
    if epoch_range is None:
        epoch_range = range(iomats.shape[1])
        
    rank_mixing_scores = np.array([
        [
            get_rank_domain_mixing_score(res, snap_ind, run_ind, layer=layer)
            for snap_ind in epoch_range
        ]
        for run_ind in range(iomats.shape[0])
    ])

    rank_mean, rank_ci = util.get_mean_and_ci(rank_mixing_scores)
    xaxis = [res['snap_epochs'][e] for e in epoch_range]
    ax.plot(xaxis, rank_mean, label=label, **plot_params)
    if with_ci:
        ax.fill_between(xaxis, *rank_ci, alpha=0.3)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score of full vs. domain rank ratio')
    ax.set_title(f'Domain mixing based on SVD rank ({layer} layer)')


def plot_repr_dendrogram(ax, res, snap_type, snap_ind, title_addon=None):
    """Similar to plot_rsa, but show dendrogram rather than RSA matrix"""
    input_names = _get_names_for_snapshots(snap_type, **res['net_params'])

    dists_compressed = distance.squareform(res['repr_dists'][snap_type]['snaps'][snap_ind])
    z = hierarchy.linkage(dists_compressed, optimal_ordering=True)
    hierarchy.dendrogram(z, labels=input_names, count_sort=True, ax=ax)
    title = f'Epoch {res["snap_epochs"][snap_ind]}'
    if title_addon is not None:
        title += f' ({title_addon})'
    ax.set_title(title)


def plot_repr_embedding(ax, res, snap_type, snap_ind, colors=None):
    """Similar to plot_rsa, but plot 2D embeddings of items or contexts using MDS"""
    input_names = _get_names_for_snapshots(snap_type, **res['net_params'])
    embedding = MDS(n_components=2, dissimilarity='precomputed')
    reprs_embedded = embedding.fit_transform(res['repr_dists'][snap_type]['snaps'][snap_ind])
    
    ax.scatter(*reprs_embedded.T, c=colors)
    for pos, name in zip(reprs_embedded, input_names):
        ax.annotate(name, pos)
        
    
def plot_repr_trajectories(res, snap_type, dims=2, title_label=''):
    """
    Plot trajectories of each item or context representation over training
    using MDS. Can plot in 3D by settings dims to 3.
    Returns figure and axes.
    """
    embedding = MDS(n_components=dims, dissimilarity='precomputed')
    reprs_embedded = embedding.fit_transform(res['repr_dists'][snap_type]['all'])

    # reshape and permute to aid plotting
    n_snaps = len(res['snap_epochs'])
    n_domains = res['net_params']['n_train_domains']
    reprs_embedded = reprs_embedded.reshape((n_snaps, n_domains, -1, dims))
    reprs_embedded = reprs_embedded.transpose((1, 2, 3, 0))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection=('3d' if dims == 3 else None))
    
    input_names = _get_names_for_snapshots(snap_type, **res['net_params'])
    
    if 'item' in snap_type:
        if 'item_clusters' in res['net_params']:
            input_groups = dd.item_group(clusters=res['net_params']['item_clusters'])
        elif 'cluster_info' in res['net_params']:
            input_groups = dd.item_group(clusters=res['net_params']['cluster_info'])
        else:
            input_groups = dd.item_group()
    elif 'context' in snap_type:
        # No "groups," but use symbols for individual contexts (per domain) instead.
        input_groups = np.arange(4)
    else:
        raise ValueError('Unrecognized snapshot type')
        
    input_names = np.array(input_names).reshape((n_domains, -1))

    colors = dd.get_domain_colors()
    markers = ['o', 's', '*', '^']
    for dom_reprs, dom_labels, color in zip(reprs_embedded, input_names, colors):
        for reprs, label, group in zip(dom_reprs, dom_labels, input_groups):
            linestyle = markers[group] + '-'
            ax.plot(*reprs, linestyle, label=label, markersize=4, color=color, linewidth=0.5)

    # add start and end markers on top of everything else
    for dom_reprs, dom_labels, color in zip(reprs_embedded, input_names, colors):
        for reprs, label, group in zip(dom_reprs, dom_labels, input_groups):
            marker = markers[group]

            ax.plot(*reprs[:, 0], 'g' + marker, markersize=8)
            ax.plot(*reprs[:, 0], marker, markersize=5, color=color)
            ax.plot(*reprs[:, -1], 'r' + marker, markersize=8)
            ax.plot(*reprs[:, -1], marker, markersize=5, color=color)

    ax.set_title(f'{title_label} {snap_type} representations over training\n' +
                 'color = domain, marker = type within domain')

    return fig, ax


def plot_hl_input_pattern_correlations(ax, res, run_num, snap_index, title_label=''):
    """
    Plots a matrix of how strongly the weights of each hidden layer neuron
    onto the attribute layer correlate with the correct output pattern for
    each item/context pair.
    """
    
    with np.load(res['path'], allow_pickle=True) as resfile:
        if 'parameters' not in res or 'ys' not in resfile:
            raise RuntimeError("Selected results file doesn't have needed data.")
        
        ha_weights = resfile['parameters'].item()['hidden_to_attr.weight'][run_num, snap_index, ...]
        ys = res['ys'][run_num]
        
    ys_norm = ys - np.mean(ys, axis=1, keepdims=True)
    ys_norm /= np.std(ys_norm, axis=1, keepdims=True)
    
    weights_norm = ha_weights - np.mean(ha_weights, axis=0, keepdims=True)
    weights_norm /= np.std(weights_norm, axis=0, keepdims=True)
    
    corrs = ys_norm @ weights_norm / ys_norm.shape[1]
    
    epoch = res['snap_epochs'][snap_index]
    
    # get labels for item/context combinations
    _, item_names = dd.get_items(**res['net_params'])
    _, ctx_names = dd.get_contexts(**res['net_params'])
    
    n_domains = res['net_params']['n_domains']
    items_per_domain = np.split(np.array(item_names), n_domains)
    ctx_per_domain = np.split(np.array(ctx_names), n_domains)
    
    input_names = []
    for items, ctx in zip(items_per_domain, ctx_per_domain):
        input_names.append([f'{iname}/{cname}' for cname in ctx for iname in items])
    
    input_names = np.concatenate(input_names)
            
    image = util.imshow_centered_bipolar(ax, corrs, interpolation='nearest')
    ax.set_yticks(range(len(input_names)))
    ax.set_yticklabels(input_names)
    ax.set_xticks([])
    ax.set_xlabel('Hidden layer neurons')
    ax.set_title(f'{title_label} correlation of hidden-to-attribute weights\n' + 
                 f'with input attributes (run {run_num+1}, epoch {epoch})')
    add_colorbar(image)
    
    return image
    

def make_ortho_item_rsa_models(n_domains, ctx_per_domain=4, attrs_per_context=50, clusters='4-2-2', **_extra):
    """
    Makes a set of model RDMs that have unit norm and are pairwise orthogonal, and
    are hopefully useful for interpreting item representations in the DDNet.
    
    All models have zeros along the diagonal.
    
        - 'uniformity'                - Constant term for all distinct item pairs. All other models sum to 0.
        - 'attribute_similarity'      - Centered distance between item attribute vectors within each domain 
                                        (Fig. R3)  
        - 'same_vs_different_domain'  - Contrasts mean distance between domains vs. within domain
        - 'cross_domain_group_match'  - Contrasts distance between same-group vs. different-group
                                        cross-domain pairs (regrdless of any within-domain distances).
                                        Considers just "circle" and "non-circle" groups for simplicity.

    Returns the set of RDMs as a dict.
    """    

    # spread
    n_items = n_domains * dd.ITEMS_PER_DOMAIN
    #models = {'spread': util.norm_rdm(np.ones((n_items, n_items)))}
    models = {'uniformity': np.full((n_items, n_items), 1 / n_items**2)}
    
    # in-domain attribute distance
    item_attr_rdm = dd.get_item_attribute_rdm(ctx_per_domain, attrs_per_context)
    item_attr_model_1domain = util.center_and_norm_rdm(item_attr_rdm) / np.sqrt(n_domains)
    models['attribute_similarity'] = block_diag(*[item_attr_model_1domain for _ in range(n_domains)])
    
    # cross vs. within-domain
    is_domain_eq = block_diag(*[np.ones((dd.ITEMS_PER_DOMAIN, dd.ITEMS_PER_DOMAIN), dtype=bool)
                                for _ in range(n_domains)])
    nz_where_domain_ne = 1 - is_domain_eq
    models['same_vs_different_domain'] = util.center_and_norm_rdm(nz_where_domain_ne)
    
    # cross-domain group
    is_circle = np.equal(dd.item_group(clusters=clusters), 0)
    is_diff_group = is_circle[:, np.newaxis] != is_circle[np.newaxis, :]
    diff_group_centered = is_diff_group - np.mean(is_diff_group)
    diff_group_tiled = np.tile(diff_group_centered, (n_domains, n_domains))
    diff_group_tiled[is_domain_eq] = 0  # make block diagonal zero
    models['cross_domain_group_match'] = util.norm_rdm(diff_group_tiled)
    
#     # combination of attribute distance and cross-domain group
#     tiled_item_attr = np.tile(item_attr_model_1domain, (n_domains, n_domains))
#     models['attr_with_cross_domain'] = util.center_and_norm_rdm(models['attribute_similarity'] + (0.5 * nz_where_domain_ne) * tiled_item_attr)
     
    return models
    

def make_ortho_context_rsa_models(n_domains, ctx_per_domain=4, **_extra):
    """
    Makes a set of model RDMs for context representations. Similar to make_ortho_item_rsa_models,
    but with only the 'uniformity' and 'cross_vs_in_domain' types.
    """
    # models = {'spread': util.norm_rdm(1 - np.eye(n_domains * ctx_per_domain))}
    n_contexts = n_domains * ctx_per_domain
    models = {'uniformity': np.full((n_contexts, n_contexts), 1 / n_contexts**2)}
    
    # cross vs. within-domain
    is_domain_eq = block_diag(*[np.ones((ctx_per_domain, ctx_per_domain), dtype=bool)
                                for _ in range(n_domains)])
    nz_where_domain_ne = 1 - is_domain_eq
    models['same_vs_different_domain'] = util.center_and_norm_rdm(nz_where_domain_ne)
    
    return models


def test_model_validity(n_domains=4):
    """Make sure all item and context model RDMs are pairwise orthogonal with unit norm"""
    item_models = make_ortho_item_rsa_models(n_domains)
    item_model_mat = np.stack([model.ravel() for model in item_models.values()], axis=1)
    ctx_models = make_ortho_context_rsa_models(n_domains)
    ctx_model_mat = np.stack([model.ravel() for model in ctx_models.values()], axis=1)
    
    if not np.allclose(item_model_mat.T @ item_model_mat, np.eye(len(item_models))):
        print('Warning: item model RDMs are not orthogonal.')
    else:
        print('Item model RDMs are orthogonal.')
    
    if not np.allclose(np.linalg.norm(item_model_mat, axis=0), np.ones(len(item_models))):
        print('Warning: item model RDMs do not all have unit norm.')
    else:
        print('Item model RDMs all have unit norm.')
    
    if not np.allclose(ctx_model_mat.T @ ctx_model_mat, np.eye(len(ctx_models))):
        print('Warning: context model RDMs are not orthogonal.')
    else:
        print('Context model RDMs are orthogonal.')
    
    if not np.allclose(np.linalg.norm(ctx_model_mat, axis=0), np.ones(len(ctx_models))):
        print('Warning: context model RDMs do not all have unit norm.')
    else:
        print('Context model RMDs all have unit norm.')


def get_rdm_projections(res, snap_type='item', models=None):
    """
    Wraps get_rdm_projections in net_analysis.py, using the orthogonal RSA models specified above by default.
    """
    if models is None:
        if 'item' in snap_type:
            models = make_ortho_item_rsa_models(**res['net_params'])
        elif 'context' in snap_type:
            models = make_ortho_context_rsa_models(**res['net_params'])
        else:
            raise ValueError(f'Snapshot type {snap_type} not recognized')
            
    return net_analysis.get_rdm_projections(res, snap_type, models)


def plot_rdm_projections(res, snap_type, axs, label=None, **plot_params):
    """
    Plot time series of item or context RDM projections onto given axes, with 95% CI.
    model_types should be a list of the same size as axs.
    """    
    # get all the projections to start
    projections = get_rdm_projections(res, snap_type)
    model_types = projections.keys()
    
    axs = axs.ravel()
    if len(axs) != len(model_types):
        raise ValueError(f'Wrong number of axes given (expected {len(model_types)})')
    
    layer = 'hidden layer' if 'hidden' in snap_type else 'all layers' if 'full' in snap_type else 'repr layer'
    input_type = 'item' if 'item' in snap_type else 'context'
    
    for ax, mtype in zip(axs, model_types):
        try:
            mean, (lower, upper) = util.get_mean_and_ci(projections[mtype])
        except KeyError:
            raise ValueError(f'Model type {mtype} not defined for {snap_type} snapshots.')
        
        ax.plot(res['snap_epochs'], mean, label=label, **plot_params)
        ax.fill_between(res['snap_epochs'], lower, upper, **{'alpha': 0.3, **plot_params})
        
        ax.set_title(f'Projection of {input_type} RDMs in {layer} onto {mtype} model')


def make_dict_for_regression(res_array):
    """
    Make a dict of regressors and response variables to use to test effects of things
    like RDM projections on things like model generalization accuracy.
    Can be used as the 'data' parameter to patsy.dmatrices.
    Uses all results in res_array concatenated together in time.
    **Assumes snapshot and report epochs are the same, which is true for pretty much all my runs**
    """
    run_dicts = []
    runs_with_column = {} # to figure out which columns are in each run
    
    for res in res_array:
        # start with rdm projections (with full hidden layer state)
        run_dict = {('item_' + key): proj.ravel() for key, proj in get_rdm_projections(res, snap_type='item_full').items()}
        run_dict.update({('ctx_' + key): proj.ravel() for key, proj in get_rdm_projections(res, snap_type='context_full').items()})
        
        # add reports
        with np.load(res['path'], allow_pickle=True) as resfile:
            report_dict = resfile['reports'].item()            
        run_dict.update({key: report.ravel() for key, report in report_dict.items() if 'etg' not in key})
        
        for key in run_dict:
            if key in runs_with_column:
                runs_with_column[key] += 1
            else:
                runs_with_column[key] = 1
                
        run_dicts.append(run_dict)

    # concatenate all runs across time
    shared_keys = [key for key, count in runs_with_column.items() if count == len(res_array)]
    return {key: np.concatenate([run_dict[key] for run_dict in run_dicts]) for key in shared_keys}
    
        
def fit_linear_model(formula, data_dict):
    """
    Creates a statsmodel OLS model for the R-style (patsy) formula given the
    variables in data_dict (created with make_dict_for_regression).
    Returns the statsmodels results object.
    """
    y, x = dmatrices(formula, data=data_dict, return_type='dataframe')
    model = OLS(y, x)
    return model.fit()

def get_attr_freq_dist_mats(res, train_items=slice(None), normalize=False):
    """
    Returns a matrix for each individual run indicating how much the mean # of 
    attributes shared with other items differs for each item pair
    """
    item_mat = dd.make_io_mats(**res['net_params'])[0][:, train_items]
    attr_mats = res['ys']
    attr_freq_dist_mats = [pa.get_attr_freq_dist_mat(item_mat, attr_mat) for attr_mat in attr_mats]
    if normalize:
        attr_freq_dist_mats = [util.center_and_norm_rdm(mat) for mat in attr_freq_dist_mats]
    return np.stack(attr_freq_dist_mats)


def get_svd_dist_mats(res, train_items=slice(None), modes_to_use=slice(None), normalize=False, n_domains=None):
    """
    Returns a matrix for each individual run indicating the difference between each pair of items
    as a cityblock distance of their SVD loadings. This is supposed to capture info abount hierarchical position.
    """
    ys = res['ys']
    item_mat = dd.make_io_mats(**res['net_params'])[0][:, train_items]
    if n_domains is None:
        n_domains = item_mat.shape[1] // dd.ITEMS_PER_DOMAIN
    svd_dist_mats = [pa.get_contextfree_item_svd_dist(item_mat, y, n_domains, modes_to_use) for y in ys]
    if normalize:
        return np.stack([util.center_and_norm_rdm(mat) for mat in svd_dist_mats])
    return np.stack(svd_dist_mats)


def plot_attr_freq_dist_correlation(ax, res, snap_type='item_full', train_items=slice(None),
                                    label=None, **plot_params):
    """
    Plot the correlation between item snapshot distances at each epoch and the absolute
    differences in mean # of attributes shared with other items. This seems to be an
    important factor for the item RDMs early in training.
    """
    snap_dists = res['repr_dists'][snap_type]['snaps_each']
    corrs = np.empty(snap_dists.shape[:2])
        
    attr_freq_dists = get_attr_freq_dist_mats(res, train_items=train_items)
    
    # iterate over runs
    for run_dists, attr_freq_dist, corr_vec in zip(snap_dists, attr_freq_dists, corrs):        
        # correlate condensed distances to avoid diagonal (could be varying offset on off-diagonal entries)
        attr_freq_dist_cd = distance.squareform(attr_freq_dist)        
        item_repr_dists_cd = [distance.squareform(dist_mat[train_items, train_items]) for dist_mat in run_dists]
        corr_vec[:] = [np.corrcoef(attr_freq_dist_cd, idist)[0, 1] for idist in item_repr_dists_cd]
    
    # now plot, with confidence interval
    mean, ci = util.get_mean_and_ci(corrs)
    ax.plot(res['snap_epochs'], mean, label=label, **plot_params)
    ax.fill_between(res['snap_epochs'], *ci, alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Correlation (r)')
    ax.set_title('Correlation of item representation distance with\ndifference in attribute frequency')
