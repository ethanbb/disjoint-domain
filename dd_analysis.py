"""Useful functions for analyzing results of disjoint-domain net runs"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D # noqa
from scipy.cluster import hierarchy
from scipy.spatial import distance
from scipy import stats
from sklearn.manifold import MDS
from itertools import product

import disjoint_domain as dd

report_titles = {
    'loss': 'Mean loss',
    'accuracy': 'Mean accuracy',
    'weighted_acc': 'Mean accuracy (weighted)',
    'etg_item': 'Epochs to learn new items',
    'etg_context': 'Epochs to learn new contexts',
    'test_accuracy': 'Accuracy on novel item/context pairs',
    'test_weighted_acc': 'Mean generalization accuracy (weighted)'
}


def get_mean_repr_dists(repr_snaps):
    """
    Make distance matries (RSAs) of given representations of items or contexts,
    averaged over training runs.

    repr_snaps is a n_runs x n_snap_epochs x n_inputs x n_rep array, where
    n_inputs is n_items or n_contexts and n_rep is the size of the representation.

    Outputs a dict with keys:
    'all': gives the mean distances between all representations over
    all epochs. This is useful for making an MDS map of the training process.

    'snaps': gives the mean distsances between items or contexts at each
    recorded epoch. This is the n_inputs x n_inputs block diagonal of dists_all, stacked.
    """

    n_runs, n_snap_epochs, n_inputs, n_rep = repr_snaps.shape
    snaps_flat = np.reshape(repr_snaps, (n_runs, n_snap_epochs * n_inputs, n_rep))
    dists_all = np.stack([distance.pdist(run_snaps) for run_snaps in snaps_flat])
    mean_dists_all = distance.squareform(np.nanmean(dists_all, axis=0))

    mean_dists_snaps = np.empty((n_snap_epochs, n_inputs, n_inputs))
    for k_epoch in range(n_snap_epochs):
        this_slice = slice(k_epoch * n_inputs, (k_epoch+1) * n_inputs)
        mean_dists_snaps[k_epoch] = mean_dists_all[this_slice, this_slice]

    return {'all': mean_dists_all, 'snaps': mean_dists_snaps}


def get_result_means(res_path):
    """Get dict of data (meaned over runs) from saved file"""
    with np.load(res_path, allow_pickle=True) as resfile:
        snaps = resfile['snapshots'].item()
        reports = resfile['reports'].item()
        net_params = resfile['net_params'].item()
        train_params = resfile['train_params'].item()

    mean_repr_dists = {
        snap_type: get_mean_repr_dists(repr_snaps)
        for snap_type, repr_snaps in snaps.items()
    }

    mean_reports = {
        report_type: np.mean(report, axis=0)
        for report_type, report in reports.items()
    }

    snap_epochs = dd.calc_snap_epochs(
        train_params['snap_freq'], train_params['snap_freq_scale'],
        train_params['num_epochs'])

    report_epochs = np.arange(0, train_params['num_epochs'], train_params['report_freq'])
    etg_epochs = report_epochs[::train_params['reports_per_test']]

    return {
        'path': res_path,
        'repr_dists': mean_repr_dists,
        'reports': mean_reports,
        'net_params': net_params,
        'train_params': train_params,
        'snap_epochs': snap_epochs,
        'report_epochs': report_epochs,
        'etg_epochs': etg_epochs
    }


def auto_subplots(n_rows, n_cols, ax_dims=(4, 4)):
    """Make subplots, automatically adjusting the figsize, and without squeezing"""
    figsize = (ax_dims[0] * n_cols, ax_dims[1] * n_rows)
    return plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)


def make_plot_grid(n, n_cols=3, ax_dims=(4, 4), ravel=True):
    """
    Create a figure with n axes arranged in a grid, and return the fig and flat array of axes
    ax_dims is the width, height of each axes in inches (or whatever units matplotlib uses)
    """
    n_rows = (n-1) // n_cols + 1
    fig, axs = auto_subplots(n_rows, n_cols, ax_dims=ax_dims)
    return fig, axs.ravel() if ravel else axs


def outside_legend(ax, **legend_params):
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), **legend_params)


def plot_report(ax, res, report_type, **plot_params):
    """Make a standard plot of mean loss, accuracy, etc. over training"""
    xaxis = res['etg_epochs'] if report_type[:3] == 'etg' else res['report_epochs']
    ax.plot(xaxis, res['reports'][report_type], **plot_params)
    ax.set_xlabel('Epoch')
    ax.set_title(report_titles[report_type])


def plot_individual_reports(ax, res, all_reports, report_type, **plot_params):
    """
    Take a closer look by looking at a statistic for each run in a set.
    all_reports should be the full reports dict of runs x epochs matrices,
    loaded from the results file.
    """
    xaxis = res['etg_epochs'] if report_type[:3] == 'etg' else res['report_epochs']

    for (i, report), (col, ls) in zip(enumerate(all_reports[report_type]),
                                      product(mcolors.TABLEAU_COLORS, ['-', '--', '-.', ':'])):
        ax.plot(xaxis, report, color=col, linestyle=ls, label=f'Run {i+1}', **plot_params)

    ax.set_xlabel('Epoch')
    ax.set_title(report_titles[report_type] + ' (each run)')
    outside_legend(ax, ncol=2, fontsize='x-small')

    
def _get_names_for_snapshots(snap_type, **net_params):
    """Helper to retrieve input names depending on the type of snapshot"""
    if 'item' in snap_type:
        names = dd.get_items(**net_params)[1]
    elif 'context' in snap_type:
        names = dd.get_contexts(**net_params)[1]
    else:
        raise ValueError('Unrecognized snapshot type')
    
    return names
    

def plot_rsa(ax, res, snap_type, snap_ind, title_addon=None, item_order='domain-outer', colorbar=True, rsa_mat=None):
    """
    Plot an RSA matrix for the representation of items or contexts at a particular epoch
    item_order: controls the order of items in the matrix - either a string or an
    array to use a custom permutation.
        'domain-outer' (default): items are grouped together by domain
        'domain-inner': first item of each domain, then second item, etc.
        'group-outer': sorted by group (circles/squares/stars), then by domain
    If 'rsa_mat' is provided, overrides the matrix to plot.
    """
    input_names = _get_names_for_snapshots(snap_type, **res['net_params'])
    n_inputs = len(input_names)

    if rsa_mat is None:
        rsa_mat = res['repr_dists'][snap_type]['snaps'][snap_ind]

    n_domains = res['net_params']['n_domains']
    if item_order == 'domain-outer':
        perm = None
    elif item_order == 'domain-inner':
        inds = np.reshape(np.arange(n_inputs, dtype=int), (n_domains, -1))    
        perm = inds.T.ravel()

    elif item_order == 'group-outer':
        if 'item' not in snap_type:
            raise ValueError('group-outer only works for items')
        groups_one = dd.item_group(np.arange(dd.ITEMS_PER_DOMAIN), **res['net_params'])
        groups = np.tile(groups_one, n_domains)
        perm = np.argsort(groups, kind='stable')
    else:
        perm = item_order

    if perm is not None:
        input_names = [input_names[i] for i in perm]
        rsa_mat = rsa_mat[np.ix_(perm, perm)]

    image = ax.imshow(rsa_mat)
    ax.set_xticks(range(n_inputs))
    ax.set_xticklabels(input_names)
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_yticks(range(n_inputs))
    ax.set_yticklabels(input_names)

    title = f'Epoch {res["snap_epochs"][snap_ind]}'
    if title_addon is not None:
        title += f' ({title_addon})'
    ax.set_title(title)
    
    if colorbar:
        ax.get_figure().colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    
    return image


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
    n_domains = res['net_params']['n_domains']
    reprs_embedded = reprs_embedded.reshape((n_snaps, n_domains, -1, dims))
    reprs_embedded = reprs_embedded.transpose((1, 2, 3, 0))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection=('3d' if dims == 3 else None))
    
    input_names = _get_names_for_snapshots(snap_type, **res['net_params'])
    
    if 'item' in snap_type:
        input_groups = dd.item_group(np.arange(dd.ITEMS_PER_DOMAIN), **res['net_params'])
    elif 'context' in snap_type:
        # No "groups," but use symbols for individual contexts (per domain) instead.
        input_groups = np.arange(4)
    else:
        raise ValueError('Unrecognized snapshot type')
        
    input_names = np.array(input_names).reshape((n_domains, -1))

    # skip red since it conflicts with the "end of trajectory" marker
    colors = list(filter(lambda c: c[-3:] != 'red', list(mcolors.TABLEAU_COLORS)))
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
            
    image = ax.imshow(corrs, interpolation='nearest')
    ax.set_yticks(range(len(input_names)))
    ax.set_yticklabels(input_names)
    ax.set_xticks([])
    ax.set_xlabel('Hidden layer neurons')
    ax.set_title(f'{title_label} correlation of hidden-to-attribute weights\n' + 
                 f'with input attributes (run {run_num+1}, epoch {epoch})')
    ax.get_figure().colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    
    return image


def make_unit_comparison_matrix(b_positive):
    """
    Make a model RDM with unit norm that compares two sets of pairs.
    Input should be a square bool matrix; output is a double matrix of the
    same shape with all positive entries equal w/ sum of squares 1/2,
    all negative entries equal w/ sum of squares 1/2,
    and each entry ij is positive iff b_positive[i,j] is true.
    """
    n_pos = np.sum(b_positive)
    n_neg = b_positive.size - n_pos
    
    # balance values such that sum of squares of both same and different cells is 1/2
    val_pos = np.sqrt(n_neg)
    val_neg = -np.sqrt(n_pos)
    rdm = np.where(b_positive, val_pos, val_neg)
    return rdm / np.linalg.norm(rdm)


def get_group_model_rdm(n_domains, snap_type='item'):
    """
    Make matrix, with unit Frobenius norm, that compares within-item-group to
    between-group dissimilarity irrespective of domain. The projection of an RDM onto
    this matrix indicates how strongly the representations considered make this distinction.
    """
    if 'item' not in snap_type:
        raise ValueError('Group model RDM for contexts does not exist')
        
    # make bool matrix of whether item pairs cross groups
    _, item_names = dd.get_items(n_domains)
    # TODO: rather than just binary same/different, use actual RDM of item attributes
    # Also make sure all the model RDMs are orthogonal.
    
    group_symbol = np.array([name[-1] for name in item_names])
    # find where group differs using broadcasting
    is_different_group = group_symbol[:, np.newaxis] != group_symbol[np.newaxis, :]
    
    return make_unit_comparison_matrix(is_different_group)


def get_domain_model_rdm(n_domains, snap_type='item'):
    """
    Similar to get_group_model_rdm, but with similarity between
    all items (or contexts) within each domain rather than between groups across domains.
    """
    input_names = _get_names_for_snapshots(snap_type, n_domains=n_domains)
    domain_letter = np.array([name[0] for name in input_names])
    is_different_domain = domain_letter[:, np.newaxis] != domain_letter[np.newaxis, :]
    
    return make_unit_comparison_matrix(is_different_domain)


def get_individual_model_rdm(n_domains, snap_type='item'):
    """
    Make a model RDM that simply treats each item (or context) as only similar to itself.
    """
    n_inputs = len(_get_names_for_snapshots(snap_type, n_domains=n_domains))
    is_different_item = ~np.eye(n_inputs, dtype=bool)
    return make_unit_comparison_matrix(is_different_item)


def get_model_rdm(model_type, n_domains, snap_type):
    return {
        'group': get_group_model_rdm,
        'domain': get_domain_model_rdm,
        'individual': get_individual_model_rdm
    }[model_type](n_domains, snap_type)

    
def get_rdm_projections(res, snap_type='item', normalize=False):
    """
    Make new "reports" (for each run, over time) of the projection of item similarity
    matrices onto the model cross-domain and domain RDM
    snap_name is the key of interest under the saved "snapshots" dict.
    If normalize is True, normalizes the magnitude of the RDM first.
    """
    # Get the full snapshots (for each run)
    with np.load(res['path'], allow_pickle=True) as resfile:
        try:
            snaps = resfile['snapshots'].item()[snap_type]
        except KeyError:
            raise ValueError(snap_type + ' snapshots not found for this dataset')

    n_domains = res['net_params']['n_domains']
    model_types = ['domain', 'individual']
    if 'item' in snap_type:
        model_types.append('group')
        
    models = {mtype: get_model_rdm(mtype, n_domains, snap_type) for mtype in model_types}
    
    n_runs, n_snap_epochs = snaps.shape[:2]
    projections = {dim: np.empty((n_runs, n_snap_epochs)) for dim in models}
    
    for k_run in range(n_runs):
        for k_epoch in range(n_snap_epochs):
            rdm = distance.squareform(distance.pdist(snaps[k_run, k_epoch]))
            if normalize:
                rdm = rdm / np.linalg.norm(rdm)
            
            for dim, model in models.items():
                projections[dim][k_run, k_epoch] = np.nansum(rdm * model)

    return projections


def get_mean_and_ci(series_set):
    """
    Given a set of N time series, compute and return the mean
    along with 95% confidence interval using a t-distribution.
    """
    n = series_set.shape[0]
    mean = np.mean(series_set, axis=0)
    stderr = np.std(series_set, axis=0) / np.sqrt(n)
    tdist = stats.t(loc=mean, scale=stderr, df=n-1)
    interval = stats.t.interval(0.95, df=n-1, loc=mean, scale=stderr)
    
    return mean, interval


def plot_rdm_projections(res, snap_type, model_types, axs, label=None, **plot_params):
    """
    Plot time series of item or context RDM projections onto given axes, with 95% CI.
    model_types should be a list of the same size as axs.
    """
    axs = axs.ravel()
    if len(axs) != len(model_types):
        raise ValueError('Wrong number of axes given')
    
    # get all the projections to start
    projections = get_rdm_projections(res, snap_type)
    
    layer = 'hidden' if 'hidden' in snap_type else 'repr'
    input_type = 'item' if 'item' in snap_type else 'context'
    
    for ax, mtype in zip(axs, model_types):
        try:
            mean, (lower, upper) = get_mean_and_ci(projections[mtype])
        except KeyError:
            raise ValueError(f'Model type {mtype} not defined for {snap_type} snapshots.')
        
        ax.plot(res['snap_epochs'], mean, label=label, **plot_params)
        ax.fill_between(res['snap_epochs'], lower, upper, **{'alpha': 0.3, **plot_params})
        ax.set_title(f'Correlation of {input_type} RDMs in {layer} layer with {mtype} model')
        
    