"""Useful functions for analyzing results of disjoint-domain net runs"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D # noqa
from scipy.cluster import hierarchy
from scipy.spatial import distance
from sklearn.manifold import MDS
from itertools import product

import disjoint_domain as dd

report_titles = {
    'loss': 'Mean loss',
    'accuracy': 'Mean accuracy',
    'etg_item': 'Epochs to learn new items',
    'etg_context': 'Epochs to learn new contexts',
    'test_accuracy': 'Accuracy on novel item/context pairs'
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
        input_type: get_mean_repr_dists(repr_snaps)
        for input_type, repr_snaps in snaps.items()
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


def plot_report(ax, res, report_type, **plot_params):
    """Make a standard plot of mean loss, accuracy, etc. over training"""
    xaxis = res['etg_epochs'] if report_type[:3] == 'etg' else res['report_epochs']
    ax.plot(xaxis, res['reports'][report_type], '.-', **plot_params)
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
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2, fontsize='x-small')


def plot_rsa(ax, res, input_type, snap_ind, title_addon=None, item_order='domain-outer'):
    """
    Plot an RSA matrix for the representation of items or contexts at a particular epoch
    item_order: controls the order of items in the matrix - either a string or an
    array to use a custom permutation.
        'domain-outer' (default): items are grouped together by domain
        'domain-inner': first item of each domain, then second item, etc.
        'type-outer': grouped by type (circles/squares/stars), then by domain
    """
    _, input_names = {
        'item': dd.get_items,
        'context': dd.get_contexts
    }[input_type](**res['net_params'])
    n_inputs = len(input_names)

    rsa_mat = res['repr_dists'][input_type]['snaps'][snap_ind]

    n_domains = res['net_params']['n_domains']
    if item_order == 'domain-outer':
        perm = None
    elif item_order == 'domain-inner':
        inds = np.reshape(np.arange(n_inputs, dtype=int), (n_domains, -1))    
        perm = inds.T.ravel()

    elif item_order == 'type-outer':
        if input_type != 'item':
            raise ValueError('type-outer only works for items')
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
    
    return image


def plot_repr_dendrogram(ax, res, input_type, snap_ind, title_addon=None):
    """Similar to plot_rsa, but show dendrogram rather than RSA matrix"""
    _, input_names = {
        'item': dd.get_items,
        'context': dd.get_contexts
    }[input_type](**res['net_params'])

    dists_compressed = distance.squareform(res['repr_dists'][input_type]['snaps'][snap_ind])
    z = hierarchy.linkage(dists_compressed, optimal_ordering=True)
    hierarchy.dendrogram(z, labels=input_names, count_sort=True, ax=ax)
    title = f'Epoch {res["snap_epochs"][snap_ind]}'
    if title_addon is not None:
        title += f' ({title_addon})'
    ax.set_title(title)


def plot_repr_trajectories(res, input_type, dims=2, title_label=''):
    """
    Plot trajectories of each item or context representation over training
    using MDS. Can plot in 3D by settings dims to 3.
    Returns figure and axes.
    """
    embedding = MDS(n_components=dims, dissimilarity='precomputed')
    reprs_embedded = embedding.fit_transform(res['repr_dists'][input_type]['all'])

    # reshape and permute to aid plotting
    n_snaps = len(res['snap_epochs'])
    n_domains = res['net_params']['n_domains']
    reprs_embedded = reprs_embedded.reshape((n_snaps, n_domains, -1, dims))
    reprs_embedded = reprs_embedded.transpose((1, 2, 3, 0))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection=('3d' if dims == 3 else None))
    
    if input_type == 'item':
        _, input_names = dd.get_items(**res['net_params'])
        input_groups = dd.item_group(np.arange(dd.ITEMS_PER_DOMAIN), **res['net_params'])
    else:
        _, input_names = dd.get_contexts(**res['net_params'])
        input_groups = np.arange(4)
        
    input_names = np.array(input_names).reshape((n_domains, -1))

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

    ax.set_title(f'{title_label} {input_type} representations over training\n' +
                 'color = domain, marker = type within domain')

    return fig, ax
