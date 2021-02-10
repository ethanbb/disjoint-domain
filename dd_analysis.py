"""Useful functions for analyzing results of disjoint-domain net runs"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # noqa
from mpl_toolkits import axes_grid1
from scipy.cluster import hierarchy
from scipy.spatial import distance
from scipy.linalg import block_diag
from scipy import stats
from sklearn.manifold import MDS
from statsmodels.regression.linear_model import OLS
from patsy import dmatrices

import disjoint_domain as dd

report_titles = {
    'loss': 'Mean loss',
    'accuracy': 'Mean accuracy',
    'weighted_acc': 'Mean accuracy (weighted)',
    'etg_item': 'Epochs to learn new items',
    'etg_context': 'Epochs to learn new contexts',
    'etg_domain': 'Epochs to learn new domain',
    'test_accuracy': 'Accuracy on novel item/context pairs',
    'test_weighted_acc': 'Mean generalization accuracy (weighted)'
}


def get_mean_repr_dists(repr_snaps, metric='euclidean', calc_all=True,
                        include_individual=False):
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
    
    If calc_all is False, only computes the distances within each epoch. (Distances across
    epochs are generally only used to make an MDS plot.)
    
    If include_individual is True, includes snaps_each which is not meaned across runs.
    """

    def dist_fn(snaps):
        if metric == 'spearman':
            return 1 - stats.spearmanr(snaps, axis=1)[0]
        else:
            return distance.squareform(distance.pdist(snaps, metric=metric))
    
    if calc_all:
        n_runs, n_snap_epochs, n_inputs, n_rep = repr_snaps.shape
        snaps_flat = np.reshape(repr_snaps, (n_runs, n_snap_epochs * n_inputs, n_rep))        
        dists_all = np.stack([dist_fn(run_snaps) for run_snaps in snaps_flat])
        mean_dists_all = np.nanmean(dists_all, axis=0)

        mean_dists_snaps = np.empty((n_snap_epochs, n_inputs, n_inputs))
        for k_epoch in range(n_snap_epochs):
            this_slice = slice(k_epoch * n_inputs, (k_epoch+1) * n_inputs)
            mean_dists_snaps[k_epoch] = mean_dists_all[this_slice, this_slice]
            
        dists_out = {'all': mean_dists_all, 'snaps': mean_dists_snaps}
        
        if include_individual:
            dists_snaps = np.empty((n_runs, n_snap_epochs, n_inputs, n_inputs))
            for k_epoch in range(n_snap_epochs):
                this_slice = slice(k_epoch * n_inputs, (k_epoch+1) * n_inputs)
                dists_snaps[:, k_epoch] = dists_all[:, this_slice, this_slice]

            dists_out['snaps_each'] = dists_snaps
            
        return dists_out
            
    else:
        # Directly calculate distances at each epoch
        dists_snaps = np.stack([
            np.stack([dist_fn(epoch_snaps) for epoch_snaps in run_snaps])
            for run_snaps in repr_snaps
        ])
        mean_dists_snaps = np.nanmean(dists_snaps, axis=0)
        
        if include_individual:
            return {'snaps': mean_dists_snaps, 'snaps_each': dists_snaps}
        else:
            return {'snaps': mean_dists_snaps}


def get_mean_and_ci(series_set):
    """
    Given a set of N time series, compute and return the mean
    along with 95% confidence interval using a t-distribution.
    """
    n = series_set.shape[0]
    mean = np.mean(series_set, axis=0)
    stderr = np.std(series_set, axis=0) / np.sqrt(n)
    interval = stats.t.interval(0.95, df=n-1, loc=mean, scale=stderr)
    
    return mean, interval


def get_result_means(res_path, subsample_snaps=1, runs=slice(None),
                     dist_metric='euclidean', calc_all_repr_dists=True, include_individual_rdms=False):
    """
    Get dict of data (meaned over runs) from saved file
    If subsample_snaps is > 1, use only every nth snapshot
    Indexes into runs using the 'runs' argument
    """    
    with np.load(res_path, allow_pickle=True) as resfile:
        snaps = resfile['snapshots'].item()
        reports = resfile['reports'].item()
        net_params = resfile['net_params'].item()
        train_params = resfile['train_params'].item()
        ys = resfile['ys']
        
    # take subset of snaps and reports if necessary
    snaps = {stype: snap[runs, ::subsample_snaps, ...] for stype, snap in snaps.items()}
    reports = {rtype: report[runs, ...] for rtype, report in reports.items()}

    mean_repr_dists = {
        snap_type: get_mean_repr_dists(repr_snaps, metric=dist_metric,
                                       calc_all=calc_all_repr_dists, include_individual=include_individual_rdms)
        for snap_type, repr_snaps in snaps.items()
    }
    
    # also do full item and context repr dists
    item_full_snaps = np.concatenate([snaps[stype] for stype in ['item', 'item_hidden'] if stype in snaps],
                                     axis=3)
    mean_repr_dists['item_full'] = get_mean_repr_dists(item_full_snaps, metric=dist_metric,
                                                       calc_all=calc_all_repr_dists, include_individual=include_individual_rdms)
    ctx_full_snaps = np.concatenate([snaps[stype] for stype in ['context', 'context_hidden'] if stype in snaps],
                                    axis=3)
    mean_repr_dists['context_full'] = get_mean_repr_dists(ctx_full_snaps, metric=dist_metric,
                                                          calc_all=calc_all_repr_dists, include_individual=include_individual_rdms)

    report_stats = {
        report_type: get_mean_and_ci(report)
        for report_type, report in reports.items()
    }
    
    report_means = {report_type: rstats[0] for report_type, rstats in report_stats.items()}
    report_cis = {report_type: rstats[1] for report_type, rstats in report_stats.items()}

    snap_epochs = dd.calc_snap_epochs(
        train_params['snap_freq'], train_params['snap_freq_scale'],
        train_params['num_epochs'])[::subsample_snaps]

    report_epochs = np.arange(0, train_params['num_epochs'], train_params['report_freq'])
    etg_epochs = report_epochs[::train_params['reports_per_test']]

    return {
        'path': res_path,
        'repr_dists': mean_repr_dists,
        'reports': report_means,
        'report_cis': report_cis,
        'net_params': net_params,
        'train_params': train_params,
        'snap_epochs': snap_epochs,
        'report_epochs': report_epochs,
        'etg_epochs': etg_epochs,
        'ys': ys
    }


def auto_subplots(n_rows, n_cols, ax_dims=(4, 4), prop_cycle=None):
    """Make subplots, automatically adjusting the figsize, and without squeezing"""
    figsize = (ax_dims[0] * n_cols, ax_dims[1] * n_rows)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    for ax in axs.ravel():
        ax.set_prop_cycle(prop_cycle)
    return fig, axs


def make_plot_grid(n, n_cols=3, ax_dims=(4, 4), ravel=True, prop_cycle=None):
    """
    Create a figure with n axes arranged in a grid, and return the fig and flat array of axes
    ax_dims is the width, height of each axes in inches (or whatever units matplotlib uses)
    """
    n_rows = (n-1) // n_cols + 1
    fig, axs = auto_subplots(n_rows, n_cols, ax_dims=ax_dims, prop_cycle=prop_cycle)
    return fig, axs.ravel() if ravel else axs


def outside_legend(ax, **legend_params):
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), **legend_params)
    
    
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot. (source: https://stackoverflow.com/a/33505522)"""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def plot_report(ax, res, report_type, with_ci=True, label=None, **plot_params):
    """Make a standard plot of mean loss, accuracy, etc. over training"""
    xaxis = res['etg_epochs'] if report_type[:3] == 'etg' else res['report_epochs']
    ax.plot(xaxis, res['reports'][report_type], label=label, **plot_params)
    if with_ci:
        ax.fill_between(xaxis, *res['report_cis'][report_type], alpha=0.3)
        
    ax.set_xlabel('Epoch')
    ax.set_title(report_titles[report_type])


def plot_individual_reports(ax, res, all_reports, report_type, **plot_params):
    """
    Take a closer look by looking at a statistic for each run in a set.
    all_reports should be the full reports dict of runs x epochs matrices,
    loaded from the results file.
    """
    xaxis = res['etg_epochs'] if report_type[:3] == 'etg' else res['report_epochs']

    for (i, report) in enumerate(all_reports[report_type]):
        ax.plot(xaxis, report, label=f'Run {i+1}', **plot_params)

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
    
    
def plot_matrix_with_labels(ax, mat, labels, colorbar=True, **imshow_params):
    """Helper to plot a square matrix with each row/column labeled with 'labels'"""
    n = len(labels)
    assert mat.shape[0] == mat.shape[1], 'Matrix must be square'
    assert n == mat.shape[0], 'Wrong number of labels'
    
    image = ax.imshow(mat, **imshow_params)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    
    if colorbar:
        add_colorbar(image)
    
    return image
    

def plot_rsa(ax, res, snap_type, snap_ind, title_addon=None, item_order='domain-outer',
             colorbar=True, rsa_mat=None):
    """
    Plot an RSA matrix for the representation of items or contexts at a particular epoch
    item_order: controls the order of items in the matrix - either a string or an
    array to use a custom permutation.
        'domain-outer' (default): items are grouped together by domain
        'domain-inner': first item of each domain, then second item, etc.
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
    else:
        perm = item_order

    if perm is not None:
        input_names = [input_names[i] for i in perm]
        rsa_mat = rsa_mat[np.ix_(perm, perm)]

    image = plot_matrix_with_labels(ax, rsa_mat, input_names, colorbar=colorbar)

    title = f'Epoch {res["snap_epochs"][snap_ind]}'
    if title_addon is not None:
        title += f' ({title_addon})'
    ax.set_title(title)

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
            
    image = ax.imshow(corrs, interpolation='nearest')
    ax.set_yticks(range(len(input_names)))
    ax.set_yticklabels(input_names)
    ax.set_xticks([])
    ax.set_xlabel('Hidden layer neurons')
    ax.set_title(f'{title_label} correlation of hidden-to-attribute weights\n' + 
                 f'with input attributes (run {run_num+1}, epoch {epoch})')
    add_colorbar(image)
    
    return image


def norm_rdm(dist_mat):
    """Helper to make a unit model RDM"""
    return dist_mat / np.linalg.norm(dist_mat)


def center_and_norm_rdm(dist_mat):
    """Helper to make a unit model RDM that sums to 0"""
    dist_mat_centered = dist_mat - np.mean(dist_mat)
#     dist_vec = distance.squareform(dist_mat)
#     dist_mat_centered = distance.squareform(dist_vec - np.mean(dist_vec))
    return norm_rdm(dist_mat_centered)
    

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
    #models = {'spread': norm_rdm(np.ones((n_items, n_items)))}
    models = {'uniformity': np.full((n_items, n_items), 1 / n_items**2)}
    
    # in-domain attribute distance
    item_attr_rdm = dd.get_item_attribute_rdm(ctx_per_domain, attrs_per_context)
    item_attr_model_1domain = center_and_norm_rdm(item_attr_rdm) / np.sqrt(n_domains)
    models['attribute_similarity'] = block_diag(*[item_attr_model_1domain for _ in range(n_domains)])
    
    # cross vs. within-domain
    is_domain_eq = block_diag(*[np.ones((dd.ITEMS_PER_DOMAIN, dd.ITEMS_PER_DOMAIN), dtype=bool)
                                for _ in range(n_domains)])
    nz_where_domain_ne = 1 - is_domain_eq
    models['same_vs_different_domain'] = center_and_norm_rdm(nz_where_domain_ne)
    
    # cross-domain group
    is_circle = np.equal(dd.item_group(clusters=clusters), 0)
    is_diff_group = is_circle[:, np.newaxis] != is_circle[np.newaxis, :]
    diff_group_centered = is_diff_group - np.mean(is_diff_group)
    diff_group_tiled = np.tile(diff_group_centered, (n_domains, n_domains))
    diff_group_tiled[is_domain_eq] = 0  # make block diagonal zero
    models['cross_domain_group_match'] = norm_rdm(diff_group_tiled)
    
#     # combination of attribute distance and cross-domain group
#     tiled_item_attr = np.tile(item_attr_model_1domain, (n_domains, n_domains))
#     models['attr_with_cross_domain'] = center_and_norm_rdm(models['attribute_similarity'] + (0.5 * nz_where_domain_ne) * tiled_item_attr)
     
    return models
    

def make_ortho_context_rsa_models(n_domains, ctx_per_domain=4, **_extra):
    """
    Makes a set of model RDMs for context representations. Similar to make_ortho_item_rsa_models,
    but with only the 'uniformity' and 'cross_vs_in_domain' types.
    """
    #models = {'spread': norm_rdm(1 - np.eye(n_domains * ctx_per_domain))}
    n_contexts = n_domains * ctx_per_domain
    models = {'uniformity': np.full((n_contexts, n_contexts), 1 / n_contexts**2)}
    
    # cross vs. within-domain
    is_domain_eq = block_diag(*[np.ones((ctx_per_domain, ctx_per_domain), dtype=bool)
                                for _ in range(n_domains)])
    nz_where_domain_ne = 1 - is_domain_eq
    models['same_vs_different_domain'] = center_and_norm_rdm(nz_where_domain_ne)
    
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


def plot_rsa_model(ax, model, input_type='item'):
    """Helper to plot model RDM with a good colormap (to show 0 entries as white)"""
    names = _get_names_for_snapshots(input_type)
    max_absval = np.max(np.abs(model))
    return plot_matrix_with_labels(ax, model, names, cmap='seismic', vmin=-max_absval, vmax=max_absval)


def get_rdm_projections(res, snap_type='item', normalize=True):
    """
    Make new "reports" (for each run, over time) of the projection of item similarity
    matrices onto the model cross-domain and domain RDM
    snap_name is the key of interest under the saved "snapshots" dict.
    Also includes 'spread' which is just the fro-norm of the RDM.
    If normalize is True, normalizes each RDM
    before projecting onto each model matrix.
    'item_full' and 'context_full' are special "snap types" that combine (concatenate) all
    snapshots with item inputs and context inputs respectively (i.e. repr and hidden layers).
    """
    # Get the full snapshots (for each run)
    with np.load(res['path'], allow_pickle=True) as resfile:
        snap_dict = resfile['snapshots'].item()
        if snap_type == 'item_full':
            snaps = [snap_dict[key] for key in ['item', 'item_hidden'] if key in snap_dict]
            snaps = np.concatenate(snaps, axis=3)
        elif snap_type == 'context_full':
            snaps = [snap_dict[key] for key in ['context', 'context_hidden'] if key in snap_dict]
            snaps = np.concatenate(snaps, axis=3)
        else:
            try:
                snaps = snap_dict[snap_type]
            except KeyError:
                raise ValueError(snap_type + ' snapshots not found for this dataset')

    if 'item' in snap_type:
        models = make_ortho_item_rsa_models(**res['net_params'])
    elif 'context' in snap_type:
        models = make_ortho_context_rsa_models(**res['net_params'])
    else:
        raise ValueError(f'Snapshot type {snap_type} not recognized')
            
    n_runs, n_snap_epochs = snaps.shape[:2]
    projections = {dim: np.empty((n_runs, n_snap_epochs)) for dim in models}
    projections['spread'] = np.empty((n_runs, n_snap_epochs))
    
    for k_run in range(n_runs):
        for k_epoch in range(n_snap_epochs):
            rdm = distance.squareform(distance.pdist(snaps[k_run, k_epoch]))
            
            # make special "spread" one which is the fro norm
            normed_rdm = np.linalg.norm(rdm)
            projections['spread'][k_run, k_epoch] = normed_rdm
            
            if normalize:
                rdm /= normed_rdm
            
            for dim, model in models.items():
                projections[dim][k_run, k_epoch] = np.nansum(rdm * model)

    return projections


def plot_rdm_projections(res, snap_type, axs, normalize=False, label=None, **plot_params):
    """
    Plot time series of item or context RDM projections onto given axes, with 95% CI.
    model_types should be a list of the same size as axs.
    """    
    # get all the projections to start
    projections = get_rdm_projections(res, snap_type, normalize=normalize)
    model_types = projections.keys()
    
    axs = axs.ravel()
    if len(axs) != len(model_types):
        raise ValueError(f'Wrong number of axes given (expected {len(model_types)})')
    
    layer = 'hidden layer' if 'hidden' in snap_type else 'all layers' if 'full' in snap_type else 'repr layer'
    input_type = 'item' if 'item' in snap_type else 'context'
    
    for ax, mtype in zip(axs, model_types):
        try:
            mean, (lower, upper) = get_mean_and_ci(projections[mtype])
        except KeyError:
            raise ValueError(f'Model type {mtype} not defined for {snap_type} snapshots.')
        
        ax.plot(res['snap_epochs'], mean, label=label, **plot_params)
        ax.fill_between(res['snap_epochs'], lower, upper, **{'alpha': 0.3, **plot_params})
        
        ax.set_title(f'Projection of{" normalized" if normalize else ""}' + 
                     f' {input_type} RDMs in {layer} onto {mtype} model')


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


def _mean_attr_freqs_for_attr_vecs(y, ctx_per_domain, train_items=slice(None)):
    """
    Given some matrix of attributes (columns) for each item (rows),
    return a vector of mean attribute frequencies.
    """
    inputs_per_domain = dd.ITEMS_PER_DOMAIN * ctx_per_domain
    n_items = len(y) // ctx_per_domain
    n_domains = n_items // dd.ITEMS_PER_DOMAIN

    # first collapse attribute matrix over contexts
    y_collapsed = np.empty((n_items, y.shape[1]))
    for d in range(n_domains):
        for i in range(dd.ITEMS_PER_DOMAIN):
            d_offset = d * inputs_per_domain
            i_abs = i + d * dd.ITEMS_PER_DOMAIN
            strided_item_inds = np.arange(0, inputs_per_domain, dd.ITEMS_PER_DOMAIN) + d_offset + i
            y_collapsed[i_abs, :] = np.sum(y[strided_item_inds], axis=0)

    # exlcude items that weren't used for training
    y_collapsed = y_collapsed[train_items]

    # for each item, combine attributes with frequency of each attribute
    attr_freq = np.sum(y_collapsed, axis=0)
    return (y_collapsed @ attr_freq) / np.sum(y_collapsed, axis=1)


def get_mean_attr_freqs(res, train_items=slice(None)):
    """
    Returns a vector of length n_items for each individual run which reports the the mean frequency of
    all "on" attributes (# of items with that attribute), across all contexts, for each item.
    """
    ys = res['ys']
    ctx_per_domain = res['net_params']['ctx_per_domain']
    return np.stack([_mean_attr_freqs_for_attr_vecs(y, ctx_per_domain, train_items)
                     for y in ys])


def get_attr_freq_dist_mats(res, train_items=slice(None)):
    """
    Returns a matrix for each individual run indicating how close the mean # of 
    attributes shared with other items is for each item
    """
    mean_attr_freqs = get_mean_attr_freqs(res, train_items)
    return np.abs(mean_attr_freqs[:, np.newaxis, :] - mean_attr_freqs[:, :, np.newaxis])


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
    mean, ci = get_mean_and_ci(corrs)
    ax.plot(res['snap_epochs'], mean, label=label, **plot_params)
    ax.fill_between(res['snap_epochs'], *ci, alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Correlation (r)')
    ax.set_title('Correlation of item representation distance with\ndifference in attribute frequency')
