"""Functions for analyzing the training statistics and representations of a feedforward network after training"""
from copy import deepcopy
import numpy as np
from scipy import stats
from scipy.spatial import distance

import util


def calc_mean_pairwise_repr_fn(repr_snaps, pairwise_fn, calc_all, include_individual=False):
    """
    Make matrices of some function on pairs of input representations, such as distance or correlation,
    averaged over training runs.

    repr_snaps is a n_runs x n_snap_epochs x n_inputs x n_rep array, where
    n_inputs is the number of input instances and n_rep is the size of the representation.

    Outputs a dict with keys:
        'snaps': gives the mean values of the function over runs between all pairs of inputs for each
        recorded epoch, with dimension 0 corresponding to epoch.
    
        'all': only included if calc_all is True. Gives mean values of the function calculated on all possible
        pairs of conditions, where a condition is an (input, epoch) pair. This is useful for making an MDS map
        of the training process. The nth n_inputs x n_inputs block diagonal entry of the resulting matrix is 
        identical to the nth subarray (along dimension 0) in 'snaps'.
        
        'snaps_each': only included if include_individual is True. Same as 'snaps' but without averaging
        across runs, so dimension 0 is runs and dimension 1 is epochs.
    """
    
    if calc_all:
        n_runs, n_snap_epochs, n_inputs, n_rep = repr_snaps.shape
        snaps_flat = np.reshape(repr_snaps, (n_runs, n_snap_epochs * n_inputs, n_rep))        
        dists_all = np.stack([pairwise_fn(run_snaps) for run_snaps in snaps_flat])
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
            np.stack([pairwise_fn(epoch_snaps) for epoch_snaps in run_snaps])
            for run_snaps in repr_snaps
        ])
        mean_dists_snaps = np.nanmean(dists_snaps, axis=0)
        
        if include_individual:
            return {'snaps': mean_dists_snaps, 'snaps_each': dists_snaps}
        else:
            return {'snaps': mean_dists_snaps}

        
def calc_mean_repr_dists(repr_snaps, dist_metric='euclidean', include_individual=False, calc_all=True, **_extra):
    def dist_fn(snaps):
        # noinspection PyTypeChecker
        return distance.squareform(distance.pdist(snaps, metric=dist_metric))
    
    return calc_mean_pairwise_repr_fn(repr_snaps, dist_fn, include_individual=include_individual, calc_all=calc_all)


def calc_mean_repr_corr(repr_snaps, corr_type='pearson', include_individual=False, **_extra):
    if corr_type == 'pearson':
        def corr_fn(snaps):
            return np.corrcoef(snaps)
    elif corr_type == 'spearman':
        def corr_fn(snaps):
            return stats.spearmanr(snaps, axis=1)[0]
    else:
        raise ValueError(f'Unrecognized correlation type "{corr_type}"')
    
    return calc_mean_pairwise_repr_fn(repr_snaps, corr_fn, calc_all=False, include_individual=include_individual)
        

def get_result_means(res_path, subsample_snaps=1, runs=slice(None), dist_metric='euclidean', corr_type='pearson',
                     compute_full_rdms=False, include_individual_corr_mats=False, include_individual_rdms=False,
                     extra_keys=None):
    """
    Get dict of data (meaned over runs) from saved file
    If subsample_snaps is > 1, use only every nth snapshot
    Indexes into runs using the 'runs' argument
    """
    if extra_keys is None:
        extra_keys = []
    
    with np.load(res_path, allow_pickle=True) as resfile:
        snaps = resfile['snapshots'].item() if 'snapshots' in resfile else {}
        reports = resfile['reports'].item()
        net_params = resfile['net_params'].item()
        train_params = resfile['train_params'].item()
        
        if 'per_net_params' in resfile:
            extra_keys.append('per_net_params')
        extra = {key: resfile[key] for key in extra_keys}
        
    num_epochs = train_params['num_epochs']
    total_epochs = sum(num_epochs) if isinstance(num_epochs, tuple) else num_epochs
    
    # take subset of snaps and reports if necessary
    snaps = {stype: snap[runs, ::subsample_snaps, ...] for stype, snap in snaps.items()}
    reports = {rtype: report[runs, ...] for rtype, report in reports.items()}
    
    # if we did domain holdout testing, exclude the last domain from snapshots
    nd = net_params['n_domains'] if 'n_domains' in net_params else len(net_params['trees'])
    nd_train = nd
    if 'domains_to_hold_out' in train_params and train_params['domains_to_hold_out'] > 0:
        nd_train = nd - train_params['domains_to_hold_out']
    elif ('holdout_testing' in train_params and train_params['holdout_testing'] == 'domain' or
          'do_tree_holdout' in train_params and train_params['do_tree_holdout']):
        nd_train = nd - 1

    net_params['n_train_domains'] = nd_train
    if nd_train < nd:
        for key, snap in snaps.items():
            snaps[key] = np.delete(snap, slice(snap.shape[2] // nd * nd_train, None), axis=2)

    report_stats = {
        report_type: util.get_mean_and_ci(report)
        for report_type, report in reports.items()
    }
    
    report_means = {report_type: rstats[0] for report_type, rstats in report_stats.items()}
    report_cis = {report_type: rstats[1] for report_type, rstats in report_stats.items()}
    
    report_mstats = {
        report_type: util.get_median_and_ci(report)
        for report_type, report in reports.items()
    }
    
    report_meds = {report_type: rstats[0] for report_type, rstats in report_mstats.items()}
    report_med_cis = {report_type: rstats[1] for report_type, rstats in report_mstats.items()}

    if len(snaps) > 0:
        if compute_full_rdms or include_individual_rdms:
            extra['repr_dists'] = {
                snap_type: calc_mean_repr_dists(repr_snaps, metric=dist_metric, include_individual=include_individual_rdms, 
                                                calc_all=compute_full_rdms)
                for snap_type, repr_snaps in snaps.items()
            }
            
        extra['repr_corr'] = {
            snap_type: calc_mean_repr_corr(repr_snaps, corr_type=corr_type, include_individual=include_individual_corr_mats)
            for snap_type, repr_snaps in snaps.items()
        }
        
        try:
            snap_freq_scale = train_params['snap_freq_scale']
        except KeyError:
            snap_freq_scale = 'lin'
            
        try:
            include_final_eval = train_params['include_final_eval']
        except KeyError:
            include_final_eval = False

        extra['snap_epochs'] = util.calc_snap_epochs(train_params['snap_freq'], total_epochs,
                                                     snap_freq_scale, include_final_eval)[::subsample_snaps]

    report_epochs = np.arange(len(report_means['loss'])) * train_params['report_freq']
#     report_epochs = np.arange(0, total_epochs + 1, train_params['report_freq'])
    
    if 'reports_per_test' in train_params:
        if train_params['reports_per_test'] == np.inf:
            extra['etg_epochs'] = report_epochs[[0]]
        else:
            extra['etg_epochs'] = report_epochs[::train_params['reports_per_test']]
        
    return {
        'path': res_path,
        'snaps': snaps,
        'reports': report_means,
        'report_cis': report_cis,
        'report_meds': report_meds,
        'report_med_cis': report_med_cis,
        'net_params': net_params,
        'train_params': train_params,
        'report_epochs': report_epochs,
        **extra
    }


def load_nested_runs_w_cache(res_path_dict, res_path_cache, res_data_cache, load_settings):
    """
    Given a nested dictionary of run paths, recursively load results using given load settings, using the cached data when possible.
    Supports a special syntax to support loading runs with multiple held-out domains into multiple entries of the output.
    Instead of a path, the run that copies from OtherRun (at the same level of hierarchy), using the held-out domain N
    (counting from the first held-out domain) is specified as {'inherit_from': 'OtherRun', 'etg_domain': N}.
    """
    loaded_data = {}
    # first load all entries that have paths
    for key, val in res_path_dict.items():
        if not isinstance(val, dict):
            loaded_data[key] = (res_data_cache[key] if res_data_cache is not None and res_path_cache is not None and
                                key in res_path_cache and res_path_cache[key] == val
                                else get_result_means(val, **load_settings))
            
    for key, val in res_path_dict.items():
        if isinstance(val, dict):
            if 'inherit_from' in val:  # inherit from other loaded dataset, switching out the held-out domain 
                loaded_data[key] = deepcopy(loaded_data[val['inherit_from']])
                etg_key = f'etg_domain{val["etg_domain"]}'
                for report_key in ['reports', 'report_cis', 'report_meds', 'report_med_cis']:
                    if report_key in loaded_data[key]:
                        assert etg_key in loaded_data[key][report_key], f'Cannot inherit from {val["inherit_from"]} - held-out domain {val["etg_domain"]} missing'
                        loaded_data[key][report_key]['etg_domain'] = loaded_data[key][report_key][etg_key]    
            else:  # recurse
                sub_res_path_cache = res_path_cache[key] if key in res_path_cache else None
                sub_res_data_cache = res_data_cache[key] if key in res_data_cache else None
                loaded_data[key] = load_nested_runs_w_cache(val, sub_res_path_cache, sub_res_data_cache, load_settings)
            
    # put the results in the original order
    return {key: loaded_data[key] for key in res_path_dict}


def flatten_data_dict(data_dict, prefix=None):
    """Flatten a dict containing loaded runs, using the UNIX path convention, without flattening the runs themselves"""
    res = {}
    for key, val in data_dict.items():
        abskey = '/'.join([prefix, key]) if prefix is not None else key
        if not isinstance(val, dict) or 'snaps' in val:
            res[abskey] = val
        else:
            res.update(flatten_data_dict(val, prefix=abskey))
    return res


def plot_report(ax, res, report_type, with_ci=True, median=False, label=None, title=None,
                **plot_params):
    """Make a standard plot of mean loss, accuracy, etc. over training"""
    center_key = 'report_meds' if median else 'reports'
    ci_key = 'report_med_cis' if median else 'report_cis'
    
    xaxis = res['etg_epochs'] if 'etg' in report_type else res['report_epochs']
    ax.plot(xaxis, res[center_key][report_type], label=label, **plot_params)
    if with_ci:
        ax.fill_between(xaxis, *res[ci_key][report_type], alpha=0.3, **plot_params)
        
    ax.set_xlabel('Epoch')
    if title is None:
        title = report_type
    ax.set_title(title)
    
    
def plot_individual_reports(ax, res, report_type, title=None, **plot_params):
    """Take a closer look by looking at a statistic for each run in a set."""
    with np.load(res['path'], allow_pickle=True) as resfile:
        all_reports = resfile['reports'].item()[report_type]

    xaxis = res['etg_epochs'] if report_type[:3] == 'etg' else res['report_epochs']

    for (i, report) in enumerate(all_reports):
        ax.plot(xaxis, report, label=f'Run {i+1}', **plot_params)

    ax.set_xlabel('Epoch')
    if title is None:
        title = report_type
    ax.set_title(title + ' (each run)')
    util.outside_legend(ax, ncol=2, fontsize='x-small')
    
    
def plot_rdm(ax, res, snap_type, snap_ind, labels, title_addon=None,
             colorbar=True, actual_rdm=None, tick_fontsize='x-small'):
    """
    Plot a representational dissimilarity matrix (RDM) for the representation of inputs at a particular epoch.
    If 'actual_rdm' is provided, overrides the matrix to plot.
    """
    if actual_rdm is None:
        rdm = res['repr_dists'][snap_type]['snaps'][snap_ind]
    else:
        rdm = actual_rdm

    image = util.plot_matrix_with_labels(ax, rdm, labels, colorbar=colorbar, tick_fontsize=tick_fontsize, bipolar=False)

    title = f'Epoch {res["snap_epochs"][snap_ind]}'
    if title_addon is not None:
        title += f'\n({title_addon})'
    ax.set_title(title)

    return image


def plot_repr_corr(ax, res, snap_type, snap_ind, labels, title_addon=None,
                   colorbar=True, actual_mat=None, tick_fontsize='x-small'):
    """
    Plot a matrix of the correlation between input representations at a particular epoch.
    If actual_mat is provided, overrides the matrix to plot.
    """
    if actual_mat is None:
        mat = res['repr_corr'][snap_type]['snaps'][snap_ind]
    else:
        mat = actual_mat
        
    image = util.plot_matrix_with_labels(ax, mat, labels, colorbar=colorbar, tick_fontsize=tick_fontsize,
                                         bipolar=True, max_val=1.0)
    
    # domain dividers
    n_domains = res['net_params']['n_train_domains']
    items_per_domain = mat.shape[0] // n_domains
    for divider_pt in np.arange(items_per_domain-0.5, mat.shape[0]-0.5, items_per_domain):
        ax.plot(ax.get_xlim(), [divider_pt, divider_pt], 'k', lw=1)
        ax.plot([divider_pt, divider_pt], ax.get_ylim(), 'k', lw=1)
    
    title = f'Epoch {res["snap_epochs"][snap_ind]}'
    if title_addon is not None:
        title += f'\n({title_addon})'
    ax.set_title(title)

    return image


def get_rdm_projections(res, snap_type, models, corr_type='pearson'):
    """
    Make new "reports" (for each run, over time) of the projection of item similarity
    matrices onto the model cross-domain and domain RDM
    snap_name is the key of interest under the saved "snapshots" dict.
    'item_full' and 'context_full' are special "snap types" that combine (concatenate) all
    snapshots with item and context inputs respectively (i.e. repr and hidden layers).
    
    If models are passed in, it should be a dictionary of normalized model matrices.
    Models can also be n_runs x n_items x n_items to use a separate one for each run.
    """
    # Get the full snapshots (for each run)
    snaps_each = res['repr_dists'][snap_type]['snaps_each']
            
    n_runs, n_snap_epochs = snaps_each.shape[:2]
    projections = {dim: np.empty((n_runs, n_snap_epochs)) for dim in models}
    
    for k_run, run_snaps in enumerate(snaps_each):
        for k_epoch, rdm in enumerate(run_snaps):
            normed_rdm = util.center_and_norm_rdm(rdm)

            for dim, model in models.items():
                if model.ndim > 3:
                    raise ValueError('Models must be at most 3-dimensional')
                if model.ndim == 3:
                    model = model[k_run]
                    
                rdm_to_use = rdm if dim == 'uniformity' else normed_rdm
                
                if corr_type == 'pearson':
                    projections[dim][k_run, k_epoch] = np.corrcoef([rdm_to_use.ravel(), model.ravel()])[0, 1]
                elif corr_type == 'spearman':
                    projections[dim][k_run, k_epoch] = stats.spearmanr(rdm_to_use, model, axis=None)[0]
                elif corr_type == 'kendall':
                    projections[dim][k_run, k_epoch] = stats.kendalltau(rdm_to_use, model)[0]
                else:
                    raise ValueError(f'Unrecognized correlation type "{corr_type}"')                   
    return projections