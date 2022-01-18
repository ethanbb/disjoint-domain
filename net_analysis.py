"""Functions for analyzing the training statistics and representations of a feedforward network after training"""
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


def calc_mean_repr_corr(repr_snaps, corr_type='pearson', include_individual_corr_mats=False, **_extra):
    if corr_type == 'pearson':
        def corr_fn(snaps):
            return np.corrcoef(snaps)
    elif corr_type == 'spearman':
        def corr_fn(snaps):
            return stats.spearmanr(snaps, axis=1)[0]
    else:
        raise ValueError(f'Unrecognized correlation type "{corr_type}"')
    
    return calc_mean_pairwise_repr_fn(repr_snaps, corr_fn, calc_all=False, include_individual=include_individual_corr_mats)
        

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
    did_dho = 'holdout_testing' in train_params and train_params['holdout_testing'] == 'domain'
    did_dho = did_dho or 'do_tree_holdout' in train_params and train_params['do_tree_holdout']
    if did_dho:
        net_params['n_train_domains'] = nd - 1
        for key, snap in snaps.items():
            snaps[key] = np.delete(snap, slice(snap.shape[2] // nd * (nd - 1), None), axis=2)
    else:
        net_params['n_train_domains'] = nd

    report_stats = {
        report_type: util.get_mean_and_ci(report)
        for report_type, report in reports.items()
    }
    
    report_means = {report_type: rstats[0] for report_type, rstats in report_stats.items()}
    report_cis = {report_type: rstats[1] for report_type, rstats in report_stats.items()}

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
        'net_params': net_params,
        'train_params': train_params,
        'report_epochs': report_epochs,
        **extra
    }


def plot_report(ax, res, report_type, with_ci=True, label=None, title=None, **plot_params):
    """Make a standard plot of mean loss, accuracy, etc. over training"""
    xaxis = res['etg_epochs'] if 'etg' in report_type else res['report_epochs']
    ax.plot(xaxis, res['reports'][report_type], label=label, **plot_params)
    if with_ci:
        ax.fill_between(xaxis, *res['report_cis'][report_type], alpha=0.3)
        
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
    
    title = f'Epoch {res["snap_epochs"][snap_ind]}'
    if title_addon is not None:
        title += f'\n({title_addon})'
    ax.set_title(title)

    return image
