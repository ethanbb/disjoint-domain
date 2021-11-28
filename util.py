import torch
import numpy as np
from scipy import stats
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
import seaborn as sns


def init_torch(device=None, torchfp=None, use_cuda_if_possible=True):
    """Establish floating-point type and device to use with Pytorch"""

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

    def zeros_fn(size):
        return torch.zeros(size, device=device)

    return device, torchfp, zeros_fn


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


def permute(a):
    return choose_k(a, len(a))


def choose_k_set_bits(a, k=1):
    """Get permutation of k indices of a that are set (a should be a boolean array)"""
    return choose_k(np.flatnonzero(a), k)


def get_mean_and_ci(series_set):
    """
    Given a set of N time series, compute and return the mean
    along with 95% confidence interval using a t-distribution.
    """
    n = series_set.shape[0]
    mean = np.mean(series_set, axis=0)
    stderr = np.std(series_set, axis=0) / np.sqrt(n)
    interval = np.stack([
        stats.t.interval(0.95, df=n-1, loc=m, scale=std) if std > 0 else (m, m)
        for (m, std) in zip(mean, stderr)
    ], axis=1)
    
    return mean, interval


def calc_snap_epochs(snap_freq, num_epochs, snap_freq_scale='lin'):
    """Given the possibility of taking snapshots on a log scale, get the actual snapshot epochs"""
    if snap_freq_scale == 'log':
        snap_epochs = np.arange(0, np.log2(num_epochs), snap_freq)
        snap_epochs = np.exp2(snap_epochs)
    elif snap_freq_scale == 'lin':
        snap_epochs = np.arange(0, num_epochs, snap_freq)
    else:
        raise ValueError(f'Unkonwn snap_freq_scale {snap_freq_scale}')

    return list(np.unique(np.round(snap_epochs).astype(int)))


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


def imshow_pos(ax, mat, max_val=None, **imshow_args):
    """Plot an image/matrix with only nonnegative values using a perceptually uniform color palette"""
    if np.any(mat < 0):
        raise ValueError('Matrix must be nonnegative to use imshow_pos')
    
    cmap = sns.color_palette('rocket', as_cmap=True)
    if max_val is None:
        max_val = np.max(mat)
    return ax.imshow(mat, cmap=cmap, vmin=0, vmax=max_val, interpolation='nearest', **imshow_args)


def imshow_centered_bipolar(ax, mat, max_absval=None, **imshow_args):
    """Plot an image/matrix with a good diverging color palette, with the neutral color at 0"""
    cmap = sns.color_palette('icefire', as_cmap=True)
    if max_absval is None:
        max_absval = np.max(np.abs(mat))
    return ax.imshow(mat, cmap=cmap, vmin=-max_absval,
                     vmax=max_absval, interpolation='nearest', **imshow_args)


def plot_matrix_with_labels(ax, mat, labels, colorbar=True, label_cols=True,
                            tick_fontsize='medium', bipolar=True, max_val=None, **imshow_args):
    """Helper to plot matrix with each row labeled with 'labels' and also each column if desired."""       
    n = len(labels)
    assert n == mat.shape[0], 'Wrong number of labels'
    if label_cols:
        assert mat.shape[0] == mat.shape[1], 'Matrix must be square'
    
    plot_fn = imshow_centered_bipolar if bipolar else imshow_pos
    image = plot_fn(ax, mat, max_val, **imshow_args)

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    
    if label_cols:
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels)
    
    if colorbar:
        add_colorbar(image)
        
    for ticklabel in ax.get_xticklabels() + ax.get_yticklabels():
        ticklabel.set_fontsize(tick_fontsize)
    
    return image


def get_attribute_rdm(attrs, metric='cityblock'):
    if isinstance(attrs, np.ndarray):
        attrs = [attrs]
    mean_dist = np.nanmean(np.stack([distance.pdist(a, metric=metric) for a in attrs]), axis=0)
    return distance.squareform(mean_dist)
