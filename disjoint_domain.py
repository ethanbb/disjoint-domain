import numpy as np
from scipy.linalg import block_diag
from scipy.cluster import hierarchy
from scipy.spatial import distance
import matplotlib.pyplot as plt
import torch

ITEMS_PER_DOMAIN = 8


def choose_k_inds(n, k=1):
    """Get random permutation of k indices in range(n)"""
    return torch.randperm(n, device='cpu')[:k]


def choose_k(a, k=1):
    """
    Get permutation of k items from a using PyTorch
    (want to make sure we use the same rng for everything so it's deterministic given the seed)
    """
    return a[choose_k_inds(len(a), k)]


def choose_k_set_bits(a, k=1):
    """Get permutation of k indices of a that are set (a should be a boolean array)"""
    return choose_k(np.flatnonzero(a), k)


def _make_4_similar_attr_vecs(centroid):
    """Useful for making 'circles,' and reused if simplified == True"""
    set_bits = choose_k_set_bits(1 - centroid, 4)
    unset_bits = choose_k_set_bits(centroid, 4)

    similar_vecs = np.tile(centroid.copy(), (4, 1))
    for k in range(4):
        similar_vecs[k, set_bits[k]] = 1
        similar_vecs[k, unset_bits[k]] = 0

    return similar_vecs


def _make_attr_vecs(ctx_per_domain, attrs_per_context, simplified=False):
    """
    Make some attribute vectors that conform to the Euclidean distance plot (Figure R3, bottom).
    There are 8 items. Outputs a list of ctx_per_domain 8 x attrs_per_context matrices.
    These attributes are simply repeated for each domain.

    Input attrs_per_context must be at least 50 (approximately) in order for the distances to be correct.
    Each vector has 25 attributes activated within the current context, so that cross-context distances are sqrt(50).
    
    Simplified flag: if true, makes 4 "circles" and 4 "squares," where the similarity among squares is
    the same as similarity among circles. This is a symmetric case to test whether the network is picking
    up some higher-level property of the circles ("typicality") or just using the representation layer to make
    the most useful split of the items in each domain.
    """

    attrs = [np.zeros((ITEMS_PER_DOMAIN, attrs_per_context)) for _ in range(ctx_per_domain)]

    for kC in range(ctx_per_domain):
        # first, handle items 1-4, which all pairwise have distance 2, i.e. each pair differs by 4 bits.
        # choose centroid randomly
        circ_centroid = np.zeros(attrs_per_context)
        circ_centroid_set = choose_k(np.arange(attrs_per_context), 25)
        circ_centroid_unset = np.setdiff1d(range(attrs_per_context), circ_centroid_set)
        circ_centroid[circ_centroid_set] = 1

        # now choose 2 distinct bits to flip for each of the 4 vectors, keeping total # set = 25
        attrs[kC][:4] = _make_4_similar_attr_vecs(circ_centroid)

        # pick centroid for other 4 items, which should be 40 bits away from this centroid.
        other_centroid = circ_centroid.copy()
        other_centroid[choose_k(circ_centroid_unset, 20)] = 1
        other_centroid[choose_k(circ_centroid_set, 20)] = 0
        
        if simplified:
            # proceed as in the circle centroid to make 4 "squares"
            attrs[kC][4:] = _make_4_similar_attr_vecs(other_centroid)
            continue

        # now square and star centroids, which are centered on other_centroid and differ by 12 bits
        set_bits = choose_k_set_bits(1-other_centroid, 6)
        unset_bits = choose_k_set_bits(other_centroid, 6)

        square_centroid = other_centroid.copy()
        square_centroid[set_bits[:3]] = 1
        square_centroid[unset_bits[:3]] = 0

        star_centroid = other_centroid.copy()
        star_centroid[set_bits[3:]] = 1
        star_centroid[unset_bits[3:]] = 0

        # squares (vectors 5 and 6) differ by just 2 bits. be a little imprecise and let one of them be the centroid.
        attrs[kC][4:6, :] = square_centroid
        attrs[kC][5, choose_k_set_bits(square_centroid)] = 0
        attrs[kC][5, choose_k_set_bits(1-square_centroid)] = 1

        # stars differ by 10 bits. again be a little imprecise, let one differ from centroid by 4 and the other by 6 (all unique)
        set_bits = choose_k_set_bits(1-star_centroid, 5)
        unset_bits = choose_k_set_bits(star_centroid, 5)

        attrs[kC][6, :] = star_centroid
        attrs[kC][6, set_bits[:2]] = 1
        attrs[kC][6, unset_bits[:2]] = 0

        attrs[kC][7, :] = star_centroid
        attrs[kC][7, set_bits[2:]] = 1
        attrs[kC][7, unset_bits[2:]] = 0

    return attrs


def make_io_mats(ctx_per_domain=4, attrs_per_context=50, n_domains=4, simplified=False):
    """Make the actual item, context, and attribute matrices, across a given number of domains."""

    # First make it for a single domain, then use block_diag to replicate.
    item_mat_1 = np.tile(np.eye(ITEMS_PER_DOMAIN), (ctx_per_domain, 1))
    context_mat_1 = np.repeat(np.eye(ctx_per_domain), ITEMS_PER_DOMAIN, axis=0)
    attrs = _make_attr_vecs(ctx_per_domain, attrs_per_context, simplified=simplified)
    attr_mat_1 = block_diag(*attrs)

    item_mat = block_diag(*[item_mat_1 for _ in range(n_domains)])
    context_mat = block_diag(*[context_mat_1 for _ in range(n_domains)])
    attr_mat = block_diag(*[attr_mat_1 for _ in range(n_domains)])

    return item_mat, context_mat, attr_mat


def plot_item_attributes(ctx_per_domain=4, attrs_per_context=50, simplified=False):
    """Item and context inputs and attribute outputs for each input combination (regardless of domain)"""

    item_mat, context_mat, attr_mat = make_io_mats(ctx_per_domain, attrs_per_context, n_domains=1, simplified=simplified)

    fig = plt.figure()

    # plot 1: items
    ax = fig.add_subplot(1, 20, (1, 3))
    ax.set_title('Items')
    ax.imshow(item_mat, aspect='auto', interpolation='nearest')
    ax.set_yticks([])
    ax.set_xticks(range(0, ITEMS_PER_DOMAIN, 2))

    # plot 2: contexts
    ax = fig.add_subplot(1, 20, 4)
    ax.set_title('Contexts')
    ax.imshow(context_mat, aspect='auto', interpolation='nearest')
    ax.set_yticks([])
    ax.set_xticks(range(0, context_mat.shape[1], 2))

    # plot 3: attributes
    ax = fig.add_subplot(1, 20, (5, 20))
    ax.set_title('Attributes')
    ax.imshow(attr_mat, aspect='auto', interpolation='nearest')
    ax.set_yticks([])
    ax.set_xticks(range(0, attr_mat.shape[1], 10))


def plot_item_attribute_dendrogram(ctx_per_domain=4, attrs_per_context=50, simplified=False):
    """Dendrogram of similarities between the items' attributes, collapsed across contexts"""

    attrs = _make_attr_vecs(ctx_per_domain, attrs_per_context, simplified=simplified)

    fig, ax = plt.subplots()
    mean_dist = np.mean(np.stack([distance.pdist(a) for a in attrs]), axis=0)
    z = hierarchy.linkage(mean_dist)
    hierarchy.dendrogram(z, ax=ax)
    ax.set_title('Item attribute similarities, collapsed across contexts')
    ax.set_ylabel('Euclidean distance')
    ax.set_xlabel('Input #')


def init_torch(device=None, torchfp=None, use_cuda_if_possible=True):
    """Establish floating-point type and device to use with PyTorch"""

    if device is None:
        if use_cuda_if_possible and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

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


def item_group(n, simplified=False, **_extra):
    """Equivalent of circle/square/star, to identify 'types' of items"""
    group = n // 4
    if not simplified:
        group += n // 6
    return group


def item_group_symbol(n, simplified=False):
    return ['*', '@', '$'][item_group(n, simplified)]
    

def domain_name(n):
    return chr(ord('A') + n)


def get_items(n_domains=4, simplified=False, **_extra):
    """Get item tensors (without repetitions) and their corresponding names"""
    items = torch.eye(ITEMS_PER_DOMAIN * n_domains)
    item_names = [domain_name(d) + str(n + 1) + item_group_symbol(n, simplified)
                  for d in range(n_domains) for n in range(ITEMS_PER_DOMAIN)]
    return items, item_names


def get_contexts(n_domains=4, ctx_per_domain=4, **_extra):
    """Get context tensors (without repetitions) and their corresponding names"""
    contexts = torch.eye(ctx_per_domain * n_domains)
    context_names = [domain_name(d) + str(n+1) for d in range(n_domains) for n in range(ctx_per_domain)]
    return contexts, context_names


def get_net_dims(n_domains=4, ctx_per_domain=4, attrs_per_context=50, **_extra):
    """Get some basic facts about the default architecture, possibly with overrides"""
    n_items = ITEMS_PER_DOMAIN * n_domains
    n_ctx = ctx_per_domain * n_domains

    return ctx_per_domain, n_domains, n_items, n_ctx, attrs_per_context


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
