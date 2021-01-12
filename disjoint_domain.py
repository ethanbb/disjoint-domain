import numpy as np
from scipy.linalg import block_diag
from scipy.cluster import hierarchy
from scipy.spatial import distance
import matplotlib.pyplot as plt
import torch

ITEMS_PER_DOMAIN = 8
ATTRS_SET_PER_ITEM = 25


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


def choose_k_set_bits(a, k=1):
    """Get permutation of k indices of a that are set (a should be a boolean array)"""
    return choose_k(np.flatnonzero(a), k)


def get_cluster_sizes(clusters):
    """Parse a 'clusters' string such as '4-2-2_suffix' to get the size of each cluster"""
    sizes = [int(sz) for sz in clusters.partition('_')[0].split('-')]
    if sum(sizes) != ITEMS_PER_DOMAIN:
        raise ValueError('Invalid clusters specification')
    
    return sizes


def _make_n_dist_d_attr_vecs(centroid, n=4, d=4):
    """
    Useful for making 'circles' and similar structures in other attr cluster settings
    Here d is the *Hamming* distance between vectors, which is the square of Euclidean distance
    (and each vector is half the Hamming distance from the centroid as to each other vector)
    """
    try:
        all_set_bits = choose_k_set_bits(1 - centroid, n * d // 4)
        all_unset_bits = choose_k_set_bits(centroid, n * d // 4)
    except ValueError:
        raise ValueError(f'Not enough attributes to make {n} vectors of dist {d} from each other')
        
    similar_vecs = np.tile(centroid.copy(), (n, 1))
    
    for similar_vec, set_bits, unset_bits in zip(similar_vecs, np.split(all_set_bits, n),
                                                 np.split(all_unset_bits, n)):
        similar_vec[set_bits] = 1
        similar_vec[unset_bits] = 0

    return similar_vecs


def _make_3_group_attr_vecs(ctx_per_domain, attrs_per_context, clusters='4-2-2', **_extra):
    """
    Make some attribute vectors that conform to the Euclidean distance plot (Figure R3, bottom).
    There are 8 items. Outputs a list of ctx_per_domain 8 x attrs_per_context matrices.
    These attributes are simply repeated for each domain.

    attrs_per_context must be at least 50 (approximately) in order for the distances to be correct.
    Each vector has 25 attributes activated within the current context, so that cross-context distances are sqrt(50).
    
    clust_sizes: 3-item list of # of circles, squares, and stars. Currently # of stars must be 2.
    """
    if attrs_per_context < 50:
        raise ValueError('Need >= 50 attrs for standard attribute vecs')
    
    clust_sizes = get_cluster_sizes(clusters)
    if len(clust_sizes) != 3 or clust_sizes[2] != 2:
        raise ValueError('Invalid clust_sizes')
    n_circles, n_squares, _ = clust_sizes

    attrs = [np.empty((ITEMS_PER_DOMAIN, attrs_per_context)) for _ in range(ctx_per_domain)]

    for attr_mat in attrs:
        # first, handle circles, which all pairwise have distance 2, i.e. each pair differs by 4 bits.
        # choose centroid randomly
        circ_centroid = np.zeros(attrs_per_context)
        circ_centroid_set = choose_k(np.arange(attrs_per_context), ATTRS_SET_PER_ITEM)
        circ_centroid_unset = np.setdiff1d(range(attrs_per_context), circ_centroid_set, assume_unique=True)
        circ_centroid[circ_centroid_set] = 1

        # now choose 2 distinct bits to flip for each of the 'circle' vectors, keeping total # set = 25
        attr_mat[:n_circles] = _make_n_dist_d_attr_vecs(circ_centroid, n_circles, 4)

        # pick centroid for other items, which should be 40 bits away from this centroid.
        other_centroid = circ_centroid.copy()
        other_centroid[choose_k(circ_centroid_unset, 20)] = 1
        other_centroid[choose_k(circ_centroid_set, 20)] = 0

        # now square and star centroids, which are centered on other_centroid and differ by 12 bits
        square_centroid, star_centroid = _make_n_dist_d_attr_vecs(other_centroid, 2, 12)

        # squares differ by just 2 bits. be a little imprecise and let one of them be the centroid.
        attr_mat[n_circles:6, :] = square_centroid
        sq_set_bits = choose_k_set_bits(1-square_centroid, n_squares-1)
        sq_unset_bits = choose_k_set_bits(square_centroid, n_squares-1)
        for attr_vec, set_bit, unset_bit in zip(attr_mat[n_circles+1:6], sq_set_bits, sq_unset_bits):
            attr_vec[set_bit] = 1
            attr_vec[unset_bit] = 0

        # stars differ by 10 bits. again be a little imprecise, let one differ from centroid by 4 and the other by 6 (all unique)
        set_bits = choose_k_set_bits(1-star_centroid, 5)
        unset_bits = choose_k_set_bits(star_centroid, 5)

        attr_mat[6, :] = star_centroid
        attr_mat[6, set_bits[:2]] = 1
        attr_mat[6, unset_bits[:2]] = 0

        attr_mat[7, :] = star_centroid
        attr_mat[7, set_bits[2:]] = 1
        attr_mat[7, unset_bits[2:]] = 0

    return attrs


def _make_2_group_attr_vecs(ctx_per_domain, attrs_per_context, clusters='4-4',
                            intragroup_dists=[4, 12], intergroup_dist=40):
    """
    Make attribute vectors with 2 clusters in a systematic way. All distances are Hamming and
    should be divisible by 4 (2 in the case of intergroup)
    """
    clust_sizes = get_cluster_sizes(clusters)
    max_disjoint_bits = max(ATTRS_SET_PER_ITEM, attrs_per_context - ATTRS_SET_PER_ITEM)
    
    # Validate everything first
    if intergroup_dist % 2 != 0 or intergroup_dist // 2 > max_disjoint_bits:
        raise ValueError(f'Invalid intergroup distance - must be even and <= {max_disjoint_bits * 2}')
    
    if any([dist % 4 != 0 for dist in intragroup_dists]):
        raise ValueError('Invalid intragroup distances - must all be multiples of 4')
    if any([dist // 4 * n > max_disjoint_bits for dist, n in zip(intragroup_dists, clust_sizes)]):
        raise ValueError('Not enough attributes per cluster for these sizes and intragroup distances')
    
    attrs = [np.empty((ITEMS_PER_DOMAIN, attrs_per_context)) for _ in range(ctx_per_domain)]
        
    for attr_mat in attrs:
        circ_centroid = np.zeros(attrs_per_context)
        circ_centroid_set = choose_k(np.arange(attrs_per_context), ATTRS_SET_PER_ITEM)
        circ_centroid_unset = np.setdiff1d(range(attrs_per_context), circ_centroid_set, assume_unique=True)
        circ_centroid[circ_centroid_set] = 1
        
        # make cirlces
        n_circles, n_squares = clust_sizes
        circ_dists, square_dists = intragroup_dists
        attr_mat[:n_circles] = _make_n_dist_d_attr_vecs(circ_centroid, n_circles, circ_dists)
        
        # make square centroid
        square_centroid = circ_centroid.copy()
        square_centroid[choose_k(circ_centroid_unset, intergroup_dist // 2)] = 1
        square_centroid[choose_k(circ_centroid_set, intergroup_dist // 2)] = 0
        
        # make squares
        attr_mat[n_circles:] = _make_n_dist_d_attr_vecs(square_centroid, n_squares, square_dists)
        
    return attrs
    

def _make_equidistant_attr_vecs(ctx_per_domain, attrs_per_context, intragroup_dists=[10], **_extra):
    """
    Make attribute vectors that are all equidistant from each other with Hamming distance `dist`. 
    attrs_per_context must be at least ATTRS_SET_PER_ITEM + (dist/2) * (ITEMS_PER_DOMAIN-1)
    (by default, at least 60). Also, dist must be even.
    """
    dist = intragroup_dists[0]
    if dist % 2 != 0:
        raise ValueError('dist must be even')
    half_dist = dist // 2
    
    if attrs_per_context < ATTRS_SET_PER_ITEM + half_dist * (ITEMS_PER_DOMAIN-1):
        raise ValueError(f'Need more attrs to get equidistant vecs with distance {dist}')
    
    n_rot = half_dist * ITEMS_PER_DOMAIN  # portion of vector that rotates for each item
    n_fixed_set = ATTRS_SET_PER_ITEM - half_dist

    attrs = [np.zeros((ITEMS_PER_DOMAIN, attrs_per_context)) for _ in range(ctx_per_domain)]
    
    for attr_mat in attrs:
        # pick fixed set and rotating indices
        fixed_set_and_rot_inds = choose_k_inds(attrs_per_context, n_fixed_set + n_rot)
        fixed_set_inds, rot_inds = fixed_set_and_rot_inds.split([n_fixed_set, n_rot])
        rot_inds_each = rot_inds.split(half_dist)
        
        # set fixed indices for all items and rotating indices for each individually
        attr_mat[:, fixed_set_inds] = 1
        for attr_vec, inds in zip(attr_mat, rot_inds_each):
            attr_vec[inds] = 1
    
    return attrs
    
    
def normalize_cluster_info(cluster_info):
    """
    Get a dict that specifies information about the attribute clusters.
    Input is either a string (equivalent to {'clusters': cluster_info}) or
    a dict with keys 'clusters', 'intragroup_dists', and 'intergroup_dist'.
    The latter two keys are optional and default to None.
    """
    if isinstance(cluster_info, str):
        return {'clusters': cluster_info}
    return cluster_info
    

def make_attr_vecs(ctx_per_domain, attrs_per_context, cluster_info):
    """Wrapper to make any set of attr vectors (returns a list of one matrix per context)"""
    cluster_info = normalize_cluster_info(cluster_info)
    n_clusts = len(get_cluster_sizes(cluster_info['clusters']))
    
    try:
        attr_vec_fn = {
            1: _make_equidistant_attr_vecs,
            2: _make_2_group_attr_vecs,
            3: _make_3_group_attr_vecs
        }[n_clusts]
    except KeyError:
        raise ValueError('Invalid clusters specification')
        
    return attr_vec_fn(ctx_per_domain, attrs_per_context, **cluster_info)

    
def make_io_mats(ctx_per_domain=4, attrs_per_context=50, n_domains=4, 
                 cluster_info='4-2-2', last_domain_cluster_info=None):
    """
    Make the actual item, context, and attribute matrices, across a given number of domains.
    If one_equidistant is true, replaces the last domain's attrs with equidistant attr vectors.
    Cluster_info and last_domain_cluster_info should be valid inputs to normalize_cluster_info.
    By default (when None), last_domain_clusters is the same as clusters.
    """

    # First make it for a single domain, then use block_diag to replicate.
    item_mat_1 = np.tile(np.eye(ITEMS_PER_DOMAIN), (ctx_per_domain, 1))
    item_mat = block_diag(*[item_mat_1 for _ in range(n_domains)])
    
    context_mat_1 = np.repeat(np.eye(ctx_per_domain), ITEMS_PER_DOMAIN, axis=0)
    context_mat = block_diag(*[context_mat_1 for _ in range(n_domains)])
    
    # New behavior: generate a new set of attr vecs for each domain.
    if last_domain_cluster_info is None:
        last_domain_cluster_info = cluster_info
    
    cluster_info = normalize_cluster_info(cluster_info)
    last_domain_cluster_info = normalize_cluster_info(last_domain_cluster_info)
        
    domain_attrs = [make_attr_vecs(ctx_per_domain, attrs_per_context, cluster_info)
                    for _ in range(n_domains - 1)]
    domain_attrs.append(make_attr_vecs(ctx_per_domain, attrs_per_context, last_domain_cluster_info))
    attr_mat = block_diag(*[block_diag(*attrs) for attrs in domain_attrs])

    return item_mat, context_mat, attr_mat


def plot_item_attributes(ctx_per_domain=4, attrs_per_context=50, cluster_info='4-2-2'):
    """Item and context inputs and attribute outputs for each input combination (regardless of domain)"""

    item_mat, context_mat, attr_mat = make_io_mats(ctx_per_domain, attrs_per_context, n_domains=1,
                                                   cluster_info=cluster_info)

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


def get_item_attribute_rdm(ctx_per_domain=4, attrs_per_context=50, cluster_info='4-2-2'):
    """Make RDM of similarities between the items' attributes, collapsed across contexts"""
    
    attrs = make_attr_vecs(ctx_per_domain, attrs_per_context, cluster_info)
    mean_dist = np.mean(np.stack([distance.pdist(a) for a in attrs]), axis=0)
    return distance.squareform(mean_dist)

    
def plot_item_attribute_dendrogram(ctx_per_domain=4, attrs_per_context=50, cluster_info='4-2-2'):
    """Dendrogram of similarities between the items' attributes, collapsed across contexts"""

    dist_mat = get_item_attribute_rdm(ctx_per_domain, attrs_per_context, cluster_info)
    condensed_dist = distance.squareform(dist_mat)
    
    fig, ax = plt.subplots()
    z = hierarchy.linkage(condensed_dist)
    with plt.rc_context({'lines.linewidth': 2.5}):
        hierarchy.dendrogram(z, ax=ax, orientation='right', color_threshold=0.6*max(z[:, 2]),
                             distance_sort='ascending')
    ax.set_title('Item attribute similarities, collapsed across contexts')
    ax.set_xlabel('Euclidean distance')
    ax.set_ylabel('Input #')
    return fig, ax


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


def item_group(n=slice(None), clusters='4-2-2', **_extra):
    """Equivalent of circle/square/star, to identify 'types' of items"""
    if isinstance(clusters, dict):
        clusters = clusters['clusters']
    cluster_sizes = get_cluster_sizes(clusters)
    groups = [i for i, ksize in enumerate(cluster_sizes) for _ in range(ksize)]
    return groups[n]


def item_group_symbol(n, clusters='4-2-2'):
    if isinstance(clusters, dict):
        clusters = clusters['clusters']
    if len(get_cluster_sizes(clusters)) == 1:
        return ''
    symbol_array = np.array(['*', '@', '$'])
    return symbol_array[item_group(n, clusters)]
    

def domain_name(n):
    return chr(ord('A') + n)


def get_domain_colors():
    """For plotting MDS trajectories. From 'saltwater taffy' palette in Illustrator."""
    return [
        '#aadff1',
        '#fbb57f',
        '#f59db2',
        '#80caa5',
        '#fbed91'
    ]


def get_items(n_domains=4, cluster_info='4-2-2', last_domain_cluster_info=None, **_extra):
    """Get item tensors (without repetitions) and their corresponding names"""
    cluster_info = normalize_cluster_info(cluster_info)
    last_domain_cluster_info = (cluster_info if last_domain_cluster_info is None
                                else normalize_cluster_info(last_domain_cluster_info))
    
    items = torch.eye(ITEMS_PER_DOMAIN * n_domains)
    all_clusters = [cluster_info['clusters']] * n_domains
    if last_domain_cluster_info is not None:
        last_domain_cluster_info = normalize_cluster_info(last_domain_cluster_info)
        all_clusters[-1] = last_domain_cluster_info['clusters']
        
    item_names = [domain_name(d) + str(n + 1) + item_group_symbol(n, clst)
                  for d, clst in enumerate(all_clusters)
                  for n in range(ITEMS_PER_DOMAIN)]
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
