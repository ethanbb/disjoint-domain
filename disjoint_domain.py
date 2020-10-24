import numpy as np
from scipy.linalg import block_diag
from scipy.cluster import hierarchy
from scipy.spatial import distance
import matplotlib.pyplot as plt

N_ITEMS = 8


def _make_attr_vecs(n_contexts, attrs_per_context, rng):
    """
    Make some attribute vectors that conform to the Euclidean distance plot (Figure R3, bottom).
    There are 8 items. Outputs a list of n_contexts 8 x attrs_per_context matrices.
    These attributes are simply repeated for each domain.

    Input attrs_per_context must be at least 50 (approximately) in order for the distances to be correct.
    Each vector has 25 attributes activated within the current context, so that cross-context distances are sqrt(50).
    """

    if rng is None:
        rng = np.random.default_rng()

    attrs = [np.zeros((N_ITEMS, attrs_per_context), dtype='int32') for _ in range(n_contexts)]

    for kC in range(n_contexts):
        # first, handle items 1-4, which all pairwise have distance 2, i.e. each pair differs by 4 bits.
        # choose centroid randomly
        circ_centroid = np.zeros(attrs_per_context, dtype='int32')
        circ_centroid_set = rng.choice(attrs_per_context, 25, replace=False)
        circ_centroid_unset = np.setdiff1d(range(attrs_per_context), circ_centroid_set)
        circ_centroid[circ_centroid_set] = 1

        # now choose 2 distinct bits to flip for each of the 4 vectors, keeping total # set = 25
        set_bits = rng.choice(circ_centroid_unset, 4, replace=False)
        unset_bits = rng.choice(circ_centroid_set, 4, replace=False)

        for k in range(4):
            attrs[kC][k, :] = circ_centroid
            attrs[kC][k, set_bits[k]] = 1
            attrs[kC][k, unset_bits[k]] = 0

        # pick centroid for other 4 items, which should be 40 bits away from this centroid.
        other_centroid = circ_centroid.copy()
        other_centroid[rng.choice(circ_centroid_unset, 20, replace=False)] = 1
        other_centroid[rng.choice(circ_centroid_set, 20, replace=False)] = 0

        # now square and star centroids, which are centered on other_centroid and differ by 12 bits
        set_bits = rng.choice(np.flatnonzero(other_centroid == 0), 6, replace=False)
        unset_bits = rng.choice(np.flatnonzero(other_centroid), 6, replace=False)

        square_centroid = other_centroid.copy()
        square_centroid[set_bits[:3]] = 1
        square_centroid[unset_bits[:3]] = 0

        star_centroid = other_centroid.copy()
        star_centroid[set_bits[3:]] = 1
        star_centroid[unset_bits[3:]] = 0

        # squares (vectors 5 and 6) differ by just 2 bits. be a little imprecise and let one of them be the centroid.
        attrs[kC][4:6, :] = square_centroid
        attrs[kC][5, rng.choice(np.flatnonzero(square_centroid))] = 0
        attrs[kC][5, rng.choice(np.flatnonzero(square_centroid == 0))] = 1

        # stars differ by 10 bits. again be a little imprecise, let one differ from centroid by 4 and the other by 6 (all unique)
        set_bits = rng.choice(np.flatnonzero(star_centroid == 0), 5, replace=False)
        unset_bits = rng.choice(np.flatnonzero(star_centroid), 5, replace=False)

        attrs[kC][6, :] = star_centroid
        attrs[kC][6, set_bits[:2]] = 1
        attrs[kC][6, unset_bits[:2]] = 0

        attrs[kC][7, :] = star_centroid
        attrs[kC][7, set_bits[2:]] = 1
        attrs[kC][7, unset_bits[2:]] = 0

    return attrs


def make_io_mats(n_contexts=4, attrs_per_context=50, rng=None, n_domains=4):
    """Make the actual item, context, and attribute matrices, across a given number of domains."""

    # First make it for a single domain, then use block_diag to replicate.
    item_mat_1 = np.tile(np.eye(N_ITEMS), (n_contexts, 1))
    context_mat_1 = np.repeat(np.eye(n_contexts), N_ITEMS, axis=0)
    attrs = _make_attr_vecs(n_contexts, attrs_per_context, rng)
    attr_mat_1 = block_diag(*attrs)

    item_mat = block_diag(*[item_mat_1 for _ in range(n_domains)])
    context_mat = block_diag(*[context_mat_1 for _ in range(n_domains)])
    attr_mat = block_diag(*[attr_mat_1 for _ in range(n_domains)])

    return item_mat, context_mat, attr_mat


def plot_item_attributes(n_contexts=4, attrs_per_context=50, rng=None):
    """Item and context inputs and attribute outputs for each input combination (regardless of domain)"""

    item_mat, context_mat, attr_mat = make_io_mats(n_contexts, attrs_per_context, rng, n_domains=1)

    fig = plt.figure()

    # plot 1: items
    ax = fig.add_subplot(1, 20, (1, 3))
    ax.set_title('Items')
    ax.imshow(item_mat, aspect='auto', interpolation='nearest')
    ax.set_yticks([])
    ax.set_xticks(range(0, N_ITEMS, 2))

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


def plot_item_attribute_dendroram(n_contexts=4, attrs_per_context=50, rng=None):
    """Dendrogram of similarities between the items' attributes, collapsed across contexts"""

    attrs = _make_attr_vecs(n_contexts, attrs_per_context, rng)

    fig, ax = plt.subplots()
    mean_dist = np.mean(np.stack([distance.pdist(a) for a in attrs]), axis=0)
    z = hierarchy.linkage(mean_dist)
    hierarchy.dendrogram(z, ax=ax)
    ax.set_title('Item attribute similarities, collapsed across contexts')
    ax.set_ylabel('Euclidean distance')
    ax.set_xlabel('Input #')
