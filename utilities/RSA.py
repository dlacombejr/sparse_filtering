import numpy as np
import pylab as pl
from sklearn.manifold import MDS
from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.stats import pearsonr, spearmanr
from matplotlib.collections import LineCollection


def rsa(representations):

    """
    :param representations: n x d array of representations where n = representational units (e.g., neurons) and
        d = representations (e.g., activations for all images or regression coefficients for each object category)
    :return:
        rdm - n x n array of representational dissimilarity between all pairwise comparisons of representational units
        NOTE: representational similarity can be computed with either pearson or spearman correlation coefficient
    """

    # define units of comparison
    neurons = representations.shape[0]

    # initialize empty RDM
    rdm = np.zeros((neurons, neurons))

    # loop over each pairwise comparison and calculate representational dissimilarity
    for i in xrange(neurons):
        for j in xrange(neurons):
            rdm[i, j] = 1 - spearmanr(representations[i, :].T, representations[j, :].T)[0]

    # visualize the rdm
    pl.imshow(rdm, interpolation='nearest')
    pl.show()

    return rdm


def multidimensional_scaling(rdm, labels):

    # perform multidimensional scaling
    mds = MDS(
        n_components=2,
        max_iter=3000,
        dissimilarity='precomputed'
    )

    positions = mds.fit(rdm).embedding_
    positions /= positions.max()

    # visualize the embedding in a figure
    figure = plt.figure(1)
    ax = plt.axes([0., 0., 1., 1.])

    plt.scatter(positions[:, 0], positions[:, 1])

    # plot the edges
    segments = [[positions[i, :], positions[j, :]] for i in range(len(positions)) for j in range(len(positions))]
    values = np.abs(rdm)
    lc = LineCollection(
        segments,
        zorder=0,
        cmap=plt.cm.YlGnBu,
        norm=plt.Normalize(0, values.max())
    )
    lc.set_array(rdm.flatten())
    lc.set_linewidths(2 * np.ones(len(segments)))
    ax.add_collection(lc)

    # add labels
    for index, label in enumerate(labels):
        plt.annotate(label, (positions[index, 0], positions[index, 1]))

    plt.show()


def agglomerative_clustering(rdm, labels=None):

    # compute and plot first dendrogram
    fig = pl.figure(figsize=(8, 8))
    # ax1 = fig.add_axes([0.09, 0.1, 0.2, 0.6])
    # y = sch.linkage(rdm, method='single')  # centroid
    # z1 = sch.dendrogram(y, orientation='right')
    # ax1.set_xticks([])
    # ax1.set_yticks([])

    # compute and plot second dendrogram
    ax2 = fig.add_axes([0.3, 0.71, 0.6, 0.2])
    y = sch.linkage(rdm, method='average')  # centroid
    z1 = sch.dendrogram(y)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # plot distance matrix
    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.6])
    idx1 = z1['leaves']
    idx2 = idx1  # list(reversed(idx1))
    rdm = rdm[idx1, :]
    rdm = rdm[:, idx2]
    im = axmatrix.matshow(rdm, aspect='auto', origin='lower', cmap=pl.cm.YlGnBu)
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    labels_1 = idx1
    labels_2 = idx2
    if labels is not None:
        labels_1 = labels[idx1]
        labels_2 = labels[idx2]

    axmatrix.set_xticks(range(rdm.shape[0]))
    axmatrix.set_xticklabels(labels_1, minor=False)
    axmatrix.xaxis.set_label_position('bottom')
    axmatrix.xaxis.tick_bottom()

    pl.xticks(rotation=90, fontsize=8)

    axmatrix.set_yticks(range(rdm.shape[0]))
    axmatrix.set_yticklabels(labels_2, minor=False)
    axmatrix.yaxis.set_label_position('left')
    axmatrix.yaxis.tick_left()

    pl.yticks(fontsize=8)

    # plot colorbar
    axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.6])
    pl.colorbar(im, cax=axcolor)
    im.set_clim(vmin=0, vmax=1)
    fig.show()
    fig.savefig('dendrogram.png')


# # example test
# classes = 10
# examples = 100
# x = np.random.rand(classes, examples)

# create set of words for CIFAR-10 categories
categories = set()
categories.add('airplane')
categories.add('automobile')
categories.add('bird')
categories.add('cat')
categories.add('deer')
categories.add('dog')
categories.add('frog')
categories.add('horse')
categories.add('ship')
categories.add('truck')

#### create a similarity matrix
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
brown_ic = wordnet_ic.ic('ic-brown.dat')

# empty similarity matix
N = len(categories)
similarity_matrix = np.zeros((N, N))

# initialize counters
x_index = 0
y_index = 0
# loop over all pairwise comparisons
for category_x in categories:
    for category_y in categories:
        print category_x, category_y
        x = wn.synset(str(category_x) + str('.n.01'))
        y = wn.synset(str(category_y) + str('.n.01'))
        # enter similarity value into the matrix
        similarity_matrix[x_index, y_index] = x.path_similarity(y)  # x.jcn_similarity(y, brown_ic)
        # iterate x counter
        x_index += 1
    # reinitialize x counter and iterate y counter
    x_index = 0
    y_index += 1

# # get rdm
# rdm = rsa(x)
rdm = 1 - similarity_matrix

# multidimensional scaling
# categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
categories = ['horse', 'automobile', 'deer', 'dog', 'frog', 'cat', 'truck', 'ship', 'airplane', 'bird']
multidimensional_scaling(rdm, categories)

# agglomerative clustering
categories = np.asarray(categories)
agglomerative_clustering(rdm, labels=categories)
