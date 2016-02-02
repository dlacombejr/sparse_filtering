import numpy as np
from utilities.connections import distMat


def continuity(weights):

    """
    :param weights: nxd array where n = neurons and d = weights
    :return: continuity value based on simulated cortical sheet
    """

    # determine number of neurons
    neurons = weights.shape[0]

    # get distance matrix
    distances = distMat(neurons, d=neurons)

    # get similarity value between all pairwise neurons
    similarity = np.dot(weights.T, weights)

    # calculate continuity
    c = np.multiply(distances, similarity)

    return c
