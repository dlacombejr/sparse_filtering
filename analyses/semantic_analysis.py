import os
import numpy as np
import pylab as pl
from scipy.io import loadmat, savemat


def main():

    # load in activation data
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "saved", "activations_concatenated.mat")
    data = loadmat(file_path)['master']  # [examples, neurons, image-space]

    # load in data labels
    file_path = os.path.join(base_path, "data", "CIFAR_data.mat")
    train_labels = loadmat(file_path)['y']

    # define important variables
    classes = np.unique(train_labels)
    dimension = np.sqrt(data.shape[1])

    # create class-specific activation matrices
    class_specific_activations = np.zeros((classes, dimension, dimension))

    # fill activation matrix by looping over classes
    for label in xrange(len(classes)):
        class_data = data[train_labels == label, :, :]
        class_data = np.sum(class_data, axis=(0, 2))
        class_specific_activations[label, :, :] = \
            class_specific_activations[label, :, :] + class_data.reshape((dimension, dimension))

    # plot all of the class-specific matrices
    for label in xrange(len(classes)):
        pl.subplot(2, 5, label + 1)
        pl.imshow(
            class_specific_activations[label, :, :],
            interpolation='nearest',
            cmap=pl.cm.gray
        )
        pl.xticks([])
        pl.yticks([])

    pl.show()

if __name__ == '__main__':
    main()