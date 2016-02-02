import os
import h5py
import numpy as np
import pylab as pl
from scipy.io import loadmat, savemat


def main():

    # get the folders in "saved" and select most recent
    base_path = os.path.dirname(__file__)
    folder_path = os.path.join(base_path, "saved")
    folders = os.listdir(folder_path)
    folder = folders[3]  # select most recent model folder  # -1

    # load in activation data
    file_path = os.path.join(folder_path, folder, "concatenated_activations.mat")
    # data = loadmat(file_path)['master']  # [examples, neurons, image-space]
    data = h5py.File(file_path, 'r')['master']
    data = np.array(data)
    data = data.T
    print data.shape

    # load in data labels
    file_path = os.path.join(base_path, "data", "CIFAR_data.mat")
    train_labels = loadmat(file_path)['y']

    # define important variables
    classes = len(np.unique(train_labels))
    dimension = np.sqrt(data.shape[1])

    # create class-specific activation matrices
    class_specific_activations = np.zeros((classes, dimension, dimension))

    # fill activation matrix by looping over classes
    for label in xrange(classes):
        index = (train_labels.view(np.ndarray).ravel() == label)
        class_data = data[index, :, :]
        class_data = np.sum(class_data, axis=(0, 2))
        class_specific_activations[label, :, :] = class_data.reshape((dimension, dimension))

    # plot all of the class-specific matrices (these should be different if object-category selective)
    for label in xrange(classes):
        pl.subplot(2, 5, label + 1)
        pl.imshow(
            class_specific_activations[label, :, :],
            interpolation='nearest',
        )
        pl.xticks([])
        pl.yticks([])

    pl.show()

    # save class_specific_activations
    csa = {'csa': class_specific_activations}
    savemat("csa.mat", csa)

if __name__ == '__main__':
    main()
