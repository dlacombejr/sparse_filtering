import os
import numpy.matlib
import numpy as np
from scipy.io import loadmat, savemat
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelBinarizer


def main():

    # load in activation data
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "saved", "activations_concatenated.mat")
    data = loadmat(file_path)['master']  # [examples, neurons, image-space]

    # TODO: scale and normalize data

    # load in data labels
    file_path = os.path.join(base_path, "data", "CIFAR_data.mat")
    train_labels = loadmat(file_path)['y']

    # augment training_labels to account for extra examples in image-space
    y_labels = numpy.matlib.repmat(train_labels, 1, data.shape[2]).reshape((data.shape[0] * data.shape[2], 1))

    # convert labels to binary vector
    lb = LabelBinarizer()
    lb.fit(train_labels)
    y_labels = lb.transform(y_labels)

    # perform neuron-wise regularized linear regression to obtain coefficients
    neurons = data.shape[1]
    classes = 10
    coefficients = np.zeros((neurons, classes))
    for neuron in xrange(data.shape[1]):
        x = data[:, neuron, :].reshape(data.shape[0] * data.shape[2], 1)
        clf = Ridge(alpha=1.0)
        clf.fit(y_labels, x)
        coefficients[neuron, :] = clf.coef_

    # save the coefficients
    c = {'coefficients': coefficients}
    savemat('coefficients.mat', c)

if __name__ == '__main__':
    main()