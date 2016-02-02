import os
import h5py
import glob
import time
import numpy.matlib
import numpy as np
import pylab as pl
import sparse_filtering as sf
from utilities.connections import distMat
from scipy.stats import pearsonr, spearmanr
from utilities import visualize
from scipy.io import loadmat, savemat
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelBinarizer
from scipy.spatial.distance import cdist


def main():

    # get the folders in "saved" and select most recent
    base_path = os.path.dirname(__file__)
    folder_path = os.path.join(base_path, "saved")
    folders = os.listdir(folder_path)
    folder = folders[3]  # select most frequent folder  # -1

    # load in activation data
    print "loading in the data..."
    file_path = os.path.join(folder_path, folder, "concatenated_activations.mat")
    # data = loadmat(file_path)['master']  # [examples, neurons, image-space]
    data = h5py.File(file_path, 'r')['master']
    data = np.array(data)
    data = data.T
    print data.shape
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
    print "performing neuron-wise regularized linear regression..."
    neurons = data.shape[1]
    classes = 10
    coefficients = np.zeros((neurons, classes))
    for neuron in xrange(data.shape[1]):
        print neuron
        x = data[:, neuron, :].reshape(data.shape[0] * data.shape[2], 1)
        clf = Ridge(alpha=1.0)
        clf.fit(y_labels, x)
        coefficients[neuron, :] = clf.coef_

    # save the coefficients
    c = {'coefficients': coefficients}
    coefficient_path = os.path.join(folder_path, folder, "coefficients.mat")
    savemat(coefficient_path, c)

    # visualize histogram of coefficients
    pl.hist(np.abs(coefficients.flatten()), bins=30)
    pl.title('Frequency Distribution of Coefficient Values')
    pl.xlabel('Coefficient Value')
    pl.ylabel('Frequency')
    pl.show()

    # todo: find the N sparse filters from the data
    model = ['SparseFilter']
    n_filters = 10
    input_dim = coefficients.shape[1]
    dimensions = ([n_filters, input_dim],)  # number of filters equals number of classes
    pool = None
    group = None
    step = None
    learn_rate = .001
    opt = 'GD'
    convolution = 'n'
    test = 'n'
    batch_size = 1000
    random = 'n'
    weights = None
    iterations = 1000
    channels = 1
    n_batches = coefficients.shape[0] / batch_size
    if n_batches == 0:
        n_batches = 1

    # construct the network
    print "building model..."
    model = sf.Network(
        model_type=model,
        weight_dims=dimensions,
        p=pool,
        group_size=group,
        step=step,
        lr=learn_rate,
        opt=opt,
        c=convolution,
        test=test,
        batch_size=batch_size,
        random=random,
        weights=weights
    )

    # compile the training, output, and test functions for the network
    print "compiling theano functions..."
    train, outputs, test = model.training_functions(np.float32(coefficients))

    # train the sparse filtering network
    print "training network..."
    t = time.time()
    cost = {}
    weights = {}
    layer = None
    for l in xrange(model.n_layers):

        layer = 'layer' + str(l)
        cost_layer = []
        w = None

        # iterate over training epochs
        for epoch in xrange(iterations):

            # go though [mini]batches
            for batch_index in xrange(n_batches):

                c, w = train[l](index=batch_index)
                cost_layer.append(c)
                print("Layer %i cost at epoch %i and batch %i: %f" % (l + 1, epoch, batch_index, c))

        # add layer cost and weights to the dictionaries
        cost[layer] = cost_layer
        weights[layer] = w

    # calculate and display elapsed training time
    elapsed = time.time() - t
    print('Elapsed training time: %f' % elapsed)

    # order the components based on their activations (proxy for amount of variance explained)
    activations, _, _, _, _, _ = outputs[0](np.float32(coefficients))
    activations_summed = np.sum(np.abs(activations), axis=1)
    index = np.argsort(activations_summed)
    weights[layer] = weights[layer][index]

    # save the components (each column represents a component with each element the value for each object category)
    components_path = os.path.join(folder_path, folder, 'weights.mat')
    savemat(components_path, weights)

    # plot the cost function over time
    visualize.plotCost(cost)

    # visualize the components with respect to the object categories
    pl.imshow(weights[layer], interpolation='nearest')
    pl.title('Sparse Filtering Components')
    pl.xlabel('Weights')
    pl.ylabel('Filters')
    pl.xticks(np.arange(1, 10, 10))
    pl.yticks(np.arange(1, 10, 10))
    pl.show()

    # project the components back onto the cortical sheet (i.e., the dot product between each neuron's model
    # coefficients and each component)
    projections = activations
    visualize.drawplots(projections.T, color='gray', convolution=convolution,
                        pad=0, examples=None, channels=channels)

    # todo: compare the similarity of adjacent neurons of different distances and visualize
    distance_measure = 'cityblock'
    max_distance = cdist(np.atleast_2d([0, 0]), np.atleast_2d([np.sqrt(neurons), np.sqrt(neurons)]), distance_measure)
    continuity_data = np.zeros((1, max_distance))
    distances = distMat(neurons, d=neurons * 100, kind=distance_measure)
    pl.imshow(distances)
    pl.show()
    divisor = np.zeros((1, max_distance))
    for i in xrange(neurons):
        for j in xrange(neurons):
            correlation = pearsonr(coefficients[i, :].T, coefficients[j, :].T)
            d = distances[i, j]
            print d, correlation
            continuity_data[0, d] += correlation[0]
            divisor[0, d] += 1
            c += 1

    correlation_averages = continuity_data / divisor
    correlation_averages = correlation_averages[~np.isnan(correlation_averages)]
    # correlation_std = np.std(continuity_data, axis=0)
    # correlation_std = correlation_std[~np.isnan(correlation_std)] # todo: allow computation of std
    temp_std = np.linspace(.2, .1, len(correlation_averages))
    print temp_std
    print correlation_averages
    hypothetical_averages = [1., 0.7, 0.5, 0.4, 0.28, 0.21, 0.15, 0.09, 0.07, 0.05]
    hypothetical_stds = np.linspace(.07, .1, len(correlation_averages) - 1)
    fig = visualize.plot_mean_std(correlation_averages[0:10], temp_std[0:10], hypothetical_averages, hypothetical_stds)
    fig.show()

if __name__ == '__main__':
    main()
