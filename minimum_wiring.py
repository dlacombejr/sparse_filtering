import os
import theano
import scipy
import numpy as np
import pylab as pl
import cPickle as pickle
import theano.tensor as t
import sparse_filtering as sf
import utilities.init as init
from utilities.BP import backprop
from scipy.io import loadmat, savemat
from utilities.connections import distMat
from utilities.visualize import drawplots


# switches
convolutional = 'n'


def rectify(X):
    return t.maximum(X, 0.)


def softmax(X):
    e_x = t.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = t.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = t.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


def final_layer(X, w_out):

    out = softmax(t.dot(X, w_out))

    return out


# def distMat(neurons):
#
#     # define dimension of cortical sheet
#     dim = np.sqrt(neurons)
#
#     # create coordinates
#     coordinates = []
#     for i in range(int(dim)):
#         for j in range(int(dim)):
#             coordinates.append([i, j])
#
#     # get distance from center position
#     center = [np.ceil(dim / 2), np.ceil(dim / 2)]
#     distances = scipy.spatial.distance.cdist(coordinates, np.atleast_2d(center)).reshape((dim, dim))
#
#     # roll it to first position
#     back = int(np.floor(dim / 2))
#     distances = np.roll(distances, -back, axis=0)
#     distances = np.roll(distances, -back, axis=1)
#
#     return distances


# load in the network(s; topological and non-topological) into a dictionary
print "loading model(s)..."
models = {}
model_folders = ['SF', 'tSF']
model_names = ['ConvolutionalSF_model', 'TopologicalConvolutionalSF_model']
base_path = os.path.dirname(__file__)
for model in xrange(len(model_names)):
    # model_file_name = model_names[model] + ".pkl"
    file_path = os.path.join(base_path, "saved", model_folders[model], 'model.pkl')
    models[model_names[model]] = pickle.load(open(file_path, 'rb'))

# load in the training and testing data (should be preprocessed)
print "loading data..."
if convolutional == 'n':
    file_name = "patches.mat"
    file_path = os.path.join(base_path, "data", file_name)
    data = loadmat(file_path)['X']
    data = np.float32(data.T)
elif convolutional == 'y':
    train_file_name = "STL_10_lcn_train.mat"
    test_file_name = "STL_10_lcn_test.mat"
    train_file_path = os.path.join(base_path, "data", train_file_name)
    test_file_path = os.path.join(base_path, "data", test_file_name)
    train_data = loadmat(train_file_path)['X']
    test_data = loadmat(test_file_path)['X']
    train_data = np.float32(train_data)
    test_data = np.float32(test_data)

# load in the corresponding labels
if convolutional == 'y':
    print "loading labels..."
    train_labels_file = "train.mat"
    test_labels_file = "test.mat"
    train_labels_path = os.path.join(base_path, "data", train_labels_file)
    test_labels_path = os.path.join(base_path, "data", test_labels_file)
    train_labels = loadmat(train_labels_path)['y']
    test_labels = loadmat(test_labels_path)['y']

# compile functions and grab test
print "compiling theano functions..."
test = {}
for model in model_names:
    if convolutional == 'n':
        _, test[model], _ = models[model].training_functions(data)  # using output as test
    elif convolutional == 'y':
        _, _, test[model] = models[model].training_functions(train_data)

# get the output activations of the last layer in the network(s) for next layer / training and test data
if convolutional == 'n':
    print "getting output of (both) model(s) for training second layer..."
    out = {}
    for model in model_names:
        out[model] = test[model][models[model].n_layers - 1](data)
        # print out[model][0].T[0:625].shape
        # drawplots(out[model][0].T[0:625].shape, color='gray', convolution='n', pad=0, examples=None, channels=1)
elif convolutional == 'y':
    print "getting output of (both) model(s) for train and test data..."
    train_out = {}
    test_out = {}
    for model in model_names:
        temp = test[model][models[model].n_layers - 1](train_data)
        train_out[model] = temp.reshape(temp.shape[0], temp.shape[1] * temp.shape[2] * temp.shape[3])

# train a fully connected network on the output from the training data test the classification accuracy of the final
# layer based on the outputs for the test data (optional)
# OR
# train a fully connected sparse filtering network on the second layer
final_weights = {}
weights = None
for model in model_names:

    if convolutional == 'n':

        # construct the network
        print "building model..."
        model_ = sf.Network(
            model_type=['SparseFilter'],
            weight_dims=([100, 625],),
            p=None,
            group_size=None,
            step=None,
            lr=0.001,
            opt='GD',
            c=convolutional,
            test='n',
            batch_size=50000,
            random='n',
            weights=None
        )

        # normalize the training data
        data = out[model][0].T
        data = data - np.tile(data.mean(axis=1), (625, 1)).T

        # compile the training, output, and test functions for the network
        print "compiling theano functions..."
        train, _, _ = model_.training_functions(data)

        # train the sparse filtering network
        print "training network..."
        for epoch in xrange(200):  # 100

            cost, weights = train[0](index=0)
            print("Layer %i cost at epoch %i and batch %i: %f" % (1, epoch, 0, cost))

    elif convolutional == 'y':

        print "setting up network for" + model
        X = t.ftensor4()
        Y = t.fmatrix()
        w_out = init.init_weights((train_out[model].shape[1], 10))

        py_x = final_layer(X, w_out)
        y_x = t.argmax(py_x, axis=1)

        cost = t.mean(t.nnet.categorical_crossentropy(py_x, Y))
        params = [w_out]
        updates = RMSprop(cost, params, lr=0.001)

        print "compiling theano functions..."
        train = theano.function(inputs=[X, Y], outputs=[cost, w_out], updates=updates, allow_input_downcast=True)
        predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

        print "training fully connected layer..."
        max_iter = 100
        batch_size = 1000
        n_batches = train_out[model] / batch_size
        for iteration in range(max_iter):
            for batch in xrange(n_batches):
                batch_begin = batch * batch_size
                batch_end = batch_begin + batch_size
                cost, weights = train(train_out[batch_begin:batch_end])
                print "Cost at iteration %d and batch %d for model " % (iteration, batch) + model + ": %d" % cost
            accuracy = float(np.mean(np.argmax(test_labels, axis=1) == predict(test_out)))
            print "Classification performance for model " + model + " at iteration %d: %f" % (
                iteration,
                accuracy
            )

    # save weights
    final_weights[model] = weights

# visualize the weights for each class / neuron across the cortical sheet
print "visualizing weights..."
if convolutional == 'n':
    for model in model_names:
        drawplots(final_weights[model].T, color='gray', convolution='n', pad=0, examples=None, channels=1)
elif convolutional == 'y':
    for model in model_names:
        for category in xrange(train_labels.shape[1]):
            w = final_weights[model][:, category]
            w = w.reshape(np.sqrt(w.shape[0]), np.sqrt(w.shape[0]))
            pl.subplot(2, 5, category + 1)
            pl.imshow(w)

        pl.title("Weight distributions for model " + model)
        pl.show()

# find optimal neuronal positions (for N random initial positions)
print "finding optimal neuronal positions..."
optimal_positions = {}
minimal_wiring_length = {}
# initial_distance = distMat(weights.shape[0])
for model in model_names:
    entity = None
    optimal_positions[model] = []
    minimal_wiring_length[model] = []
    if convolutional == 'n':
        entity = final_weights[model].shape[0]
    elif convolutional == 'y':
        entity = train_labels.shape[1]
    for neuron in xrange(entity):

        weights = final_weights[model][neuron, :]
        distances = distMat(len(weights), d=None, kind='euclidean', inverted='n')

        wiring_lengths = np.dot(np.abs(weights), distances.T)  # should this be transposed?
        minimum_wiring = np.min(wiring_lengths)
        XY = np.argmin(wiring_lengths)

        optimal_positions[model].append(XY)  # [X, Y]  # todo: convert to coordinates
        minimal_wiring_length[model].append(minimum_wiring)

        # w = t.fvector()
        # d_mat = t.fmatrix()
        # # d_mat = t.fvector()
        #
        # x = theano.shared(np.asarray(np.floor(np.sqrt(final_weights[model].shape[0]) / 2), dtype=theano.config.floatX))
        # y = theano.shared(np.asarray(np.floor(np.sqrt(final_weights[model].shape[0]) / 2), dtype=theano.config.floatX))
        #
        # d = t.roll(d_mat, x, axis=0)  # todo: figure out how to convert to int while still being differentiable
        # d = t.roll(d, y, axis=1)
        #
        # # distances = distMat(x, y)
        #
        # cost = t.dot(d.flatten(), w)
        #
        # parameters = [x, y]
        # updates = RMSprop(cost, parameters)
        #
        # train = theano.function(inputs=[w, d_mat], outputs=[x, y, cost], updates=updates)
        #
        # max_iter = 100
        # # X = int(np.floor(np.sqrt(w.shape[0]) / 2))
        # # Y = int(np.floor(np.sqrt(w.shape[0]) / 2))
        # weights = final_weights[model][:, neuron]
        # for iteration in xrange(max_iter):
        #
        #     # distances = np.roll(initial_distance, X, axis=0)
        #     # distances = np.roll(distances, Y, axis=1)
        #
        #     X, Y, cost = train(weights, initial_distance)
        #     # X = np.round(X)
        #     # Y = np.round(Y)
        #
        # optimal_positions[model][neuron] = [X, Y]
        # minimal_wiring_length[model][neuron] = cost

# compare minimal wiring lengths
avg_wiring_length = {}
for model in model_names:
    # accumulator = 0
    # if convolutional == 'n':
    #     entity = final_weights[model].shape[0]
    # elif convolutional == 'y':
    #     entity = train_labels.shape[1]
    # for neuron in xrange(entity):
    #     accumulator += minimal_wiring_length[model][neuron]

    avg_wiring_length[model] = np.mean(minimal_wiring_length[model])
    std = np.std(minimal_wiring_length[model])
    # accumulator / train_labels.shape[1]
    print "Wiring length for model %s: %0.4f +/- (%0.4f)" % (model, avg_wiring_length[model], std)

# plot differences in bar graph
pl.bar(np.arange(len(model_names)), (avg_wiring_length[model_names[0]], avg_wiring_length[model_names[1]]))
pl.show()


pl.figure(2)
bins = np.linspace(0, 1, 100)
divider = np.amax(np.concatenate((minimal_wiring_length[model_names[0]], minimal_wiring_length[model_names[1]])))
sf_mwl = minimal_wiring_length[model_names[0]] / divider
tsf_mwl = minimal_wiring_length[model_names[1]] / divider
pl.hist(sf_mwl, bins=bins, alpha=0.5, label='Sparse Filtering')
pl.hist(tsf_mwl, bins=bins, alpha=0.5, label='Topographic Sparse Filtering')
pl.legend(loc='upper right')
pl.xlabel('Wiring Length')
pl.ylabel('Frequency')
pl.title('Minimum Wiring Length')
pl.show()
