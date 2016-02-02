import os
import time
import theano
import pylab as pl
import numpy as np
import cPickle as pickle
import sparse_filtering as sf
from theano import tensor as t
from scipy.io import loadmat, savemat
import utilities.visualize as visualize
import utilities.connections as connections


# train regular and topographic sparse filtering models
model_type_meta = [['SparseFilter'], ['GroupSF']]
topographic_parameters = [3, 1]
group_meta = [None, topographic_parameters[0]]
step_meta = [None, topographic_parameters[1]]
models = {}
f_hats = {}
data = None
n_filters = None

for m in xrange(len(model_type_meta)):

    print m

    # define global parameters
    model_type = model_type_meta[m]
    convolution = 'n'
    file_name = "patches.mat"
    channels = 1
    patch_size = 16
    n_filters = 625  # 1600
    dimensions = ([n_filters, patch_size * patch_size * channels],)  # ([100, 256],)
    pool = None
    group = group_meta[m]
    step = step_meta[m]
    learn_rate = 0.001
    iterations = [10]  # [50]  # [1]
    verbosity = 0
    opt = 'GD'
    whitening = 'y'
    test_model = 'n'
    examples = None
    batch_size = 1000  # 360  # 8000  # todo: figure out why large batches produce nan cost

    # read in the training data
    print "loading data..."
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "data", file_name)
    data = loadmat(file_path)['X']
    data -= data.mean(axis=0)
    data = np.float32(data.T)

    # determine number of batches
    n_batches, rem = divmod(data.shape[0], batch_size)

    # construct the network
    print "building model..."
    model = sf.Network(
        model_type=model_type,
        weight_dims=dimensions,
        p=pool,
        group_size=group,
        step=step,
        lr=learn_rate,
        opt=opt,
        c=convolution,
        test=test_model,
        batch_size=batch_size,
        random='y',
        weights=None
    )

    # compile the training, output, and test functions for the network
    print "compiling theano functions..."
    train, outputs, test = model.training_functions(data)

    # train the sparse filtering network
    print "training network..."
    t = time.time()
    cost = {}
    weights = {}
    for l in xrange(model.n_layers):

        cost_layer = []
        w = None

        # iterate over training epochs
        for epoch in xrange(iterations[0]):

            # go though [mini]batches
            for batch_index in xrange(n_batches):

                # create index for random [mini]batch
                index = np.int32(np.random.randint(data.shape[0], size=batch_size))

                c, w = train[l](index=index)
                cost_layer.append(c)
                print("Layer %i cost at epoch %i and batch %i: %f" % (l + 1, epoch, batch_index, c))

        # add layer cost and weights to the dictionaries
        cost['layer' + str(l)] = cost_layer
        weights['layer' + str(l)] = w

    # save model to dictionary
    models[model_type[0]] = model

    # calculate and display elapsed training time
    elapsed = time.time() - t
    print('Elapsed training time: %f' % elapsed)

    # plot the cost function over time
    visualize.plotCost(cost)

    # visualize the receptive fields of the first layer
    visualize.drawplots(weights['layer0'].T, color='gray', convolution=convolution,
                        pad=0, examples=None, channels=channels)

    # get activations of first layer and save in dictionary
    f_hat, _, _, _, _, _ = outputs[0](data)
    f_hats[model_type[0]] = f_hat

# project activations of both networks up using local connections
group_matrix = connections.gMatToroidal(n_filters, topographic_parameters[0], topographic_parameters[1], centered='n')
gf_hats = {}
for model in model_type_meta:
    model = model[0]
    gf_hats[model] = np.dot(f_hats[model].T, group_matrix)

# evaluate the sparseness of the distributions
pl.figure(1)
bins = np.linspace(0, 1, 100)
pl.hist(gf_hats['SparseFilter'].flatten(), bins=bins, alpha=0.5, label='Sparse Filtering')
pl.hist(gf_hats['GroupSF'].flatten(), bins=bins, alpha=0.5, label='Topographic Sparse Filtering')
pl.legend(loc='upper right')
pl.xlabel('Activation Value')
pl.ylabel('Frequency')
pl.title('Preservation of Sparsity based on Local Connections')
pl.show()

# evaluate the sparseness of the first level
pl.figure(2)
bins = np.linspace(0, 0.2, 100)
pl.hist(f_hats['SparseFilter'].flatten(), bins=bins, alpha=0.5, label='Sparse Filtering')
pl.hist(f_hats['GroupSF'].flatten(), bins=bins, alpha=0.5, label='Topographic Sparse Filtering')
pl.legend(loc='upper right')
pl.xlabel('Activation Value')
pl.ylabel('Frequency')
pl.title('Sparsity of the First Layer')
pl.show()
