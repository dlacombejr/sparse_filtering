import os
import time
import h5py
import theano
import numpy as np
import cPickle as pickle
import sparse_filtering as sf
from theano import tensor as t
from scipy.cluster.vq import whiten
from scipy.io import loadmat, savemat
import utilities.visualize as visualize
from theano.tensor.signal.downsample import max_pool_2d


def quadrant_pooling(quadrant_size):

    x = t.ftensor4()
    down_sample = (quadrant_size, quadrant_size)
    y = max_pool_2d(x, ds=down_sample, mode='sum')
    fn = theano.function(inputs=[x], outputs=[y])

    return fn


def main():

    # define global parameters
    model_type = ['SparseFilter']
    convolution = 'n'
    filename = 'unlabeled_10000.mat'
    # 'STL_10_unlabeled_patches.h5'  # _sample10'  # 'patches.mat'  # LCN  # _raw10  # _raw1000
    channels = 3
    patch_size = 14
    n_filters = 1600  # 1600
    dimensions = ([n_filters, patch_size * patch_size * channels],)  # ([100, 256],)
    pool = None
    group = None
    step = None
    learn_rate = 0.0001
    iterations = [100]  # [50]  # [1]
    verbosity = 2
    opt = 'GD'
    whitening = 'y'
    test_model = 'y'
    examples = None
    batch_size = 1000  # 360  # 8000  # todo: figure out why large batches produce nan cost

    # load in data
    print "loading data..."
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "data", filename)
    data = h5py.File(file_path, 'r')['patches']
    data = np.array(data)
    data = data.T

    # preprocess the data and convert to float; NOTE: data may have already been normalized using LCN (check data read)
    channel_length = patch_size ** 2
    for channel in xrange(channels):
        start = channel * channel_length
        end = start + channel_length
        data[start:end] -= data[start:end].mean(axis=0)
        data[start:end] /= data[start:end].std(axis=0) + 1e-8
        data[start:end] = whiten(data[start:end].T).T

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

    # calculate and display elapsed training time
    elapsed = time.time() - t
    print('Elapsed training time: %f' % elapsed)

    # create sub-folder for saved model
    directory_format = "./saved/%4d-%02d-%02d_%02dh%02dm%02ds"
    directory_name = directory_format % time.localtime()[0:6]
    os.mkdir(directory_name)

    # save the model for later use
    full_path = directory_name + '/model.pkl'
    pickle.dump(model, open(full_path, 'w'), pickle.HIGHEST_PROTOCOL)

    # save weights separately
    savemat(directory_name + '/weights.mat', weights)

    # create log file
    ex = None
    log_file = open(directory_name + "/log.txt", "wb")
    for m in range(len(model_type)):
        log_file.write(
            "Model layer %d: \n model:%s \n dimensions:%4s \n iterations:%3d \n" % (m,
                                                                                    model_type[m],
                                                                                    dimensions[m],
                                                                                    iterations[m])
        )
        if model == 'GroupSF' or model == 'GroupConvolutionalSF':
            log_file.write(
                " Groups: %d \n Step: %d" % (group, step)
            )
        ex = data.shape[0]
        if examples is not None:
            ex = examples

    log_file.write(
        " Data-set: %s \n Examples: %6d \n Whitened: %s" % (filename, ex, whitening)
    )
    log_file.write('\nElapsed training time: %f' % elapsed)
    log_file.close()

    # get variables and save
    if verbosity >= 1:

        for batch in xrange(n_batches):

            # get variables of interest
            activations_norm = {}
            activations_raw = {}
            activations_shuffled = {}
            reconstruction = {}
            error_recon = {}
            pooled = {}

            # f_hat, rec, err, f_hat_shuffled, f, p = outputs[l]()
            begin = batch * batch_size
            end = begin + batch_size
            f_hat, rec, err, f_hat_shuffled, f, p = outputs[model.n_layers - 1](data[begin:end])

            activations_norm['layer' + str(l) + '_batch' + str(batch)] = f_hat
            activations_raw['layer' + str(l) + '_batch' + str(batch)] = f
            activations_shuffled['layer' + str(l) + '_batch' + str(batch)] = f_hat_shuffled
            reconstruction['layer' + str(l) + '_batch' + str(batch)] = err
            error_recon['layer' + str(l) + '_batch' + str(batch)] = rec
            pooled['layer' + str(l) + '_batch' + str(batch)] = p

            # save model as well as weights and activations separately
            savemat(directory_name + '/activations_norm_' + 'layer' + str(l) + '_batch' +
                    str(batch) + '.mat', activations_norm)
            savemat(directory_name + '/activation_raw_' + 'layer' + str(l) + '_batch' +
                    str(batch) + '.mat', activations_raw)

    # display figures
    if verbosity == 2:

        # if GD, plot the cost function over time
        if opt == 'GD':
            visualize.plotCost(cost)

        # visualize the receptive fields of the first layer
        visualize.drawplots(weights['layer0'].T, color='gray', convolution=convolution,
                            pad=0, examples=None, channels=channels)

        # # visualize the distribution of lifetime and population sparseness
        # for l in xrange(len(dimensions)):
        #     layer = 'layer' + str(l)
        #     if convolution == 'n':
        #         visualize.dispSparseHist(activations_norm[layer], l)
        #     elif convolution == 'y':
        #         visualize.dispSparseHist(activations_shuffled[layer].reshape(dimensions[l][0],
        #                                                                      data.shape[0] *
        #                                                                      activations_shuffled[layer].shape[2] *
        #                                                                      activations_shuffled[layer].shape[3]),
        #                                  layer=l)
        #
        # # visualize the distribution of activity across the "cortical sheet" and reconstruction
        # visualize.drawplots(activations_norm['layer0'], color='gray', convolution=convolution,
        #                     pad=1, examples=100)
        #
        # # visualize reconstruction capabilities
        # if convolution == 'n':
        #     visualize.drawReconstruction(data[:, 0:100], error_recon['layer0'][:, 0:100], 'y', convolution, 1)
        # elif convolution == 'y':
        #     visualize.convolutional_reconstruction(data[0, :, :, :], activations_raw['layer0'], weights['layer0'],
        #                                            color='gray', convolution=convolution)
        # print('Reconstructed error: %e' % reconstruction['layer0'])
        #
        # # additional visualizations for convolutional network
        # if convolution == 'y':
        #
        #     dim = activations_raw['layer0'].shape[2]
        #
        #     # visualize an example of a convolved image
        #     visualize.visualize_convolved_image(activations_raw['layer0'], dim=dim)
        #     # print activations_raw['layer0']
        #
        #     # visualize max-pooled activations and LCN output
        #     visualize.visualize_convolved_image(pooled['layer0'][0, :, :, :].reshape(1,
        #                                                                              pooled['layer0'].shape[1],
        #                                                                              pooled['layer0'].shape[2],
        #                                                                              pooled['layer0'].shape[3]),
        #                                         dim=dim / 2)
        #
        #     # visualize an example of a LCNed convolved image after max pooling
        #     # temp = activations_raw['layer0']    #[0, :, :, :]
        #     temp = pooled['layer0']    #[0, :, :, :]
        #     # print temp.shape
        #     for i in range(temp.shape[1]):
        #         temp[0, i, :, :] = scaling.LCNinput(temp[0, i, :, :].reshape((1, 1, dim / 2, dim / 2)), kernel_shape=5)
        #     # temp = scaling.LCNinput(temp, kernel_shape=5)
        #     visualize.visualize_convolved_image(temp, dim=dim / 2)
        #     # print temp

    ''' ================================ Test the Model ======================================= '''

    # test the model if evaluating classification performance
    if test_model == 'y':
        print 'testing...'

        from sklearn import svm
        from sklearn.metrics import confusion_matrix

        # set some new local parameters
        train_data_file = "STL_10_lcn_train.mat"
        train_labels_file = "train.mat"
        test_data_file = "STL_10_lcn_test.mat"
        test_labels_file = "test.mat"
        model_type = ["ConvolutionalSF"]
        dimensions = ([1, n_filters, patch_size, patch_size], )
        convolution = 'y'
        test_model = 'y'
        batch_size = 100

        # load in STL-10 training data (all pre-normalized using LCN)
        print "loading in training and test data..."
        file_path = os.path.join(base_path, "data", train_data_file)
        train_data = loadmat(file_path)['X']
        file_path = os.path.join(base_path, "data", train_labels_file)
        train_labels = loadmat(file_path)['y']

        # load in STL-10 test data (all pre-normalized using LCN)
        file_path = os.path.join(base_path, "data", test_data_file)
        test_data = loadmat(file_path)['X']
        file_path = os.path.join(base_path, "data", test_labels_file)
        test_labels = loadmat(file_path)['y']

        # read in the pre-defined fold indices
        file_path = os.path.join(base_path, "data", "train.mat")
        fold_indices = loadmat(file_path)['fold_indices']
        fold_indices -= np.ones(fold_indices.shape)  # make zero-index

        # initialize convolutional network with learned parameters from above
        old_weights = model.layers[0].w.eval().reshape((-1, channels, patch_size, patch_size))
        old_weights = theano.shared(old_weights.astype(dtype=theano.config.floatX))
        test_model = sf.Network(
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
            weights=old_weights
        )

        # compile the training, output, and test functions for the network
        print "compiling theano functions..."
        _, _, test = test_model.training_functions(train_data)

        # train and test a SVM classifier for each layer (including pixels as baseline)
        accuracy = {}
        train_input = None
        test_input = None
        cm = None
        for layer in range(1, 2):  # range(test_model.n_layers + 1):  # skipping pixels for now

            # create dictionary for layer
            accuracy['layer' + str(layer)] = {}

            # create quadrant pooling function based on size of output from layer
            quadrant_size = test[layer - 1](test_data[0]).shape[3]
            quad_pool = quadrant_pooling(quadrant_size)

            # loop over pre-defined folds
            n_folds = fold_indices.shape[1]
            for fold in xrange(n_folds):

                # get fold data
                train_data_fold = np.squeeze(train_data[fold_indices[0][fold]])
                train_labels_fold = np.squeeze(train_labels[fold_indices[0][fold]])

                # pixel inputs
                if layer == 0:

                    if fold == 0:  # only get test data once
                        test_input = test_data.reshape(test_data.shape[0], test_data.shape[1] *
                                                       test_data.shape[2] * test_data.shape[3])

                    train_input = train_data_fold.reshape(train_data_fold.shape[0], train_data_fold.shape[1] *
                                                          train_data_fold.shape[2] * train_data_fold.shape[3])

                # hidden layers
                elif layer > 0:

                    # get the output of the current layer in the model given the training / test data and then reshape
                    # TODO: use raw output as training and testing data?
                    if fold == 0:  # only get test data once
                        print "getting test data..."
                        test_input = np.zeros((test_data.shape[0], n_filters, 2, 2))
                        n_batches = test_data.shape[0] / batch_size
                        for batch in xrange(n_batches):
                            print "for batch %d" % batch
                            batch_start = batch * batch_size
                            batch_end = batch_start + batch_size
                            temp = test[layer - 1](test_data[batch_start:batch_end])  # test_data[0:batch_size]
                            temp = temp[0]
                            # for i in xrange(2):
                            #     for j in xrange(2):
                            #         pool_size = 48
                            #         i_start = i * pool_size
                            #         i_end = i_start + pool_size
                            #         j_start = j * pool_size
                            #         j_end = j_start + pool_size
                            #         test_input[batch_start:batch_end, :, i, j] = \
                            #             np.sum(
                            #                 temp[:, :, i_start:i_end, j_start:j_end],
                            #                 axis=(2, 3)
                            #         )
                            test_input[batch_start:batch_end] = quad_pool(temp)
                        test_input = test_input.reshape(test_input.shape[0], test_input.shape[1] *
                                                        test_input.shape[2] * test_input.shape[3])

                    print "getting training data..."
                    # todo: also do quadrant pooling for training data (done) perhaps don't do batches here
                    # train_input = test[layer - 1](train_data_fold)  # test_data[0:batch_size]
                    # train_input = train_input[0].reshape(train_input[0].shape[0], train_input[0].shape[1] *
                    #                                      train_input[0].shape[2] * train_input[0].shape[3])
                    train_input = np.zeros((train_data_fold.shape[0], n_filters, 2, 2))
                    n_batches = train_data_fold.shape[0] / batch_size
                    for batch in xrange(n_batches):
                        print "for batch %d" % batch
                        batch_start = batch * batch_size
                        batch_end = batch_start + batch_size
                        temp = test[layer - 1](train_data_fold[batch_start:batch_end])  # test_data[0:batch_size]
                        temp = temp[0]
                        # for i in xrange(2):
                        #     for j in xrange(2):
                        #         pool_size = 48
                        #         i_start = i * pool_size
                        #         i_end = i_start + pool_size
                        #         j_start = j * pool_size
                        #         j_end = j_start + pool_size
                        #         train_input[batch_start:batch_end, :, i, j] = \
                        #             np.sum(
                        #                 temp[:, :, i_start:i_end, j_start:j_end],
                        #                 axis=(2, 3)
                        #         )
                        train_input[batch_start:batch_end] = quad_pool(temp)
                    train_input = train_input.reshape(train_input.shape[0], train_input.shape[1] *
                                                      train_input.shape[2] * train_input.shape[3])

                # normalize the inputs for each dimension (zero-mean and unit-variance)
                if fold == 0:  # only normalize test data once
                    test_input -= test_input.mean(axis=1)[:, np.newaxis]
                    test_input /= np.std(test_input, axis=1)[:, np.newaxis]
                train_input -= train_input.mean(axis=1)[:, np.newaxis]
                train_input /= np.std(train_input, axis=1)[:, np.newaxis]

                # train linear support vector machine
                print("Training linear SVM...")
                clf = svm.SVC(kernel="linear").fit(train_input, np.ravel(train_labels_fold[0:examples]))

                # get predictions from SVM and calculate accuracy
                print("Making predictions...")
                predictions = clf.predict(test_input)
                accuracy['layer' + str(layer)]['fold' + str(fold)] = clf.score(test_input, test_labels[0:examples])

                # display results and log them
                print("Accuracy of the classifier for fold %d at layer %1d: %0.4f" %
                      (fold, layer, accuracy['layer' + str(layer)]['fold' + str(fold)]))
                cm = confusion_matrix(test_labels[0:examples], predictions)
                log_file = open(directory_name + "/log.txt", "a")
                log_file.write(
                    "\nAccuracy of the classifier for fold %d at layer %1d: %0.4f" %
                    (fold, layer, accuracy['layer' + str(layer)]['fold' + str(fold)])
                )
                log_file.close()

            # # visualize the confusion matrix
            # if test_model == 'y' and verbosity == 2:
            #
            #     import pylab as pl
            #
            #     pl.imshow(cm, interpolation='nearest')
            #     pl.title('Confusion Matrix for Network')
            #     pl.colorbar()
            #     pl.ylabel('True Label')
            #     pl.xlabel('Predicted Label')
            #     pl.show()

        # save the test results
        savemat('accuracy', accuracy)

if __name__ == '__main__':
    main()
