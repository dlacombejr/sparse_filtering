import os
import time
# import h5py
import theano
import numpy as np
import cPickle as pickle
import sparse_filtering as sf
from theano import tensor as t
from utilities import scaling
from scipy.cluster.vq import whiten
from scipy.io import loadmat, savemat
import utilities.visualize as visualize
from theano.tensor.signal.downsample import max_pool_2d


def quadrant_pooling(quadrant_size):

    x = t.ftensor4()
    down_sample = (quadrant_size, quadrant_size)
    y = max_pool_2d(x, ds=down_sample, mode='sum', ignore_border=True)
    fn = theano.function(inputs=[x], outputs=[y])

    return fn


def main():

    """
    This script builds a deep convolutional sparse filtering network that has a final output [examples, maps, 1, 1],
    such that the entire image is viewed. The outputs of the final layer are concatenated together and serve as input
    to a new fully-connected network that uses the original sparse filtering object. The outputs of this fully connected
    layer are then used as input to a supervised classifier to evaluate the degree to which object categories are
    represented using fully unsupervised-learning.

    The standard sparse filtering algorithm can be replaced with the topographic version to evaluate semantic
    organization of high-level feature detectors.
    """

    # define global parameters
    model_type = [
        'ConvolutionalSF',
        'ConvolutionalSF',
        'ConvolutionalSF',
        'ConvolutionalSF',
        # 'ConvolutionalSF'
    ]
    convolution = 'y'
    filename = "train.mat"  # unlabeled
    channels = 3
    patch_size = 14
    n_filters = [
        100,
        200,
        400,
        800,
        # 1600
    ]  #
    # [100, 400, 1600, 6400, 25600]  # 1600  # increasing neurons x4 maintains dimensionality
    dimensions = (
        [n_filters[0], channels, 11, 11],
        [n_filters[1], n_filters[0], 4, 4],
        [n_filters[2], n_filters[1], 3, 3],
        [n_filters[3], n_filters[2], 2, 2],
        # [n_filters[4], n_filters[3], 3, 3]
    )
    # ([n_filters, patch_size * patch_size * channels],)  # ([100, 256],)
    pool = None
    group = None
    step = None
    learn_rate = 0.001  # 0.0001
    iterations = [
        1,
        1,
        1,
        1,
        # 1
    ]  # [5, 5, 5]  # [50]  # [100]
    verbosity = 0
    opt = 'GD'
    whitening = 'y'
    test_model = 'y'
    examples = None
    batch_size = 100  # 360  # 8000
    lcn_kernel = [5, 4, 3, 2]
    aws = 'y'

    # load in data
    print "loading data..."
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "data", filename)
    data = loadmat(file_path)['X']
    # data = h5py.File(file_path, 'r')['X']
    # data = np.array(data)
    # data = data.T

    # preprocess the data and convert to float; NOTE: data may have already been normalized using LCN (check data read)
    print "pre-processing data..."
    data = np.float32(data.reshape(-1, 3, int(np.sqrt(data.shape[1] / 3)), int(np.sqrt(data.shape[1] / 3))))
    data = data[0:examples, :, :, :]
    for channel in range(channels):
        data[:, channel, :, :] = np.reshape(scaling.LCNinput(data[:, channel, :, :].
                                                             reshape((data.shape[0], 1,
                                                                      data.shape[2],
                                                                      data.shape[3])),
                                                             kernel_shape=9), (
                                            data.shape[0],
                                            data.shape[2],
                                            data.shape[3]))

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
        weights=None,
        lcn_kernel=lcn_kernel
    )

    # compile the training, output, and test functions for the network
    print "compiling theano functions..."
    train, outputs, test = model.training_functions(data)

    # train the sparse filtering network
    print "training network..."
    start_time = time.time()
    cost = {}
    weights = {}
    for l in xrange(model.n_layers):

        cost_layer = []
        w = None

        # iterate over training epochs
        for epoch in xrange(iterations[l]):

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
    elapsed = time.time() - start_time
    print('Elapsed training time: %f' % elapsed)

    # create sub-folder for saved model
    directory_name = None
    if aws == 'n':
        directory_format = "./saved/%4d-%02d-%02d_%02dh%02dm%02ds"
        directory_name = directory_format % time.localtime()[0:6]
        os.mkdir(directory_name)
    elif aws == 'y':
        import boto
        from boto.s3.key import Key
        s3 = boto.connect_s3()
        my_bucket = 'dlacombejr.bucket'
        bucket = s3.get_bucket(my_bucket)
        k = Key(bucket)
        directory_format = "./saved/%4d-%02d-%02d_%02dh%02dm%02ds"
        directory_name = directory_format % time.localtime()[0:6]
        os.mkdir(directory_name)

    # save the model for later use
    full_path = directory_name + '/model.pkl'
    pickle.dump(model, open(full_path, 'w'), pickle.HIGHEST_PROTOCOL)
    if aws == 'y':
        k.key = full_path
        k.set_contents_from_filename(full_path)
        os.remove(full_path)

    # save weights separately
    savemat(directory_name + '/weights.mat', weights)
    if aws == 'y':
        k.key = directory_name + '/weights.mat'
        k.set_contents_from_filename(directory_name + '/weights.mat')
        os.remove(directory_name + '/weights.mat')

    # save the cost functions
    savemat(directory_name + '/cost.mat', cost)
    if aws == 'y':
        k.key = directory_name + '/cost.mat'
        k.set_contents_from_filename(directory_name + '/cost.mat')
        os.remove(directory_name + '/cost.mat')

    # create log file
    log_file = open(directory_name + "/log.txt", "wb")  # todo: create log file by looping through args
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
    if aws == 'y':
        k.key = directory_name + "/log.txt"
        k.set_contents_from_filename(directory_name + "/log.txt")
        os.remove(directory_name + "/log.txt")

    # collect the activations from the last layer to train fully connected layer
    activations_concatenated = []
    for batch in xrange(n_batches):
        begin = batch * batch_size
        end = begin + batch_size
        f_hat, _, _, _, _, _ = outputs[model.n_layers - 1](data[begin:end])
        activations_concatenated.append(
            f_hat.reshape(
                batch_size,
                f_hat.shape[1] * f_hat.shape[2] * f_hat.shape[3]
            )
        )

    # normalize the input
    final_input = np.asarray(activations_concatenated)
    final_input -= final_input.mean(axis=1)

    # train a regular sparse filtering network on top of final layer
    print "building model..."
    model = sf.Network(
        model_type=['SparseFilter'],
        weight_dims=([1089, final_input.shape[1]],),  # 33x33 is odd perfect square
        p=pool,
        group_size=group,
        step=step,
        lr=learn_rate,
        opt=opt,
        c='n',
        test=test_model,
        batch_size=batch_size,
        random='y',
        weights=None
    )

    # compile the training, output, and test functions for the network
    print "compiling theano functions..."
    train, outputs, test = model.training_functions(final_input)

    # train the sparse filtering network
    print "training network..."
    start_time = time.time()
    cost = {}
    weights = {}
    for l in xrange(model.n_layers):

        cost_layer = []
        w = None

        # iterate over training epochs
        for epoch in xrange(iterations[l]):

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
    elapsed = time.time() - start_time
    print('Elapsed training time: %f' % elapsed)

    # save the model for later use
    full_path = directory_name + '/model2.pkl'
    pickle.dump(model, open(full_path, 'w'), pickle.HIGHEST_PROTOCOL)
    if aws == 'y':
        k.key = full_path
        k.set_contents_from_filename(full_path)
        os.remove(full_path)

    ''' ================================ Test the Model ======================================= '''

    # test the model if evaluating classification performance
    if test_model == 'y':
        print 'testing...'

        from sklearn import svm

        # set some new local parameters
        train_data_file = "STL_10_lcn_train.mat"  # "train.mat"
        train_labels_file = "train.mat"
        test_data_file = "STL_10_lcn_test.mat"  # "test.mat"
        test_labels_file = "test.mat"
        batch_size = 100

        # todo: read in lcn data
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

        # # preproces training and test data
        # print "preprocessing training and test data..."
        # print train_data.shape

        # train_data = np.float32(train_data.reshape(-1,
        #                                            3,
        #                                            int(np.sqrt(train_data.shape[1] / 3)),
        #                                            int(np.sqrt(train_data.shape[1] / 3)))
        #                         )

        # print train_data.shape
        # for channel in range(channels):
        #     train_data[:, channel, :, :] = np.reshape(scaling.LCNinput(train_data[:, channel, :, :].
        #                                                                reshape((train_data.shape[0], 1,
        #                                                                         train_data.shape[2],
        #                                                                         train_data.shape[3])),
        #                                                                kernel_shape=9), (
        #                                               train_data.shape[0],
        #                                               train_data.shape[2],
        #                                               train_data.shape[3]))
        #
        # test_data = np.float32(test_data.reshape(-1,
        #                                          3,
        #                                          int(np.sqrt(test_data.shape[1] / 3)),
        #                                          int(np.sqrt(test_data.shape[1] / 3)))
        #                         )
        # for channel in range(channels):
        #     test_data[:, channel, :, :] = np.reshape(scaling.LCNinput(test_data[:, channel, :, :].
        #                                                               reshape((test_data.shape[0], 1,
        #                                                                        test_data.shape[2],
        #                                                                        test_data.shape[3])),
        #                                                               kernel_shape=9), (
        #                                              test_data.shape[0],
        #                                              test_data.shape[2],
        #                                              test_data.shape[3]))

        # read in the pre-defined fold indices
        file_path = os.path.join(base_path, "data", "train.mat")
        fold_indices = loadmat(file_path)['fold_indices']
        fold_indices -= np.ones(fold_indices.shape)  # make zero-index

        # train and test a SVM classifier for each layer (including pixels as baseline)
        accuracy = {}
        accuracy_list = []
        train_input = None
        test_input = None
        cm = None
        for layer in range(1, 4):  # range(test_model.n_layers + 1):  # skipping pixels for now

            # create dictionary for layer
            accuracy['layer' + str(layer)] = {}

            # create quadrant pooling function based on size of output from layer
            quadrant_size = test[layer - 1](test_data[0, :, :, :].reshape((1, 3, 96, 96)))[0].shape[3] / 2
            quad_pool = quadrant_pooling(quadrant_size)

            # loop over pre-defined folds
            n_folds = fold_indices.shape[1]
            for fold in xrange(n_folds):

                # get fold data
                fold_index = fold_indices[0][fold].astype('int')
                train_data_fold = np.squeeze(train_data[fold_index])
                train_labels_fold = np.squeeze(train_labels[fold_index])

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
                        test_input = np.zeros((test_data.shape[0], n_filters[layer - 1], 2, 2))
                        n_batches = test_data.shape[0] / batch_size
                        for batch in xrange(n_batches):
                            print "for batch %d" % batch
                            batch_start = batch * batch_size
                            batch_end = batch_start + batch_size
                            temp = test[layer - 1](test_data[batch_start:batch_end])
                            temp = temp[0]
                            test_input[batch_start:batch_end] = quad_pool(temp)[0]
                        test_input = test_input.reshape(test_input.shape[0], test_input.shape[1] *
                                                        test_input.shape[2] * test_input.shape[3])

                    print "getting training data..."
                    train_input = np.zeros((train_data_fold.shape[0], n_filters[layer - 1], 2, 2))
                    n_batches = train_data_fold.shape[0] / batch_size
                    for batch in xrange(n_batches):
                        print "for batch %d" % batch
                        batch_start = batch * batch_size
                        batch_end = batch_start + batch_size
                        temp = test[layer - 1](train_data_fold[batch_start:batch_end])
                        temp = temp[0]
                        train_input[batch_start:batch_end] = quad_pool(temp)[0]
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
                accuracy['layer' + str(layer)]['fold' + str(fold)] = clf.score(test_input, test_labels[0:examples])
                accuracy_list.append(accuracy['layer' + str(layer)]['fold' + str(fold)])

                # display results and log them
                print("Accuracy of the classifier for fold %d at layer %1d: %0.4f" %
                      (fold, layer, accuracy['layer' + str(layer)]['fold' + str(fold)]))

                log_file = open(directory_name + "/log_test.txt", "a")
                log_file.write(
                    "\nAccuracy of the classifier for fold %d at layer %1d: %0.4f" %
                    (fold, layer, accuracy['layer' + str(layer)]['fold' + str(fold)])
                )
                log_file.close()

            # calculate and print out average accuracy and std
            avg = np.mean(accuracy_list)
            std = np.std(accuracy_list)
            print "The overall accuracy of layer %d: %0.4f +/- (%0.4f)" % (layer, float(avg), float(std))

        # save for aws
        if aws == 'y':
            k.key = directory_name + "/log_test.txt"
            k.set_contents_from_filename(directory_name + "/log_test.txt")

        # save the test results
        savemat('accuracy', accuracy)

if __name__ == '__main__':
    main()
