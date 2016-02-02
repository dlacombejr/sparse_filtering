# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 17:06:50 2015

Executable file for testing different sparse filtering types/architectures

@author: dan
"""

import os
import time
import argparse
import textwrap
import numpy as np
import cPickle as pickle
import sparse_filtering as sf
import utilities.scaling as scaling
import utilities.visualize as visualize
from scipy.io import loadmat, savemat
from scipy.optimize import minimize
from scipy.cluster.vq import whiten
from utilities.parse_help import parse_dims, parse_iter


def main():
    # parse options from the command line
    parser = argparse.ArgumentParser(
        prog='PROG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
        -------------------------------------------------------------------------------------------------------------
        This is a deep neural network architecture for training sparse filters. Example uses:
            $ python test.py
            $ python test.py -m GroupSF -v 1 -g 3 -s 1
            $ python test.py -m ConvolutionalSF -d 16 1 8 8 -v 1 -w y -c y -f CIFAR_data.mat -i 100
            $ python test.py -m ConvolutionalSF ConvolutionalSF -d 16 1 6 6 16 16 4 4 -w y -c y -f CIFAR_data.mat
              -i 100 150 -t y -v 1
        -------------------------------------------------------------------------------------------------------------
        ''')
    )
    parser.add_argument("-m", "--model", default=['SparseFilter'], nargs='+', help="the model type")
    parser.add_argument("-c", "--convolution", default="n", help="convolution, yes or no")
    parser.add_argument("-f", "--filename", default="patches.mat", help="the data filename")
    parser.add_argument("-d", "--dimensions", type=int, nargs='+', default=([100, 256]),
                        help="the dimensions of the model: [neurons, input size] or [neurons, length, width]")
    parser.add_argument("-p", "--pool", type=int, nargs='+', default=None, help="pooling dimensions")
    parser.add_argument("-g", "--group", type=int, default=None, help="group size")
    parser.add_argument("-s", "--step", type=int, default=None, help="step size")
    parser.add_argument("-l", "--learn_rate", type=float, default=.001, help="learning rate")
    parser.add_argument("-i", "--iterations", type=int, nargs='+', default=[100], help="number of iterations")
    parser.add_argument("-v", "--verbosity", type=int, default=0, help="verbosity: 0 no plot; 1 plots")
    parser.add_argument("-o", "--opt", default="GD", help="optimization method: GD or L-BFGS")
    parser.add_argument("-w", "--whitening", default='n', help="whitening: 'y' or 'n'")
    parser.add_argument("-t", "--test", default='n', help="test classification performance: 'y' or 'n'")
    parser.add_argument("-a", "--channels", type=int, default=1, help="number of channels in data")
    parser.add_argument("-e", "--examples", type=int, default=None, help="number of training examples")
    parser.add_argument("-b", "--batch_size", type=int, default=1000, help="number of examples in [mini]batch")
    parser.add_argument("-z", "--aws", default='n', help="run on aws: 'y' or 'n'")
    parser.add_argument("-r", "--random", default='n', help="type of batches: random = 'y'")
    args = parser.parse_args()
    args.dimensions = parse_dims(args)
    args.iterations = parse_iter(args)

    ''' =================================== Load in the data =================================== '''

    # load in data
    print "loading data..."
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "data", args.filename)
    data = loadmat(file_path)['X']

    # reshape and preprocess data
    print "pre-processing data ..."
    video = None
    if args.filename == 'patches_video.mat':
        video = data
        data = data.reshape(data.shape[0] * data.shape[1], data.shape[2]).T

    if args.convolution == 'n':
        if args.whitening == 'y':
            data -= data.mean(axis=0)
            data = whiten(data.T).T
        elif args.whitening == 'n' and args.channels == 1:
            data -= data.mean(axis=0)
        # elif args.whitening == 'n' and args.channels == 3:
        # data = np.float32(data)
        data = np.float32(data.T)

    elif args.convolution == 'y':

        if args.filename == 'kyotoData.mat':
            data = np.float32(data.reshape(-1, 1, int(np.sqrt(data.shape[1])), int(np.sqrt(data.shape[1]))))
            data = scaling.LCNinput(data, kernel_shape=9)

        elif args.filename == 'CIFAR_data.mat':
            data = np.float32(data.reshape(-1, 1, int(np.sqrt(data.shape[1])), int(np.sqrt(data.shape[1]))))
            data = scaling.LCNinput(data, kernel_shape=5)
            data = data[0:args.examples, :, :, :]

        elif args.filename == 'STL_10.mat' or args.filename == 'Lenna.mat':
            data = np.float32(data.reshape(-1, 3, int(np.sqrt(data.shape[1] / 3)), int(np.sqrt(data.shape[1] / 3))))
            data = data[0:args.examples, :, :, :]
            args.channels = data.shape[1]
            for channel in range(args.channels):
                data[:, channel, :, :] = np.reshape(scaling.LCNinput(data[:, channel, :, :].
                                                                     reshape((data.shape[0], 1,
                                                                              data.shape[2],
                                                                              data.shape[3])),
                                                                     kernel_shape=9), (
                                                    data.shape[0],
                                                    data.shape[2],
                                                    data.shape[3]))

    # assert that batch size is valid and get number of batches
    n_batches, rem = divmod(data.shape[0], args.batch_size)
    assert rem == 0

    # other assertions
    assert len(args.model) == len(args.iterations)
    if args.model[0] == 'GroupSF' or args.model[0] == 'GroupConvolutionalSF':
        assert args.group is not None
        assert args.step is not None

    # assert that the number of neurons in each layer is a perfect square
    for layer in xrange(len(args.dimensions)):
        assert np.sqrt(args.dimensions[layer][0]) % np.floor(np.sqrt(args.dimensions[layer][0])) == 0

    ''' ============================= Build and train the network ============================= '''

    # construct the network
    print "building model..."
    model = sf.Network(
        model_type=args.model, weight_dims=args.dimensions, p=args.pool, group_size=args.group,
        step=args.step, lr=args.learn_rate, opt=args.opt, c=args.convolution, test=args.test,
        batch_size=args.batch_size, random=args.random, weights=None
    )  # TODO: custom learning rates for each layer

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
        if args.opt == 'GD':
            for epoch in xrange(args.iterations[l]):

                # go though [mini]batches
                for batch_index in xrange(n_batches):

                    c, w = train[l](index=batch_index)
                    cost_layer.append(c)
                    print("Layer %i cost at epoch %i and batch %i: %f" % (l + 1, epoch, batch_index, c))

        elif args.opt == 'L-BFGS':
            w = minimize(train[l], model.layers[l].w.eval().flatten(),
                         method='L-BFGS-B', jac=True,
                         options={'maxiter': args.iterations[l], 'disp': True})

            if args.convolution == 'n':
                w = w.x.reshape(args.dimensions[0][0], args.dimensions[0][1])
            elif args.convolution == 'y':
                w = w.x.reshape(args.dimensions[0][0], args.dimensions[0][1],
                                args.dimensions[0][2], args.dimensions[0][3])

        # add layer cost and weights to the dictionaries
        cost['layer' + str(l)] = cost_layer
        weights['layer' + str(l)] = w

    # calculate and display elapsed training time        
    elapsed = time.time() - t
    print('Elapsed training time: %f' % elapsed)

    # create sub-folder for saved model
    if args.aws == 'n':
        directory_format = "./saved/%4d-%02d-%02d_%02dh%02dm%02ds"
        directory_name = directory_format % time.localtime()[0:6]
        os.mkdir(directory_name)
    elif args.aws == 'y':
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
    if args.aws == 'y':
        k.key = full_path
        k.set_contents_from_filename(full_path)
        os.remove(full_path)

    # save weights separately
    savemat(directory_name + '/weights.mat', weights)
    if args.aws == 'y':
        k.key = directory_name + '/weights.mat'
        k.set_contents_from_filename(directory_name + '/weights.mat')
        os.remove(directory_name + '/weights.mat')

    # save the cost functions
    savemat(directory_name + '/cost.mat', cost)
    if args.aws == 'y':
        k.key = directory_name + '/cost.mat'
        k.set_contents_from_filename(directory_name + '/cost.mat')
        os.remove(directory_name + '/cost.mat')

    # create log file
    log_file = open(directory_name + "/log.txt", "wb")  # todo: create log file by looping through args
    # for arg in args:
    #     log_file.write(
    #         args.
    #     )
    for m in range(len(args.model)):
        log_file.write(
            "Model layer %d: \n model:%s \n dimensions:%4s \n iterations:%3d \n" % (m,
                                                                                    args.model[m],
                                                                                    args.dimensions[m],
                                                                                    args.iterations[m])
        )
        if args.model == 'GroupSF' or args.model == 'GroupConvolutionalSF':
            log_file.write(
                " Groups: %d \n Step: %d" % (args.group, args.step)
            )
        ex = data.shape[0]
        if args.examples is not None:
            ex = args.examples

    log_file.write(
        " Data-set: %s \n Examples: %6d \n Whitened: %s" % (args.filename, ex, args.whitening)
    )
    log_file.write('\nElapsed training time: %f' % elapsed)
    log_file.close()
    if args.aws == 'y':
        k.key = directory_name + "/log.txt"
        k.set_contents_from_filename(directory_name + "/log.txt")
        os.remove(directory_name + "/log.txt")

    ''' =============================== Verbosity Options ===================================== '''

    # get variables and saves
    if args.verbosity >= 1:

        # # get variables of interest
        # activations_norm = {}
        # activations_raw = {}
        # activations_shuffled = {}
        # reconstruction = {}
        # error_recon = {}
        # pooled = {}

        # for l in xrange(len(args.dimensions)):

            # activations_norm['layer' + str(l)] = {}
            # activations_raw['layer' + str(l)] = {}
            # activations_shuffled['layer' + str(l)] = {}
            # reconstruction['layer' + str(l)] = {}
            # error_recon['layer' + str(l)] = {}
            # pooled['layer' + str(l)] = {}

        for batch in xrange(n_batches):

            # get variables of interest
            activations_norm = {}
            activations_raw = {}
            activations_shuffled = {}
            reconstruction = {}
            error_recon = {}
            pooled = {}

            # f_hat, rec, err, f_hat_shuffled, f, p = outputs[l]()
            begin = batch * args.batch_size
            end = begin + args.batch_size
            f_hat, rec, err, f_hat_shuffled, f, p = outputs[model.n_layers - 1](data[begin:end])

            # activations_norm['layer' + str(l)]['batch' + str(batch)] = f_hat
            # activations_raw['layer' + str(l)]['batch' + str(batch)] = f
            # activations_shuffled['layer' + str(l)]['batch' + str(batch)] = f_hat_shuffled
            # reconstruction['layer' + str(l)]['batch' + str(batch)] = err
            # error_recon['layer' + str(l)]['batch' + str(batch)] = rec
            # pooled['layer' + str(l)]['batch' + str(batch)] = p

            # define [mini]batch title
            batch_title = 'layer' + str(l) + '_batch' + '%03d' % batch

            # define norm and raw file names
            norm_file_name = directory_name + '/activations_norm_' + batch_title + '.mat'
            raw_file_name = directory_name + '/activation_raw_' + batch_title + '.mat'

            activations_norm[batch_title] = f_hat
            activations_raw[batch_title] = f
            activations_shuffled[batch_title] = f_hat_shuffled
            reconstruction[batch_title] = err
            error_recon[batch_title] = rec
            pooled[batch_title] = p

            # save model as well as weights and activations separately
            savemat(norm_file_name, activations_norm)
            # savemat(raw_file_name, activations_raw)

            if args.aws == 'y':

                k.key = norm_file_name
                k.set_contents_from_filename(norm_file_name)
                os.remove(norm_file_name)

                # k.key = raw_file_name
                # k.set_contents_from_filename(raw_file_name)
                # os.remove(raw_file_name)

        # savemat(directory_name + '/weights.mat', weights)
        # if args.aws == 'y':
        #     k.key = directory_name + '/weights.mat'
        #     k.set_contents_from_filename(directory_name + '/weights.mat')
        #     os.remove(directory_name + '/weights.mat')

        #     # f_hat, rec, err, f_hat_shuffled, f, p = outputs[l]()
        #     f_hat, rec, err, f_hat_shuffled, f, p = outputs[l](data[0:args.batch_size])
        #
        #     activations_norm['layer' + str(l)] = f_hat
        #     activations_raw['layer' + str(l)] = f
        #     activations_shuffled['layer' + str(l)] = f_hat_shuffled
        #     reconstruction['layer' + str(l)] = err
        #     error_recon['layer' + str(l)] = rec
        #     pooled['layer' + str(l)] = p
        #
        # # save model as well as weights and activations separately
        # savemat(directory_name + '/weights.mat', weights)
        # savemat(directory_name + '/activations_norm.mat', activations_norm)
        # savemat(directory_name + '/activation_raw.mat', activations_raw)

    # output helper file for concatenating activations
    helper = {'batches': n_batches, 'output_size': f_hat.shape}
    helper_file_name = directory_name + '/helper.mat'
    savemat(helper_file_name, helper)
    if args.aws == 'y':
        k.key = helper_file_name
        k.set_contents_from_filename(helper_file_name)
        os.remove(helper_file_name)

    # get data if not on AWS
    if args.aws == 'n':
        f_hat, rec, err, f_hat_shuffled, f, p = outputs[model.n_layers - 1](data)
        activations_norm = {"layer0": f_hat}

    # display figures
    if args.verbosity == 2:

        # if GD, plot the cost function over time
        if args.opt == 'GD':
            visualize.plotCost(cost)

        # visualize the receptive fields of the first layer
        visualize.drawplots(weights['layer0'].T, color='gray', convolution=args.convolution,
                            pad=0, examples=None, channels=args.channels)

        # visualize the distribution of lifetime and population sparseness
        for l in xrange(len(args.dimensions)):
            layer = 'layer' + str(l)
            if args.convolution == 'n':
                visualize.dispSparseHist(activations_norm[layer], l)
            elif args.convolution == 'y':
                visualize.dispSparseHist(activations_shuffled[layer].reshape(args.dimensions[l][0],
                                                                             data.shape[0] *
                                                                             activations_shuffled[layer].shape[2] *
                                                                             activations_shuffled[layer].shape[3]),
                                         layer=l)

        # visualize the distribution of activity across the "cortical sheet" and reconstruction
        if args.filename == 'patches_video.mat':
            f_hat = activations_norm['layer0'].T.reshape(video.shape[0], video.shape[1], args.dimensions[0][0])
            visualize.videoCortex(f_hat[0:100, :, :], 'y', args.convolution, 1)
        else:
            visualize.drawplots(activations_norm['layer0'], color='gray', convolution=args.convolution,
                                pad=1, examples=100)

        # # visualize reconstruction capabilities
        # if args.convolution == 'n':
        #     visualize.drawReconstruction(data[:, 0:100], error_recon['layer0'][:, 0:100], 'y', args.convolution, 1)
        # elif args.convolution == 'y':
        #     visualize.convolutional_reconstruction(data[0, :, :, :], activations_raw['layer0'], weights['layer0'],
        #                                            color='gray', convolution=args.convolution)
        # print('Reconstructed error: %e' % reconstruction['layer0'])

        # additional visualizations for convolutional network
        if args.convolution == 'y':

            dim = activations_raw['layer0'].shape[2]

            # visualize an example of a convolved image
            visualize.visualize_convolved_image(activations_raw['layer0'], dim=dim)
            # print activations_raw['layer0']

            # visualize max-pooled activations and LCN output
            visualize.visualize_convolved_image(pooled['layer0'][0, :, :, :].reshape(1,
                                                                                     pooled['layer0'].shape[1],
                                                                                     pooled['layer0'].shape[2],
                                                                                     pooled['layer0'].shape[3]),
                                                dim=dim / 2)

            # visualize an example of a LCNed convolved image after max pooling
            # temp = activations_raw['layer0']    #[0, :, :, :]
            temp = pooled['layer0']    #[0, :, :, :]
            # print temp.shape
            for i in range(temp.shape[1]):
                temp[0, i, :, :] = scaling.LCNinput(temp[0, i, :, :].reshape((1, 1, dim / 2, dim / 2)), kernel_shape=5)
            # temp = scaling.LCNinput(temp, kernel_shape=5)
            visualize.visualize_convolved_image(temp, dim=dim / 2)
            # print temp

    ''' ================================ Test the Model ======================================= '''

    # test the model if evaluating classification performance
    if args.test == 'y':

        from sklearn import svm
        from sklearn.metrics import confusion_matrix

        train_labels = loadmat(file_path)['y']

        file_path = os.path.join(base_path, "data", "CIFAR_test.mat")
        test_data = loadmat(file_path)['X']
        test_labels = loadmat(file_path)['y']

        # reshape and normalize the data
        if args.convolution == 'y':
            test_data = np.float32(test_data.reshape(-1, 1, int(np.sqrt(test_data.shape[1])),
                                                     int(np.sqrt(test_data.shape[1]))))
            test_data = scaling.LCNinput(test_data, kernel_shape=5)
            test_data = test_data[0:args.examples, :, :, :]

        # get SVM test results for pixels to last layer
        train_input = None
        for layer in range(model.n_layers + 1):

            # pixel inputs
            if layer == 0:

                test_input = test_data.reshape(test_data.shape[0], test_data.shape[1] *
                                               test_data.shape[2] * test_data.shape[3])

                train_input = data.reshape(data.shape[0], data.shape[1] *
                                           data.shape[2] * data.shape[3])

            # hidden layers
            elif layer > 0:

                # get the output of the current layer in the model given the training / test data and then reshape
                # TODO: use raw output as training and testing data?
                test_input = test[layer - 1](test_data[0:args.batch_size])
                test_input = test_input[0].reshape(test_input[0].shape[0], test_input[0].shape[1] *
                                                   test_input[0].shape[2] * test_input[0].shape[3])

                train_input = activations_norm['layer' + str(layer - 1)]
                train_input = train_input.reshape(train_input.shape[0], train_input.shape[1] *
                                                  train_input.shape[2] * train_input.shape[3])

            # train linear support vector machine
            clf = svm.SVC(kernel="linear").fit(train_input, np.ravel(train_labels[0:args.examples]))

            # get predictions from SVM and calculate accuracy
            predictions = clf.predict(test_input)
            accuracy = clf.score(test_input, test_labels[0:args.examples])

            # display results and log them
            print("Accuracy of the classifier at layer %1d: %0.4f" % (layer, accuracy))
            cm = confusion_matrix(test_labels[0:args.examples], predictions)
            log_file = open(directory_name + "/log.txt", "a")
            log_file.write(
                "\nAccuracy of the classifier at layer %1d: %0.4f" % (layer, accuracy)
            )
            log_file.close()

    # visualize the confusion matrix
    if args.test == 'y' and args.verbosity == 2:

        import pylab as pl

        pl.imshow(cm, interpolation='nearest')
        pl.title('Confusion Matrix for Network')
        pl.colorbar()
        pl.ylabel('True Label')
        pl.xlabel('Predicted Label')
        pl.show()


if __name__ == '__main__':
    main()