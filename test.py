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
            $ python test.py -m ConvolutionalSF -d 16 1 6 6 16 1 4 4 -w y -c y -f CIFAR_data.mat -i 100 150 -t y -v 1

        In the convolutional case, the extra "1" is added automatically for broadcasting.
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
    parser.add_argument("-l", "--learn_rate", type=int, default=.01, help="learning rate")
    parser.add_argument("-i", "--iterations", type=int, nargs='+', default=[100], help="number of iterations")
    parser.add_argument("-v", "--verbosity", type=int, default=0, help="verbosity: 0 no plot; 1 plots")
    parser.add_argument("-o", "--opt", default="GD", help="optimization method: GD or L-BFGS")
    parser.add_argument("-w", "--whitening", default='n', help="whitening: 'y' or 'n'")
    parser.add_argument("-t", "--test", default='n', help="test classification performance: 'y' or 'n")
    parser.add_argument("-a", "--channels", type=int, default=1, help="number of channels in data")
    args = parser.parse_args()
    args.dimensions = parse_dims(args)
    args.iterations = parse_iter(args)

    ''' =================================== Load in the data =================================== '''

    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "data", args.filename)
    data = loadmat(file_path)['X']

    video = None
    if args.filename == 'patches_video.mat':
        video = data
        data = data.reshape(data.shape[0] * data.shape[1], data.shape[2]).T

    if args.convolution == 'n':
        if args.whitening == 'y':
            data -= data.mean(axis=0)
            data = whiten(data)
        elif args.whitening == 'n' and args.channels == 1:
            data -= data.mean(axis=0)
        # elif args.whitening == 'n' and args.channels == 3:
        data = np.float32(data)

    elif args.convolution == 'y':

        if args.filename == 'kyotoData.mat':
            data = np.float32(data.reshape(-1, 1, int(np.sqrt(data.shape[1])), int(np.sqrt(data.shape[1]))))
            data = scaling.LCNinput(data, kernel_shape=9)

        elif args.filename == 'CIFAR_data.mat':
            data = np.float32(data.reshape(-1, 1, int(np.sqrt(data.shape[1])), int(np.sqrt(data.shape[1]))))
            data = scaling.LCNinput(data, kernel_shape=5)
            data = data[0:1000, :, :, :]

        elif args.filename == 'STL_10.mat':
            data = np.float32(data.reshape(-1, 3, int(np.sqrt(data.shape[1] / 3)), int(np.sqrt(data.shape[1] / 3))))
            data = data[0:100, :, :, :]
            for channel in range(data.shape[1]):
                data[:, channel, :, :] = np.reshape(scaling.LCNinput(data[:, channel, :, :].
                                                                     reshape((data.shape[0], 1,
                                                                              data.shape[2],
                                                                              data.shape[3])),
                                                                     kernel_shape=9), (
                                                    data.shape[0],
                                                    data.shape[2],
                                                    data.shape[3]))

    ''' ============================= Build and train the network ============================= '''

    # construct the network
    model = sf.Network(model_type=args.model, weight_dims=args.dimensions, p=args.pool, group_size=args.group,
                       step=args.step, lr=0.01, opt=args.opt, c=args.convolution, test=args.test)

    # compile the training, output, and test functions for the network
    train, outputs, test = model.training_functions(data)

    # train the sparse filtering network
    t = time.time()
    cost = {}
    weights = {}
    for l in xrange(model.n_layers):

        cost_layer = []
        w = None

        # iterate over training epochs
        if args.opt == 'GD':
            for i in range(args.iterations[l]):
                c, w = train[l]()
                cost_layer.append(c)
                print("Layer %i cost at iteration %i: %f" % (l + 1, i, c))

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

    # save the model for later use
    pickle.dump(model, open('saved/model.pkl', 'w'), pickle.HIGHEST_PROTOCOL)

    ''' =============================== Verbosity Options ===================================== '''

    # display figures
    if args.verbosity == 1:

        # get variables of interest
        activations_norm = {}
        activations_raw = {}
        activations_shuffled = {}
        reconstruction = {}
        error_recon = {}
        pooled = {}

        for l in xrange(len(args.dimensions)):

            f_hat, rec, err, f_hat_shuffled, f, p = outputs[l]()

            activations_norm['layer' + str(l)] = f_hat
            activations_raw['layer' + str(l)] = f
            activations_shuffled['layer' + str(l)] = f_hat_shuffled
            reconstruction['layer' + str(l)] = err
            error_recon['layer' + str(l)] = rec
            pooled['layer' + str(l)] = p

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

            dim = 28

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

        # save model as well as weights and activations separately
        savemat('saved/weights.mat', weights)
        savemat('saved/activations_norm.mat', activations_norm)
        savemat('saved/activation_raw.mat', activations_raw)

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
            test_data = test_data[0:1000, :, :, :]

        # get the output of the last layer in the model given the training / test data and then reshape
        # TODO: use raw output as training and testing data?
        test_data = test[model.n_layers - 1](test_data)
        test_data = test_data[0].reshape(test_data[0].shape[0], test_data[0].shape[1] *
                                         test_data[0].shape[2] * test_data[0].shape[3])

        train_data = activations_norm['layer' + str(model.n_layers - 1)]
        train_data = train_data.reshape(train_data.shape[0], train_data.shape[1] *
                                        train_data.shape[2] * train_data.shape[3])

        # train linear support vector machine
        clf = svm.SVC(kernel="linear").fit(train_data, np.ravel(train_labels[0:1000]))

        # get predictions from SVM and calculate accuracy
        predictions = clf.predict(test_data)
        accuracy = clf.score(test_data, test_labels[0:1000])

        # display results
        print("Accuracy of the classifier: %0.4f" % accuracy)
        cm = confusion_matrix(test_labels[0:1000], predictions)

        import pylab as pl
        pl.imshow(cm, interpolation='nearest')
        pl.title('Confusion Matrix for Network')
        pl.colorbar()
        pl.ylabel('True Label')
        pl.xlabel('Predicted Label')
        pl.show()


if __name__ == '__main__':
    main()