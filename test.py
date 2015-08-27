# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 17:06:50 2015

Executable file for testing different sparse filtering types/architectures

@author: dan
"""

import os
import time
import scaling
import argparse
import visualize
import numpy as np
import sparse_filtering as sf
from scipy.io import loadmat, savemat
from scipy.optimize import minimize
from scipy.cluster.vq import whiten
from parse_help import parse_dims


def main():
    
    # parse options from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="SparseFilter", help="the model type")
    parser.add_argument("-c", "--convolution", default="n", help="convolution, yes or no")
    parser.add_argument("-f", "--filename", default="patches.mat", help="the data filename")
    parser.add_argument("-d", "--dimensions", type=int, nargs='+', default=([100, 256]),
                        help="the dimensions of the model: [neurons, input size] or [neurons, length, width]")
    parser.add_argument("-p", "--pool", type=int, nargs='+', default=None, help="pooling dimensions")
    parser.add_argument("-g", "--group", type=int, default=None, help="group size")
    parser.add_argument("-s", "--step", type=int, default=None, help="step size")
    parser.add_argument("-l", "--learn_rate", type=int, default=.01, help="learning rate")
    parser.add_argument("-i", "--iterations", type=int, default=100, help="number of iterations")
    parser.add_argument("-v", "--verbosity", type=int, default=0, help="verbosity: 0 no plot; 1 plots")
    parser.add_argument("-o", "--opt", default="GD", help="optimization method: GD or L-BFGS")
    parser.add_argument("-w", "--whitening", default='n', help="whitening: 'y' or 'n'")
    args = parser.parse_args()
    args.dimensions = parse_dims(args)

    # load in the data and preprocess
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "data", args.filename)
    data = loadmat(file_path)['X']
    if args.filename == 'patches_video.mat':
        video = data
        data = data.reshape(data.shape[0] * data.shape[1], data.shape[2]).T
    if args.convolution == 'n':
        if args.whitening == 'y':
            data -= data.mean(axis=0)
            data = whiten(data)
        elif args.whitening == 'n': 
            data -= data.mean(axis=0)
        data = np.float32(data)
    elif args.convolution == 'y':
        data = np.float32(data.reshape(-1, 1, int(np.sqrt(data.shape[1])), int(np.sqrt(data.shape[1]))))
        data = scaling.LCNinput(data, kernel_shape=9)
    
    # construct the network
    model = sf.Network(model_type=args.model, weight_dims=args.dimensions, p=args.pool,
                       group_size=args.group, step=args.step, lr=0.01, opt=args.opt, c=args.convolution)
    train, outputs = model.training_functions(data)
    
    # train the sparse filtering network
    t = time.time()
    cost_master = []
    for l in xrange(model.n_layers):

        # iterate over training epochs
        if args.opt == 'GD':
            for i in range(args.iterations):
                cost, w = train[l]()
                cost_master.append(cost)
                print("Layer %i cost at iteration %i: %f" % (l + 1, i, cost))

        elif args.opt == 'L-BFGS':
            w = minimize(train[l], model.layers[l].w.eval().flatten(), 
                         method='L-BFGS-B', jac=True,
                         options={'maxiter': args.iterations, 'disp': True})
                         
            if args.convolution == 'n':
                w = w.x.reshape(args.dimensions[0][0], args.dimensions[0][1])
            elif args.convolution == 'y':
                w = w.x.reshape(args.dimensions[0][0], args.dimensions[0][1], 
                                args.dimensions[0][2], args.dimensions[0][3])

    # calculate and display elapsed training time        
    elapsed = time.time() - t
    print('Elapsed training time: %f' % elapsed)
    
    # display figures
    if args.verbosity == 1:
        if args.opt == 'GD':
            visualize.plotCost(cost_master)
            visualize.drawplots(w.T, 'y', args.convolution, 0)
        if args.opt == 'L-BFGS':
            visualize.drawplots(w.T, 'y', args.convolution, 1)
            c, g, f_hat, rec, err = outputs[0]()
            visualize.dispSparseHist(f_hat)
            if args.filename == 'patches_video.mat':
                f_hat = f_hat.T.reshape(video.shape[0], video.shape[1], args.dimensions[0][0])
                visualize.videoCortex(f_hat[0:100, :, :], 'y', args.convolution, 1)
            else:
                visualize.drawplots(f_hat[:, np.random.permutation(f_hat.shape[1])[0:100]], 
                                    'y', args.convolution, 1)
                visualize.drawReconstruction(data[:, 0:100], rec[:, 0:100], 'y', args.convolution, 1)
            print('Reconstructed error: %e' % err)
            
    # save model weights
    weights = dict([('w', model.layers[0].w.eval())])
    activations = dict([('a', f_hat)])
    savemat('weights.mat', weights)
    savemat('activation.mat', activations)

if __name__ == '__main__':
    main()