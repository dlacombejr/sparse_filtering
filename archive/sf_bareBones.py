# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 19:45:32 2015

Sparse Filtering Model

@author: dan
"""

import os
import theano
import visualize
import numpy as np
from init import init_weights
from theano import tensor as t
from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.cluster.vq import whiten


class SparseFilter(object):

    """ Sparse Filtering """

    def __init__(self, w, x):

        """
        Build a sparse filtering model.

        Parameters:
        ----------
        w : ndarray
            Weight matrix randomly initialized.
        x : ndarray (symbolic Theano variable)
            Data for model.
        """

        # assign inputs to sparse filter
        self.w = w
        self.x = x

    def feed_forward(self):

        """ Performs sparse filtering normalization procedure """

        f = t.dot(self.w, self.x)               # initial activation values
        fs = t.sqrt(f ** 2 + 1e-8)              # ensure numerical stability
        l2fs = t.sqrt(t.sum(fs ** 2, axis=1))   # l2 norm of row
        nfs = fs / l2fs.dimshuffle(0, 'x')      # normalize rows
        l2fn = t.sqrt(t.sum(nfs ** 2, axis=0))  # l2 norm of column
        f_hat = nfs / l2fn.dimshuffle('x', 0)   # normalize columns

        return f_hat

    def get_cost_grads(self):

        """ Returns the cost and flattened gradients for the layer """

        cost = t.sum(t.abs_(self.feed_forward()))
        grads = t.grad(cost=cost, wrt=self.w).flatten()

        return cost, grads


def training_functions(data, model, weight_dims):

    """
    Construct training functions for the model.

    Parameters:
    ----------
    data : ndarray
        Training data for unsupervised feature learning.

    Returns:
    -------
    train_fn : list
        Callable training function for L-BFGS.
    """

    # compile the Theano training function
    cost, grads = model.get_cost_grads()
    fn = theano.function(inputs=[], outputs=[cost, grads],
                         givens={model.x: data}, allow_input_downcast=True)

    def train_fn(theta_value):

        """
        Creates a shell around training function for L-BFGS optimization
        algorithm such that weights are reshaped before calling Theano
        training function and outputs of Theano training function are
        converted to float64 for SciPy optimization procedure.

        Parameters:
        ----------
        theta_value : ndarray
            Output of SciPy optimization procedure (vectorized).

        Returns:
        -------
        c : float64
            The cost value for the model at a given iteration.
        g : float64
            The vectorized gradients of all weights
        """

        # reshape the theta value for Theano and convert to float32
        theta_value = np.asarray(theta_value.reshape(weight_dims[0],
                                                     weight_dims[1]),
                                 dtype=theano.config.floatX)

        # assign the theta value to weights
        model.w.set_value(theta_value, borrow=True)

        # get the cost and vectorized grads
        c, g = fn()

        # convert values to float64 for SciPy
        c = np.asarray(c, dtype=np.float64)
        g = np.asarray(g, dtype=np.float64)

        return c, g

    return train_fn


def main():

    # read in the data
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "data", "patches.mat")
    data = loadmat(file_path)['X']

    # preprocess the data
    data -= data.mean(axis=0)
    data = whiten(data)
    data = np.float32(data)

    # define the model variables
    weight_dims = (100, 256)        # network architecture
    w = init_weights(weight_dims)   # random weights
    x = t.fmatrix()                 # symbolic variable for data

    # build model and define training functions
    model = SparseFilter(w, x)
    train_fn = training_functions(data, model, weight_dims)

    # train the model using L-BFGS
    weights = minimize(train_fn, model.w.eval().flatten(),
                       method='L-BFGS-B', jac=True,
                       options={'maxiter': 100, 'disp': True})

    # grab the weights and visualize them
    weights = weights.x.reshape(weight_dims[0], weight_dims[1])
    visualize.drawplots(weights.T, 'y', 'n', 1)

if __name__ == '__main__':
    main()