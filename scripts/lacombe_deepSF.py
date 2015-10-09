# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 14:52:29 2015

@author: dan

==================
 Deep Sparse Filtering
==================

"""

import os
import theano
from theano import tensor as T
from scipy.io import loadmat
import numpy as np
import pylab as pl


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))


def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    acc = theano.shared(params.get_value() * 0.)
    acc_new = rho * acc + (1 - rho) * grads ** 2
    gradient_scaling = T.sqrt(acc_new + epsilon)
    grads = grads / gradient_scaling
    updates.append((acc, acc_new))
    updates.append((params, params - lr * grads))
    return updates

    
def displayData(x, weightedInput, num_viz=25):
    
    if weightedInput is not None:
        x = weightedInput
    
    pl.figure()
    dim = np.sqrt(x.shape[1])
    for i in range(num_viz):
        reshaped = np.reshape(x[i, :], (dim, dim))
        pl.subplot(np.sqrt(num_viz), np.sqrt(num_viz), i+1)
        pl.imshow(reshaped, interpolation='nearest') #,cmap=pl.cm.gray)
        pl.xticks([])
        pl.yticks([])
    pl.show()   
    
    
def plotCost(cost):
    pl.plot(cost)
    pl.title("Running Cost Function")
    pl.xlabel("Iteration")
    pl.ylabel("Cost")
    
    
def weightedSum(top_layer, bottom_layer, top_size, bottom_size, n_ins):
    rf = np.zeros(shape = (top_size[0], n_ins))
    for i in range(top_size[0]):
        w = top_layer[i, :].eval()
        w_broad = np.tile(w, (bottom_size[1], 1))
        weighted = np.multiply(bottom_layer, w_broad.T)
        summed = sum(weighted)
        rf[i, :] = summed
    return rf
    
def displayUpdate(dsf, l, hidden_layers_sizes, n_ins, saved_weighted_input):
    
    weightedInput = None    # no weighted input for first layer
    
    if l == 1:  #if second layer, weighted sum of first layer
        
        top_layer = dsf.sparse_filters[l].w
        bottom_layer = dsf.sparse_filters[l - 1].w.eval()
        bottom_size = np.array([hidden_layers_sizes[l - 1], n_ins])
        top_size = np.array([hidden_layers_sizes[l], hidden_layers_sizes[l - 1]])
        
        weightedInput = weightedSum(top_layer, 
                                    bottom_layer,
                                    top_size,
                                    bottom_size, 
                                    n_ins)
        
    elif l > 1: #if higher than second layer, weighted sum of saved weighted input
        
        top_layer = dsf.sparse_filters[l].w
        bottom_layer = saved_weighted_input
        bottom_size = np.array([hidden_layers_sizes[l - 1], n_ins])
        top_size = np.array([hidden_layers_sizes[l], hidden_layers_sizes[l - 1]])
        
        weightedInput = weightedSum(top_layer, 
                                    bottom_layer,
                                    top_size,
                                    bottom_size, 
                                    n_ins)
        
    displayData(w, weightedInput)
    return weightedInput
    
    
class SF(object):
    
    def __init__(self, input_size, hidden_layers_size, layer_input, w=None):

        if w is None:
            self.w = init_weights((hidden_layers_size, input_size))  
        
        self.input = layer_input
        
    def get_cost_updates(self):

        F = T.dot(self.w, self.input)
        Fs = T.sqrt(F**2 + 1e-8)
        L2Fs = (Fs**2).sum(axis=[1])
        L2Fs = T.sqrt(L2Fs)
        NFs = Fs/L2Fs.dimshuffle(0, 'x')
        L2Fn = (NFs**2).sum(axis=[0])
        L2Fn = T.sqrt(L2Fn)
        self.Fhat = NFs/L2Fn.dimshuffle('x', 0)        
        
        cost = T.sum(T.abs_(self.Fhat))
        updates = RMSprop(cost, self.w, lr=.001)
        
        return cost, updates
    
    def get_weights(self):
        
        weights = self.w
        return weights
        
    def feedForward(self):
        
        F = T.dot(self.w, self.input)
        Fs = T.sqrt(F**2 + 1e-8)
        L2Fs = (Fs**2).sum(axis=[1])
        L2Fs = T.sqrt(L2Fs)
        NFs = Fs/L2Fs.dimshuffle(0, 'x')
        L2Fn = (NFs**2).sum(axis=[0])
        L2Fn = T.sqrt(L2Fn)
        out = NFs/L2Fn.dimshuffle('x', 0)
        
        return out


class deepSF(object):
    
    def __init__(self, n_ins=256, hidden_layers_sizes=[300, 200]):
        
        self.sparse_filters = []
        self.n_layers = len(hidden_layers_sizes)
        self.X = T.fmatrix()
        
        for i in xrange(self.n_layers):
            
            if i == 0:
                input_size = n_ins
                layer_input = self.X
            else:
                input_size = hidden_layers_sizes[i - 1]
                layer_input = self.sparse_filters[i - 1].feedForward()
            
            sparse_filter = SF(input_size, hidden_layers_sizes[i], layer_input)
            self.sparse_filters.append(sparse_filter)

    def training_functions(self, data):       
        
        train_fns = []
        for sf in self.sparse_filters:
            
            cost, updates = sf.get_cost_updates()
            w = sf.get_weights()
            fn = theano.function([], outputs=[cost, w], updates=updates, 
                                 givens={self.X: data})
            train_fns.append(fn)
            
        return train_fns

###############################################################################
############################### MAIN SCRIPT ###################################
###############################################################################
    
# load data, normalize, and convert to float32
basepath = os.path.dirname(__file__)
filename = 'patches.mat'
filepath = os.path.join(basepath, "data", filename)
data = loadmat(filepath)['data']
data -= data.mean(axis=0)
data = np.float32(data)

# construct the Deep Sparse Filtering network
n_ins=256
hidden_layers_sizes= [100] #[300, 300, 300]
dsf = deepSF(n_ins=n_ins, hidden_layers_sizes=hidden_layers_sizes)
train = dsf.training_functions(data)

# train the sparse filtering network
maxIter = 100
disp_iter = 1
weightedInput = None
saved_weighted_input = []
master_weights = [None]*dsf.n_layers
# iterate over all layers
for l in xrange(dsf.n_layers):
    
    # assign weightedInput to saved matrix if beyond second layer
    if l > 1:
        saved_weighted_input = weightedInput
    
    # iterate over training epochs
    cost_master = []
    for i in range(maxIter):
        cost, w = train[l]()
        cost_master.append(cost)
        print("Layer %i cost at iteration %i: %f" %(l+1, i, cost))
        plotCost(cost_master)
        if disp_iter == 1:  # display subset of weights if this option is on
            weightedInput = displayUpdate(dsf, l, hidden_layers_sizes, n_ins, saved_weighted_input)
    
    # save the weights to master and visualize cost function and weights    
    master_weights[l] = w
    weightedInput = displayUpdate(dsf, l, hidden_layers_sizes, n_ins, saved_weighted_input)
    displayData(master_weights[l], weightedInput, num_viz=100)
    