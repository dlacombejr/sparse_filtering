# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:50:26 2015

@author: dan

==================
 Convolutional Deep Sparse Filtering
==================

"""

import os
import scaling
import sparse_filtering
import numpy as np
from visualize import plotCost
from scipy.io import loadmat

    
def weightedSum(top_layer, bottom_layer):
    
    dim = bottom_layer.shape[2]    
    w = bottom_layer.reshape((bottom_layer.shape[0], dim * dim))
    rf = np.zeros(shape = (top_layer.shape[0], dim * 2, dim * 2))
                           
    for kernel in range(top_layer.shape[0]):
        
        for position in range(top_layer.shape[2]):
            start = position * dim
            end = (position + 1) * dim
            rf[kernel, start:end, 0:dim] = np.multiply(w.T, top_layer[kernel, :, position, 0]).sum(axis=1).reshape((dim,dim))
            
        for position in range(top_layer.shape [3]):
            start = position * dim
            end = ((position + 1) * dim)
            rf[kernel, start:end, dim:(dim*2)] = np.multiply(w.T, top_layer[kernel, :, position, 1]).sum(axis=1).reshape((dim,dim))
    
    return rf
    
def displayUpdate(model, l, saved_weighted_input, num_viz=9):
    
    w = None
    
    if l == 0:
        
        weightedInput = None    # no weighted input for first layer
        w = model.sparse_filters[l].w.eval()
        w = w.reshape((w.shape[0], w.shape[2], w.shape[3]))
    
    
    if l == 1:  #if second layer, weighted sum of first layer
        
        top_layer = model.sparse_filters[l].w.eval()
        bottom_layer = model.sparse_filters[l - 1].w.eval()
        
        weightedInput = weightedSum(top_layer, bottom_layer)
        
    elif l > 1: #if higher than second layer, weighted sum of saved weighted input
        
        top_layer = model.sparse_filters[l].w.eval()
        bottom_layer = saved_weighted_input

        weightedInput = weightedSum(top_layer, bottom_layer)
        
#    displayData(w, weightedInput, num_viz)
    return weightedInput
    
    

###############################################################################
############################### MAIN SCRIPT ###################################
###############################################################################

# load in the data and preprocess
basepath = os.path.dirname(__file__)
filename = 'kyotoData.mat'
filepath = os.path.join(basepath, "data", filename)
data = loadmat(filepath)['X']
data = np.float32(data.reshape(-1, 1, 512, 512))
data = scaling.LCNinput(data, kernel_shape=9)

# construct the Deep Sparse Filtering network
kernels = [[25, 10, 10]]#, [30, 5, 5]] #[[9, 10, 10]]
pooling = [[2]] #, [2]]
model = sparse_filtering.network(model_type='convSF', wDims=kernels, p=None, 
                 gMat=None, gSize=None, step=None, lr=0.01)
train = model.training_functions(data)

# train the sparse filtering network
maxIter = 10
disp_iter = 0
weightedInput = None
saved_weighted_input = []
master_weights = [None]*model.n_layers

# iterate over all layers
for l in xrange(model.n_layers):

    
    # iterate over training epochs
    cost_master = []
    for i in range(maxIter):
        cost, w = train[l]()
        cost_master.append(cost)
        print("Layer %i cost at iteration %i: %f" %(l+1, i, cost))
    
    plotCost(cost_master)
