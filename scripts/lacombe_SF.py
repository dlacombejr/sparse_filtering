# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 01:21:01 2015

@author: dan

==================
 Sparse filtering
==================

"""

import os
import sparse_filtering
import numpy as np
from utilities.visualize import plotCost, drawplots
from scipy.io import loadmat
    

#################### MAIN SCRIPT #########################
    
# load data, normalize, and convert to float32
basepath = os.path.dirname(__file__)
filename = 'patches.mat'
filepath = os.path.join(basepath, "data", filename)
data = loadmat(filepath)['X']
data -= data.mean(axis=0)
data = np.float32(data)

# construct the network
wDims = [[100, 256]]
model = sparse_filtering.network(model_type='SF', wDims=wDims, p=None, 
                 gMat=None, gSize=None, step=None, lr=0.01)
train = model.training_functions(data)

# train the sparse filtering network
maxIter = 100
cost_master = []
for i in range(maxIter):
    cost, w = train[0]()
    cost_master.append(cost)
    print("The cost at iteration %i: %f" %(i, cost))

plotCost(cost_master)
drawplots(w.T) 