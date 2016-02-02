# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 21:25:20 2015

Connection matrices for group sparseness and lateral inhibition

@author: dan
"""

import scipy
import numpy as np
from sklearn.preprocessing import normalize
from scipy.stats import norm
import matplotlib.pyplot as plt


def groupMat(neurons, gSize, step):
    
    dim = np.sqrt(neurons)
    covers = dim - (gSize - step)

    if gSize == step:
        covers = dim / gSize
    
    master = np.zeros((covers * covers, dim * dim))
    c = 0
    for x in range(int(covers - 1)):
        for y in range(int(covers - 1)):
            temp = np.zeros((dim, dim))
#            temp[y * gSize:(y + 1) * gSize, x * gSize:(x + 1) * gSize] = np.ones((gSize, gSize))
            temp[y * step:(y * step) + gSize,
                 x * step:(x * step) + gSize] = np.ones((gSize, gSize))
            if gSize == step:
                temp[y * step:(y * step) + gSize,
                        x * step:(x * step) + gSize] = np.ones((gSize, gSize))

            master[c, :] = temp.flatten()
            c += 1
    return master
    
    
def gMatToroidal(neurons, gSize, step, centered='n'):
    
    dim = int(np.sqrt(neurons))
    temp = np.zeros((dim, dim))
    
    if centered == 'n': 
        temp[dim - gSize:, dim - gSize:] = 1
    elif centered == 'y':
        temp[dim - gSize:, dim - gSize:] = 1
        temp = np.roll(temp, 1, axis=0)
        temp = np.roll(temp, 1, axis=1)
    
    master = np.zeros((neurons, neurons))
    c = 0
    for x in range(int(dim)):
        for y in range(int(dim)):
            s = np.roll(temp, x + 1, axis=0)
            s = np.roll(s, y + 1, axis=1)
            master[c, :] = s.flatten()
            c += 1
            
    return master
    
    
def distMat(neurons, d=None, kind='euclidean', inverted='n'):
    
    master = np.zeros((neurons, neurons))
    dim = np.sqrt(neurons)
    
    for n in range(neurons):
    
        temp = []
        for i in range(int(dim)):
            for j in range(int(dim)):
                temp.append([i, j])
        
        center = temp[n]
        
        dist = scipy.spatial.distance.cdist(temp, np.atleast_2d(center), kind)
    
        master[n, :] = dist.T

    if inverted == 'y':
        master[master >= d] = 0
        master = 1/master
        master[master == np.inf] = 0

        master = normalize(master, norm='l1')
    
    return master
