# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 19:22:01 2015

Scaling module:

Local Contrast Normalization

@author: dan
"""

import theano
import numpy as np
from theano.tensor.nnet.conv import conv2d
from theano import tensor as T
from pylearn2.utils import sharedX
from pylearn2.datasets.preprocessing import gaussian_filter


def LCN(data, kernel_shape):
    
    X = T.ftensor4()
    filter_shape = (1, 1, kernel_shape, kernel_shape)
    filters = sharedX(gaussian_filter(kernel_shape).reshape(filter_shape))
    
    convout = conv2d(X, filters=filters, border_mode='full')
    
    # For each pixel, remove mean of 9x9 neighborhood
    mid = int(np.floor(kernel_shape/ 2.))
    centered_X = X - convout[:,:,mid:-mid,mid:-mid]
    
    # Scale down norm of 9x9 patch if norm is bigger than 1
    sum_sqr_XX = conv2d(T.sqr(X), filters=filters, border_mode='full')
    
    denom = T.sqrt(sum_sqr_XX[:,:,mid:-mid,mid:-mid])
    per_img_mean = denom.mean(axis = [2,3])
    divisor = T.largest(per_img_mean.dimshuffle(0,1, 'x', 'x'), denom)
    
    new_X = centered_X / T.maximum(1., divisor)
    new_X = new_X[:,:,mid:-mid, mid:-mid]
    
    return new_X
    

def LCNinput(data, kernel_shape):
    
    X = T.ftensor4()
    filter_shape = (1, 1, kernel_shape, kernel_shape)
    filters = sharedX(gaussian_filter(kernel_shape).reshape(filter_shape))
    
    convout = conv2d(X, filters=filters, border_mode='full')
    
    # For each pixel, remove mean of 9x9 neighborhood
    mid = int(np.floor(kernel_shape/ 2.))
    centered_X = X - convout[:,:,mid:-mid,mid:-mid]
    
    # Scale down norm of 9x9 patch if norm is bigger than 1
    sum_sqr_XX = conv2d(T.sqr(X), filters=filters, border_mode='full')
    
    denom = T.sqrt(sum_sqr_XX[:,:,mid:-mid,mid:-mid])
    per_img_mean = denom.mean(axis = [2,3])
    divisor = T.largest(per_img_mean.dimshuffle(0,1, 'x', 'x'), denom)
    
    new_X = centered_X / T.maximum(1., divisor)
    new_X = new_X[:,:,mid:-mid, mid:-mid]
    
    f = theano.function([X], new_X)
    
    return f(data)