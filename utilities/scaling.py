# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 19:22:01 2015

Scaling module:

Local Contrast Normalization

@author: dan
"""

import theano
import numpy as np
import numpy.matlib as ml
from theano.tensor.nnet.conv import conv2d
from theano import tensor as T
from pylearn2.utils import sharedX
from pylearn2.datasets.preprocessing import gaussian_filter


def LCN(data, kernel_shape):
    
    # X = T.ftensor4()

    filter_shape = (1, 1, kernel_shape, kernel_shape)
    filters = sharedX(gaussian_filter(kernel_shape).reshape(filter_shape))
    
    convout = conv2d(data, filters=filters, border_mode='full')
    
    # For each pixel, remove mean of 9x9 neighborhood
    mid = int(np.floor(kernel_shape/ 2.))
    centered_X = data - convout[:,:,mid:-mid,mid:-mid]
    
    # Scale down norm of 9x9 patch if norm is bigger than 1
    sum_sqr_XX = conv2d(T.sqr(data), filters=filters, border_mode='full')
    
    denom = T.sqrt(sum_sqr_XX[:,:,mid:-mid,mid:-mid])
    per_img_mean = denom.mean(axis = [2,3])
    divisor = T.largest(per_img_mean.dimshuffle(0, 1, 'x', 'x'), denom)
    
    new_X = centered_X / T.maximum(1., divisor)
    # new_X = new_X[:,:,mid:-mid, mid:-mid]

    new_X = T.extra_ops.squeeze(new_X)  # remove broadcastable dimension
    new_X = new_X[:, 0, :, :]  # TODO: check whether this forced squeeze is good

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
    # new_X = new_X[:,:,mid:-mid, mid:-mid]
    
    f = theano.function([X], new_X)
    
    return f(data)


def lcn_lacombe(data, kernel_shape, n_maps):

    # create basic filter that spans all feature maps
    filter_shape = (1, n_maps, kernel_shape, kernel_shape)
    filters = np.resize(gaussian_filter(kernel_shape), filter_shape)
    filters = filters / np.sum(filters)  # todo: don't scale as this makes input much smaller than weights
    filters = sharedX(filters)

    # for feature_map in xrange(data.shape[0]):
    #
    #     temp[1, feature_map, :, :] = filters
    #
    # temp = temp / ml.repmat(np.sum(temp), (1, data.shape[0], kernel_shape, kernel_shape))

    # filters = sharedX(temp)

    # data = [examples, maps, length, width]; filters = [1, maps, kernel_shape, kernel_shape]
    # output = [examples, 1, length - (kernel_shape - 1), width - (kernel_shape - 1)]
    convout = conv2d(data, filters=filters, border_mode='full')
    # convout = np.reshape(convout, (convout.shape[0], data.shape[1], convout.shape[2], convout.shape[3]))

    # For each pixel, remove mean of 9x9 neighborhood
    mid = int(np.floor(kernel_shape / 2.))
    convout = convout[:, :, mid:-mid, mid:-mid]
    centered_X = data - T.tile(convout, (1, n_maps, 1, 1))

    # Scale down norm of 9x9 patch if norm is bigger than 1
    sum_sqr_XX = conv2d(T.sqr(data), filters=filters, border_mode='full')

    denom = T.sqrt(sum_sqr_XX[:, :, mid:-mid, mid:-mid])
    per_img_mean = denom.mean(axis=[1, 2, 3])
    divisor = T.largest(per_img_mean.dimshuffle(0, 'x', 'x', 'x'), T.tile(denom, (1, n_maps, 1, 1)))

    new_X = centered_X / T.maximum(1., divisor)
    # new_X = new_X[:, :, mid:-mid, mid:-mid]  # maybe safer to return valid area

    return new_X
