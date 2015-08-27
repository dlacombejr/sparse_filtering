# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 17:59:07 2015

Initialize weights for use in sparse filtering

@author: dan
""" 

import numpy as np
import theano

def init_weights(shape):
    
    # All weights are initialized randomly and then normalized using L2 norm    
    
# Alternative means of initializing weights commented out
#    return theano.shared(floatX(np.random.randn(*shape)))
#    return theano.shared(floatX(np.random.randn(*shape) * 0.01))
    
    if len(shape) == 2:

        W = np.random.randn(*shape)
        W = np.dot(W, np.diag(1/np.sqrt(np.sum(W**2, axis = 0))))
        
    elif len(shape) == 4:
        
        W = np.random.randn(*shape)
        W = W.reshape(shape[0], shape[1], shape[2]*shape[3])
        scale_factor = (1/np.sqrt(np.sum(W**2, axis = 2)))
        W = W.T * scale_factor.T
        W = W.T.reshape(shape[0], shape[1], shape[2], shape[3])
        
    return theano.shared(np.asarray(W, dtype=theano.config.floatX))
    
    