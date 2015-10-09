# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 19:29:52 2015

Theano Gradient Procedures and Constrained Optimization

back propagation
root mean squared back propagation

@author: dan
"""


import theano
from theano import tensor as T


def backprop(cost, params, lr=0.001):
    
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    updates.append((params, params - lr * grads))
    return updates
    
        
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
    
    
def censor_updates(updates):
    
    w = updates[1][0]
    updated_w = updates[1][1]
    constrained_w = T.dot(updated_w, T.diag(1 / T.sqrt(T.sum(updated_w ** 2, axis=0))))
    new_update = [updates[0], (w, constrained_w)]
    return new_update   