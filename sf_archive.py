# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 12:11:31 2015

@author: dan
"""

import BP
import theano
import connections
import numpy as np
from init import init_weights
from theano import tensor as T
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d


class groupSF(sparse_filter):

    def __init__(self, w, X, groupMat):

        sparse_filter.__init__(self, w, X)
#        self.Fhat = self.feedForward(self.dot())
#        self.fun = self.Fhat_evaluated()
#        self.r = self.fun()
        self.r = self.feedForward(self.dot())
        sparse_filter.__init__(self, groupMat, T.sqr(self.r)) #T.sqr(self.feedForward(self.dot())))

        '''
        Considerations for improvement:
        -definte cost function as: cost = Fhat + gFhat
        -only normalize at first layer; cost = groupedFhat (didn't work well)
        '''

    def criteria(self):

        self.gFhat = self.feedForward(self.dot())

        return self.r + (self.gFhat) # + self.gFhat

#    def criteria_separate(self):
#
#        self.gFhat = self.feedForward(self.dot())
#
#        return self.Fhat, self.gFhat


class groupSF2(object): # sparse_filter):
    
    def __init__(self, w, X, groupMat):
        
        
#        sparse_filter.__init__(self, w, X)
##        self.Fhat1 = self.feedForward(self.dot())
#        self.gMat = groupMat
        self.w = w
        self.X= X
        self.gMat = groupMat
        
    def criteria(self):
        
        F = T.dot(self.w, self.X)
        Fs = T.sqrt(F**2 + 1e-8)
        L2Fs = (Fs**2).sum(axis=[1])
        L2Fs = T.sqrt(L2Fs)
        NFs = Fs/L2Fs.dimshuffle(0, 'x')
        L2Fn = (NFs**2).sum(axis=[0])
        L2Fn = T.sqrt(L2Fn)
        self.Fhat = NFs/L2Fn.dimshuffle('x', 0)
        
#        self.Fhat = self.feedForward(self.dot())   
        
        F = T.sqrt(T.dot(self.gMat, T.sqr(self.Fhat))) # self.Fhat1)) # self.feedForward(self.dot()
        Fs = T.sqrt(F**2 + 1e-8)
        L2Fs = (Fs**2).sum(axis=[1])
        L2Fs = T.sqrt(L2Fs)
        NFs = Fs/L2Fs.dimshuffle(0, 'x')
        L2Fn = (NFs**2).sum(axis=[0])
        L2Fn = T.sqrt(L2Fn)
        self.gFhat = NFs/L2Fn.dimshuffle('x', 0)
        
#        from connections import distMat
#        x = distMat(self.w.shape[0].eval(), 20)
#        inhibition = T.dot(T.sqr(self.Fhat1.T), x)
#        inhibition = self.Fhat1 * inhibition.T
        
        return T.abs_(self.Fhat) + T.abs_(self.gFhat) #+ T.abs_(inhibition)
        
    def reconstructError(self):
        
#        activation = self.dot()
#        reconstruction = T.dot(self.w.T, activation)
#        error = T.sum(T.abs_(self.X - reconstruction))
        
        return self.Fhat, self.Fhat
        
        
class superSF(sparse_filter):
    
    # normalize F based on groupF then normalize on individual basis
    def __init__(self, w, X, groupMat):
        
        sparse_filter.__init__(self, w, X)
        self.gMat = groupMat
        self.gSize = T.sum(self.gMat[0,:])
        
    def criteria(self):
        
        firstLayer = T.sqr(self.dot())
        bottomUp = T.sqrt(T.dot(self.gMat, firstLayer)) / T.sqr(self.gSize)
        topDown = T.dot(self.gMat.T, bottomUp) / T.sqr(self.gSize)
        
        F = topDown
        Fs = T.sqrt(F**2 + 1e-8)
        
        L2Fs = (Fs**2).sum(axis=[1])
        L2Fs = T.sqrt(L2Fs)
        
        NFs = Fs/L2Fs.dimshuffle(0, 'x')
        firstLayer = firstLayer/L2Fs.dimshuffle(0, 'x')
        
        L2Fn = (NFs**2).sum(axis=[0])
        L2Fn = T.sqrt(L2Fn)
        
        self.Fhat = NFs/L2Fn.dimshuffle('x', 0)
        firstLayer = firstLayer/L2Fn.dimshuffle('x', 0)
        
        Fs = T.sqrt(firstLayer**2 + 1e-8)
        L2Fs = (Fs**2).sum(axis=[1])
        L2Fs = T.sqrt(L2Fs)
        NFs = Fs/L2Fs.dimshuffle(0, 'x') 
        L2Fn = (NFs**2).sum(axis=[0])
        L2Fn = T.sqrt(L2Fn)
        self.Fhat = NFs/L2Fn.dimshuffle('x', 0)
        
        return self.Fhat
        
        
class groupSF3(sparse_filter):
    
    # normalize F based on groupF then normalize on individual basis
    def __init__(self, w, X, groupMat):
        
        sparse_filter.__init__(self, w, X)
        self.gMat = groupMat
        self.gSize = T.sum(self.gMat[0,:])
        
    def criteria(self):
        
#        firstLayer = self.feedForward(self.dot())
        firstLayer = self.dot()        
        
        Fs = T.sqrt(firstLayer**2 + 1e-8)
        L2Fs = (Fs**2).sum(axis=[1])
        L2Fs = T.sqrt(L2Fs)
        NFs = Fs/L2Fs.dimshuffle(0, 'x') 
        L2Fn = (NFs**2).sum(axis=[0])
        L2Fn = T.sqrt(L2Fn)
        self.Fhat = NFs/L2Fn.dimshuffle('x', 0)        
        
        bottomUp = T.sqrt(T.dot(self.gMat, T.sqr(firstLayer + 1e-8))) #/ T.sqr(self.gSize)
        
        Fs = T.sqrt(bottomUp**2 + 1e-8)
        L2Fs = (Fs**2).sum(axis=[1])
        L2Fs = T.sqrt(L2Fs)
        NFs = Fs/L2Fs.dimshuffle(0, 'x') 
        L2Fn = (NFs**2).sum(axis=[0])
        L2Fn = T.sqrt(L2Fn)
        bottomUp = NFs/L2Fn.dimshuffle('x', 0)
        
        topDown = T.dot(self.gMat.T, bottomUp) #/ T.sqr(self.gSize)
        
        Fs = T.sqrt(topDown**2 + 1e-8)
        L2Fs = (Fs**2).sum(axis=[1])
        L2Fs = T.sqrt(L2Fs)
        NFs = Fs/L2Fs.dimshuffle(0, 'x') 
        L2Fn = (NFs**2).sum(axis=[0])
        L2Fn = T.sqrt(L2Fn)
        topDown = NFs/L2Fn.dimshuffle('x', 0)
        
        self.Fhat = topDown
        
#        self.Fhat = firstLayer / topDown
        
#        bottomUp2 = T.sqrt(T.dot(self.gMat, T.sqr(self.Fhat + 1e-8)))
#        
#        Fs = T.sqrt(bottomUp2**2 + 1e-8)
#        L2Fs = (Fs**2).sum(axis=[1])
#        L2Fs = T.sqrt(L2Fs)
#        NFs = Fs/L2Fs.dimshuffle(0, 'x') 
#        L2Fn = (NFs**2).sum(axis=[0])
#        L2Fn = T.sqrt(L2Fn)
#        bottomUp2 = NFs/L2Fn.dimshuffle('x', 0)
#        
#        topDown2 = T.dot(self.gMat.T, bottomUp2)
#        
#        self.Fhat = self.Fhat / topDown2
        
#        Fs = T.sqrt(self.Fhat**2 + 1e-8)
#        L2Fs = (Fs**2).sum(axis=[1])
#        L2Fs = T.sqrt(L2Fs)
#        NFs = Fs/L2Fs.dimshuffle(0, 'x') 
#        L2Fn = (NFs**2).sum(axis=[0])
#        L2Fn = T.sqrt(L2Fn)
#        self.Fhat = NFs/L2Fn.dimshuffle('x', 0)
        
        ################
        
        
        
        
#        firstLayer = firstLayer / F; 
#        Fs = T.sqrt(F**2 + 1e-8)
#        
#        L2Fs = (Fs**2).sum(axis=[1])
#        L2Fs = T.sqrt(L2Fs)
#        
#        NFs = Fs/L2Fs.dimshuffle(0, 'x')
#        firstLayer = firstLayer/L2Fs.dimshuffle(0, 'x')
#        
#        L2Fn = (NFs**2).sum(axis=[0])
#        L2Fn = T.sqrt(L2Fn)
#        
#        self.Fhat = NFs/L2Fn.dimshuffle('x', 0)
#        firstLayer = firstLayer/L2Fn.dimshuffle('x', 0)
        
#        Fs = T.sqrt(self.Fhat**2 + 1e-8)
#        L2Fs = (Fs**2).sum(axis=[1])
#        L2Fs = T.sqrt(L2Fs)
#        NFs = Fs/L2Fs.dimshuffle(0, 'x') 
#        L2Fn = (NFs**2).sum(axis=[0])
#        L2Fn = T.sqrt(L2Fn)
#        self.Fhat = NFs/L2Fn.dimshuffle('x', 0)
        
        
        #########################
        
#        # feed forward first layer
#        self.Fhat = self.feedForward(self.dot())     
#        
#        # feed forward bottom up
#        bottomup = T.dot(self.gMat, self.Fhat)
#        
#        # normalize top layer
#        Fs = T.sqrt(bottomup**2 + 1e-8)
#        L2Fs = (Fs**2).sum(axis=[1])
#        L2Fs = T.sqrt(L2Fs)
#        NFs = Fs/L2Fs.dimshuffle(0, 'x') 
#        L2Fn = (NFs**2).sum(axis=[0])
#        L2Fn = T.sqrt(L2Fn)
#        self.gFhat = NFs/L2Fn.dimshuffle('x', 0)
#        
#        # topdown feedback
#        topdown = T.dot(self.gMat.T, self.gFhat) / T.sqr(self.gSize)
##        self.F2 = self.Fhat * topdown
#        
##        Fs = T.sqrt(topdown**2 + 1e-8)
##        L2Fs = (Fs**2).sum(axis=[1])
##        L2Fs = T.sqrt(L2Fs)
##        NFs = Fs/L2Fs.dimshuffle(0, 'x') 
##        L2Fn = (NFs**2).sum(axis=[0])
##        L2Fn = T.sqrt(L2Fn)
##        topdown = NFs/L2Fn.dimshuffle('x', 0)
#        
#        
#        # normalize bottom based on topdown feedback
#        Fs = T.sqrt(topdown**2 + 1e-8)
#        L2Fs = (Fs**2).sum(axis=[1])
#        L2Fs = T.sqrt(L2Fs)
#        
#        NFs = Fs/L2Fs.dimshuffle(0, 'x')
#        Fhat = self.Fhat/L2Fs.dimshuffle(0, 'x')  
#        
#        L2Fn = (NFs**2).sum(axis=[0])
#        L2Fn = T.sqrt(L2Fn)
#        
#        self.Fhat2 = NFs/L2Fn.dimshuffle('x', 0) 
#        self.Fhat = Fhat/L2Fn.dimshuffle('x', 0) 
#        
#        # final normalization
##        Fs = T.sqrt(self.Fhat**2 + 1e-8)
##        L2Fs = (Fs**2).sum(axis=[1])
##        L2Fs = T.sqrt(L2Fs)
##        NFs = Fs/L2Fs.dimshuffle(0, 'x') 
##        L2Fn = (NFs**2).sum(axis=[0])
##        L2Fn = T.sqrt(L2Fn)
##        self.Fhat = NFs/L2Fn.dimshuffle('x', 0)
        
        return self.Fhat      
        
        
class lateralModel(sparse_filter):
    
    def __init__(self, w, X, distMat):
        
        sparse_filter.__init__(self, w, X)
        self.distMat = distMat
    
    def version1(self):
        
        Fhat = self.feedForward(self.dot())
        latFhat = T.dot(T.abs_(Fhat.T), self.distMat.T)
        latFhat = T.dot(latFhat, T.abs_(Fhat))
        self.latFhat = T.diagonal(latFhat)
        
        return self.latFhat
        
    def version2(self):
        
        Fhat = self.feedForward(self.dot())
        inhibition = T.dot(Fhat.T, self.distMat.T)
        inFhat = Fhat - inhibition.T
        self.inFhat = self.feedForward(inFhat)
        
        return self.inFhat   
        
class gCSF2(convSF):    
    
#    def __init__(self, w, X, groupMat=None):
#        
#        convSF.__init__(self, w, X)
#        self.Fhat = self.feedForward(self.dot()) # [ex., neurons, dim, dim]
##        sparse_filter.__init__(self, groupMat, T.sqr(self.Fhat))
#        self.gSize = T.sum(self.gMat[0,:])
#        self.n_neurons = T.sqrt(w.shape[0])
##        self.gMat = groupMat
##        grouped = T.dot(groupMat, self.feedForward(self.dot()).T)
##        convSF.__init__(self, grouped.dimshuffle(3, 0, 1, 2), X)    
#
#        if groupMat == None:
#            
#            self.gFilter = self._gFilter()     
#
#        self.Fhat = self._reshape(self.Fhat)          
#    
#    def _gFilter(self):
#        
#        g = np.ones((1, self.n_neurons, self.gSize, self.gSize))        
#        self.gFilter = theano.shared(np.asarray(g, dtype=theano.config.floatX))
#        return self.gFilter
#    
#    def _reshape(self):
#        
#        # reshape Fhat for group convolution
#        temp = self.Fhat.reshape(self.Fhat.shape[0], self.Fhat.shape[1], 
#                                 self.Fhat.shape[2] *self.Fhat.shape[3])
#                                 # [ex., neurons, input_dim]
#        
#        temp = temp.dimshuffle(2, 0, 1) # [input_dim, ex., neurons]
#        temp = temp.reshape(self.Fhat.shape[2] *self.Fhat.shape[3], 
#                            self.Fhat.shape[0], T.sqrt(self.Fhat.shape[1]), 
#                            T.sqrt(self.Fhat.shape[1])) # [input_dim, ex., grid, grid]
#                            
#        # augment the reshaped Fhat for toroidal convolution
#        index = self.n_neurons - (self.gSize - 1)
#        corner = temp[:, :, index:, index:]
#        bottom = temp[:, :, index:, :]
#        right  = temp[:, :, :, index:]
#        a = T.concatenate((corner, bottom), axis=3)
#        b = T.concatenate((right, temp), axis=3)
#        c = T.concatenate((a, b), axis=2)
#        
#        # return augmented Fhat
#        self.augFhat = c
#        return self.augFhat # [input_dim, ex., augGrid, augGrid]
#    
#    def criteria(self):
#                
#        self.F = conv2d(self.augFhat, self.gFilter, subsample=(2, 2))
#        
#        return self.grouped
    
    #    def _group_activation(self, Fhat_morphed, gMat):
#    
#        gF = T.dot(gMat, Fhat_morphed)
#        return gF
#    
#    def _group(self):
#        
#        # define symbolic variables for Theano function
#        gMat = T.fmatrix("gMat")
#        output_model = T.fmatrix("output_model")
#        Fhat = T.ftensor3("Fhat")
#        
#         define looped function
#        def group_activation(Fhat, gMat, output_model):
#        
#            self.gF = T.dot(self.gMat, self.Fhat)
#            return gF
#            
#        # create scan across Fhat 
#        result, updates = theano.scan(fn=group_activation,
#                                      outputs_info=None,
#                                      sequences=[self.Fhat],
#                                      non_sequences=[self.gMat, self.output_model])
#        
#        return result
#        # compile and return function
#        fn = theano.function(inputs=[gMat, output_model], outputs=result, 
#                             givens={Fhat: self._reshapeUp(T.sqr(self.Fhat_l1))})
#        return fn

class groupSF(sparse_filter):
    
    ''' Group Sparse Filtering '''
    
    def __init__(self, w, X, gMat):
        
        '''
        Build a group sparse filtering model. 
        
        Parameters: 
        ----------
        w : ndarray
            Weight matrix randomly initialized. 
        X : ndarray (symbolic Theano variable)
            Data for model. 
        groupMat : ndarray
            [groups X neurons] binary array defining groups. 
        '''
        
        # initialize base sparse filter
        sparse_filter.__init__(self, w, X)
#        # get Fhat of first layer
#        self.Fhat_l1 = self.feedForward() # self.dot())
        # assign gMat to model
        self.gMat = gMat 
        # define normalization procedure
        self.norm = norm
#        # initialize group sparse filter with clamped Fhat^2
#        sparse_filter.__init__(self, gMat, T.sqr(self.Fhat_l1)) #T.sqr(self.feedForward(self.dot())))
        
    def gFeedForward(self):
        
        ''' Perform grouped sparse filtering '''
        
        self.gFhat = self.norm(T.sqrt(T.dot(self.gMat, T.sqr(self.feedForward()))))
        
        return self.gFhat
    
    def criterion(self):
        
        ''' Returns the criterion for cost function '''
        
        self.Fhat = self.feedForward()
        self.gFhat = self.gFeedForward() 
        
        return self.Fhat + (1 * (self.gFhat))
        
        
 