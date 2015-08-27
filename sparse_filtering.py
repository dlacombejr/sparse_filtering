# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 19:45:32 2015

Sparse Filtering Models

@author: dan
"""

import BP
import theano
import connections
import numpy as np
from scaling import LCN
from init import init_weights
from theano import tensor as t
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d


def norm(f):
    
    """
    Perform sparse filtering normalization procedure. 
    
    Parameters: 
    ----------
    f : ndarray
        The activation of the network. [neurons x examples]
        
    Returns: 
    -------
    f_hat : ndarray
        The row and column normalized matrix of activation.
    """
    
    fs = t.sqrt(f ** 2 + 1e-8)              # ensure numerical stability
    l2fs = t.sqrt(t.sum(fs ** 2, axis=1))   # l2 norm of row
    nfs = fs / l2fs.dimshuffle(0, 'x')      # normalize rows
    l2fn = t.sqrt(t.sum(nfs ** 2, axis=0))  # l2 norm of column
    f_hat = nfs / l2fn.dimshuffle('x', 0)   # normalize columns
    
    return f_hat
    
    
def convolutional_norm(f):
    
    """
    Perform convolutional sparse filtering normalization procedure.
    
    Parameters: 
    ----------
    f : ndarray
        The activation of the network. [examples x neurons x dim x dim]
        
    Returns: 
    -------
    f_hat : ndarray
        The row and column normalized matrix of activation.
    """

    fs = t.sqrt(f ** 2 + 1e-8)                   # ensure numerical stability
    l2fs = t.sqrt(t.sum(fs ** 2, axis=0))        # l2 norm of example dimension
    nfs = fs / l2fs.dimshuffle('x', 0, 1, 2)     # normalize non-example dimensions
    l2fn = t.sqrt(t.sum(nfs ** 2, axis=1))       # l2 norm of neuron dimension
    f_hat = nfs / l2fn.dimshuffle(0, 'x', 1, 2)  # normalize non-neuron dimensions

    return f_hat


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
        
        # define normalization procedure
        self.norm = norm
    
    def dot(self):
        
        """ Returns dot product of weights and input data """
        
        f = t.dot(self.w, self.x)
        
        return f
        
    def feed_forward(self):
        
        """ Performs sparse filtering normalization procedure """
        
        f_hat = self.norm(self.dot())
        
        return f_hat
        
    def criterion(self):
        
        """ Returns the criterion for model evaluation """
        
        return self.feed_forward()

    def reconstruct_error(self):
        
        """ Returns L1 error in reconstruction """
        
        activation = self.dot()
        reconstruction = t.dot(self.w.T, activation)
        error = t.sum(t.abs_(self.x - reconstruction))
        
        return reconstruction, error
   
    
class ConvolutionalSF(SparseFilter):
    
    """ Convolutional Sparse Filtering """
    
    def __init__(self, w, x):
        
        """
        Build a convolutional sparse filtering model. 
        
        Parameters: 
        ----------
        w : ndarray
            Weight matrix randomly initialized. 
        x : ndarray (symbolic Theano variable)
            Data for model. 
        """
        
        # initialize base sparse filter
        SparseFilter.__init__(self, w, x)
        # define normalization procedure
        self.norm = convolutional_norm
        
    def dot(self):
        
        """ Convolve input with model weights """        
        
        f = conv2d(self.x, self.w, subsample=(1, 1))
        
        return f
    
    def feed_forward(self):
        
        """ Performs convolutional sparse filtering procedure """
     
        f_hat = self.norm(self.dot())
        
        return f_hat
        
    def criterion(self):
        
        """ Returns the criterion for model evaluation """
        
        return self.feed_forward()
        
    def reconstruct_error(self):
        
        """ Returns L1 error in reconstruction """
        
        return t.sum(1), t.sum(1)
        
    def max_pool(self):
        
        """ Perform 2D max pooling """
        
        return max_pool_2d(self.feed_forward(), ds=(2, 2))
    
    
class GroupSF(SparseFilter):
    
    """ Group Sparse Filtering """
    # TODO: Add within-group sparsity term to the cost function !!!
    # TODO: Explore and publish within-group sparsity effects on classification (random group assignment!!!)
    
    def __init__(self, w, x, g_mat, clamped='n'):
        
        """
        Build a group sparse filtering model. 
        
        Parameters: 
        ----------
        w : ndarray
            Weight matrix randomly initialized. 
        x : ndarray (symbolic Theano variable)
            Data for model. 
        g_mat : ndarray
            [groups x neurons] binary array defining groups. 
        """
        
        # initialize base sparse filter
        SparseFilter.__init__(self, w, x)
        # assign g_mat to model
        self.g_mat = g_mat
        self.group_size = np.sum(self.g_mat[0, :])
        # define normalization procedure
        self.norm = norm
        # get initial f_hat
        self.f_hat_init = self.feed_forward()
        self.clamped = clamped

    def g_feed_forward(self):
        
        """ Perform grouped sparse filtering """

        if self.clamped == 'n':
            f_hat = self.feed_forward()
            gf_hat = self.norm((t.dot(self.g_mat, t.sqr(f_hat))))
        else:
            f_hat = self.f_hat_init
            gf_hat = self.norm((t.dot(self.g_mat, t.sqr(f_hat))))
        
        return gf_hat
    
    def criterion(self):
        
        """ Returns the criterion for cost function """

        # compute / determine f_hat
        if self.clamped == 'n':
            f_hat = self.feed_forward()
        else:
            f_hat = self.f_hat_init

        # compute gf_hat
        gf_hat = self.g_feed_forward()
        
        return f_hat + (1 * gf_hat)

        
class GroupConvolutionalSF(ConvolutionalSF):
    
    """ Group Convolutional Sparse Filtering """
    
    def __init__(self, w, x, g_mat):
        
        """
        Build a group convolutional sparse filtering model. 
        
        Parameters: 
        ----------
        w : ndarray
            Weight matrix randomly initialized. 
        x : ndarray (symbolic Theano variable)
            Data for model. 
        g_mat : ndarray
            [groups x neurons] binary array defining groups. 
        """
        
        # initialize base convolutional sparse filter
        ConvolutionalSF.__init__(self, w, x)
        # get f_hat from first layer
        self.f_hat_l1 = self.feed_forward()
        # assign g_mat to model
        self.g_mat = g_mat 
        # define normalization procedure
        self.norm = convolutional_norm
        
    def _reshape_up(self):
        
        """ Returns reshaped / shuffled activation for grouped matrix multiplication """
        
        # reshape f [ex., neurons, dim, dim] for grouping
        f_hat_reshaped = self.f_hat_l1.reshape((self.f_hat_l1.shape[0],
                                                self.f_hat_l1.shape[1],
                                                self.f_hat_l1.shape[2] ** 2))  # [ex., neurons, dim^2]
        f_hat_shuffled = f_hat_reshaped.dimshuffle(2, 0, 1)  # [dim^2, ex., neurons]
        
        return f_hat_shuffled
        
    def _group_activation(self, f_hat_morphed):
        
        """ Return grouped activation values """
        
        group_f = t.sqrt(t.dot(t.sqr(f_hat_morphed), self.g_mat.T))  # [dim^2, ex., groups]

        return group_f
        
    def _reshape_down(self, group_f):
        
        """ Returns reshaped/shuffled grouped activation for sparse filtering """
        
        # reshape gF [dim^2, ex., groups] for normalization procedure
        group_f_shuffled = group_f.dimshuffle(1, 2, 0)  # [ex., groups, dim^2]
        group_f_reshaped = group_f_shuffled.reshape((self.f_hat_l1.shape[0],
                                                     self.f_hat_l1.shape[1],
                                                     self.f_hat_l1.shape[2],
                                                     self.f_hat_l1.shape[3]))  # [ex., groups, dim, dim]
                                          
        return group_f_reshaped
        
    def g_feed_forward(self):
        
        """ Perform grouped convolutional sparse filtering """
        
        # reshape f_hat_l1
        f_hat_morphed = self._reshape_up()

        # get gF 
        group_f = self._group_activation(f_hat_morphed)
        
        # reshape back
        group_f_morphed = self._reshape_down(group_f)
        
        # perform normalization
        gf_hat = self.norm(group_f_morphed)

        return gf_hat
        
    def max_pool(self):
        
        """ Perform 2D max pooling """
        
        return max_pool_2d(self.feed_forward(), ds=(2, 2))
        
    def criterion(self):
        
        """ Returns the criterion for model evaluation """
        
        return self.f_hat_l1 + self.g_feed_forward()
        
    def reconstruct_error(self):
        
        """ Returns L1 error in reconstruction """
        
        return t.sum(1), t.sum(1)

    
class Layer(object):
    
    """ Layer object within network """
    
    def __init__(self, model_type='SF', weight_dims=(100, 256), layer_input=None, 
                 p=None, group_size=None, step=None, lr=0.01, c = 'n'):   
        
        """
        Builds a layer for the network by constructing a model. 
        
        Parameters: 
        ----------
        model_type : str
            The model type to build into a given layer. 
        weight_dims : list of tuples
            The dimensions of the weight matrices for each layer. 
            fully connected: [neurons x input_dim ^ 2]
            convolutional: [filters x dim x dim]
        layer_input : ndarray (symbolic Theano variable)
            The input to a given layer. 
        p : int
            The pooling size (assumed to be square).
        group_size : int
            The group size for group sparse filtering. 
        step : int
            The step size for group sparse filtering. 
        lr : int
            The learning rate for gradient descent. 
        """
        
        # assign network inputs to layer
        self.m = model_type
        self.weight_dims = weight_dims
        self.w = init_weights(weight_dims)  # TODO: constrain L2-norm of weights to sum to unity
        self.x = layer_input
        self.p = p    
        self.lr = lr
        self.c = c
        
        # build model based on model_type
        if model_type == 'SparseFilter':
            self.model = SparseFilter(self.w, self.x)
        elif model_type == 'ConvolutionalSF':
            self.model = ConvolutionalSF(self.w, self.x)
        elif model_type == 'GroupSF':
            self.g_mat = connections.gMatToroidal(self.weight_dims[0], group_size, step, centered='n')
            self.model = GroupSF(self.w, self.x, self.g_mat)
        elif model_type == 'GroupConvolutionalSF':
            self.g_mat = connections.gMatToroidal(self.weight_dims[0], group_size, step, centered='n')
            self.model = GroupConvolutionalSF(self.w, self.x, self.g_mat)
        
    def feed_forward(self):
        
        """ Feed-forward through the network """
        
        f_hat = self.model.feed_forward()
        
        return f_hat
        
    def g_feed_forward(self):
        
        """ Feed-forward through the grouped level """
        
        if self.m == 'GroupSF' or self.m == 'GroupConvolutionalSF':
            gf_hat = self.model.g_feed_forward()
        else:
            gf_hat = None
            
        return gf_hat
        
    def criterion(self):
        
        """ Return the criterion for model evaluation """
        
        return self.model.criterion()
        
    def get_cost_updates(self):
        
        """ Returns the cost and updates for the layer """

        cost = t.sum(t.abs_(self.criterion()))
        updates = BP.RMSprop(cost, self.w, lr=self.lr)
        
        return cost, updates
        
    def get_cost_grads(self):
        
        """ Returns the cost and flattened gradients for the layer """ 
         
        cost = t.sum(t.abs_(self.criterion()))  # TODO: explore log in cost function
        grads = t.grad(cost=cost, wrt=self.w).flatten()
        
        return cost, grads
    
    def get_weights(self):
        
        """ Returns the weights of the layer """        
        
        weights = self.w
        
        return weights
        
    def get_rec_err(self):
        
        """ Returns the reconstruction error of the layer """
        
        reconstruct_error = self.model.reconstruct_error()
        
        return reconstruct_error
        
        
class Network(object):
    
    """ Neural network architecture """
    
    def __init__(self, model_type='SF', weight_dims=([100, 256], []), p=None,
                 group_size=None, step=None, lr=0.01, opt='GD', c='n'):
        
        """
        Neural network constructor. Defines a network architecture that builds 
        layers, each with own model. 
        
        Parameters:
        ----------
        model_type : str
            The model type to build into a given layer. 
        weight_dims : list of tuples
            The dimensions of the weight matrices for each layer. 
            fully connected: [neurons x input_dim ^ 2]
            convolutional: [filters x dim x dim]
        p : int
            The pooling size (assumed to be square).
        group_size : int
            The group size for group sparse filtering. 
        step : int
            The step size for group sparse filtering. 
        lr : int
            The learning rate for gradient descent. 
        opt : str
            The optimization algorithm used for learning. 
        c : str
            Indicates whether the network is fully connected or convolutional. 
        """        
        
        # assign the inputs to the network
        self.layers = []
        self.n_layers = len(weight_dims)
        self.opt = opt
        self.weight_dims = weight_dims
        self.c = c
        
        # define symbolic variable for input data based on network type
        if self.c == 'n': 
            self.x = t.fmatrix()
        elif self.c == 'y': 
            self.x = t.ftensor4()
        
        # for each layer, create a layer object
        for l in xrange(self.n_layers):
            
            if l == 0:  # first layer
                layer_input = self.x
                if self.c == 'y': 
                    weight_dims[l].insert(1, 1)

            else:  # subsequent layers
                if self.c == 'n':
                    layer_input = self.layers[l - 1].feed_forward()
                else:  # i.e., convolutional
                    layer_input = self.layers[l - 1].max_pool()
                    layer_input = LCN(layer_input, kernel_shape=5)
                    weight_dims[l].insert(1, weight_dims[l - 1][0])
                    
            self.weight_dims = weight_dims  # update weight_dims
            
            # define layer and append to network layers
            layer_l = Layer(model_type, weight_dims[l], layer_input, p, group_size, step, lr, c)
            self.layers.append(layer_l)

    def training_functions(self, data):  # TODO: Move training_functions to __init__ so that train_fn is not overwritten
        
        """
        Construct training functions for each layer. 
        
        Parameters:
        ----------
        data : ndarray
            Training data for unsupervised feature learning. Can be patches, full
            images, or video. 
            
        Returns:
        -------
        train_fns : list
            List of compiled theano functions for training each layer. 
        out_fns : list
            List of compiled theano functions for retrieving important variables.
        """
        
        # initialize empty function lists
        train_fns = []
        out_fns = []
        
        # for each layer define a training function and output function
        for l in self.layers:
            
            if self.opt == 'GD':  # if gradient descent optimization
                cost, updates = l.get_cost_updates()
                w = l.get_weights()
                fn = theano.function([], outputs=[cost, w], updates=updates, 
                                     givens={self.x: data})
                train_fns.append(fn)
                
            elif self.opt == 'L-BFGS':  # if L-BFGS optimization
                cost, grads = l.get_cost_grads()
                f_hat = l.feed_forward()
                rec, err = l.get_rec_err()
                fn = theano.function(inputs=[], outputs=[cost, grads, f_hat, rec, err], 
                                     givens={self.x: data}, allow_input_downcast=True)
                                   
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
                        The cost value for layer l at a given iteration. 
                    g : float64
                        The vectorized gradients of all weights
                    """
                    
                    if self.c == 'n':  # fully connected
                        theta_value = np.asarray(theta_value.reshape(l.weight_dims[0],
                                                                     l.weight_dims[1]),
                                                 dtype=theano.config.floatX)
                    elif self.c == 'y':  # convolutional
                        theta_value = np.asarray(theta_value.reshape(l.weight_dims[0],
                                                                     l.weight_dims[1],
                                                                     l.weight_dims[2],
                                                                     l.weight_dims[3]),
                                                 dtype=theano.config.floatX)
                                                 
                    l.w.set_value(theta_value, borrow=True)
                    c, g, _, _, _ = fn()
                    c = np.asarray(c, dtype=np.float64)
                    g = np.asarray(g, dtype=np.float64)
                    
                    return c, g
                
                # append the training and output functions for layer l
                train_fns.append(train_fn)
                out_fns.append(fn)

        return train_fns, out_fns