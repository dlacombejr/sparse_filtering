import os
import theano 
import scaling
import numpy as np
import pylab as pl
import theano.tensor as t
from scipy.io import loadmat

# load in the data and preprocess
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "data", "STL_10.mat")  # args.filename)
data = loadmat(file_path)['X']

# define important variables
channels = 3
dim = np.sqrt(data.shape[1] / channels)
examples = data.shape[0]

# reshape the data
data = data.reshape((examples, dim, dim, channels), order='F')

# reshuffle the data without transposing using Theano
symbol = t.ftensor4()
symbol_shuffled = symbol.dimshuffle(0, 3, 1, 2)
fn = theano.function([symbol], outputs=[symbol_shuffled])
data_shuffled = fn(data)[0]

# show the original images
for i in range(64):
    pl.subplot(8, 8, int(i+1))
    x = data[i, :, :, :]
    pl.imshow(x) 
    pl.axis('off')
    
pl.show()

# Local Contrast Normalization of images
for i in range(channels):
                                          
    data_shuffled[:, i, :, :] = np.reshape(scaling.LCNinput(data_shuffled[:, i, :, :].reshape((data_shuffled.shape[0], 1, 
                                                                             data_shuffled.shape[2],
                                                                             data_shuffled.shape[3])), 
                                                   kernel_shape=9), 
                                  (data_shuffled.shape[0],
                                  data_shuffled.shape[2],
                                  data_shuffled.shape[3]))        

# reshuffle the data using Theano and avoiding transposing the data
symbol = t.ftensor4()
symbol_shuffled = symbol.dimshuffle(0, 2, 3, 1)
fn = theano.function([symbol], outputs=[symbol_shuffled])
data_unshuffled = fn(data_shuffled)[0] 

# show the LCN images
for i in range(64):
    pl.subplot(8, 8, int(i+1))
    x = data_unshuffled[i, :, :, :]
    x = (x - np.tile(np.amin(x), (96, 96, 3))) / np.tile(np.amax(x) - np.amin(x), (96, 96, 3))
    pl.imshow(x) 
    pl.axis('off')
    
pl.show()


