import os
import pylab as pl
from scipy.io import loadmat
from utilities.visualize import drawplots


weights = loadmat('/Users/dan/Documents/research/AWS/sparse_filtering/saved/2015-12-11_00h14m21s/weights.mat')['layer0']
drawplots(weights.T, color='gray', convolution='y', pad=0, examples=None, channels=1)
