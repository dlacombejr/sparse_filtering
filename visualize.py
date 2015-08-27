# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 19:37:18 2015

Assorted visualizaiton functions

@author: dan
"""

import numpy as np
import pylab as pl


def displayData(x, weightedInput, num_viz=9):
    
    if weightedInput is not None:
        x = weightedInput
    
    pl.figure()
    for i in range(num_viz):
        pl.subplot(np.sqrt(num_viz), np.sqrt(num_viz), i+1)
        pl.imshow(x[i,:,:], interpolation='nearest')#,cmap=pl.cm.gray)
        pl.xticks([])
        pl.yticks([])
    pl.show()   
    
    
def plotCost(cost):
    
    pl.plot(cost)
    pl.title("Running Cost Function")
    pl.xlabel("Iteration")
    pl.ylabel("Cost")
    pl.show()
    

def drawplots(W, g, c='n', pad=0):
    
    '''
    Displays data (e.g., neuron weights) in a grid of squares   
    Inputs: W (data matrix [dimensions X examples]); g (color map arg); 
            c (conv arg); pad (padding argument)
    Outpus: Plot of the tiled data reshpaed into squares
    
    '''
    
    # adjust plot parameters based on convolutional case
    if c == 'n':
        patch_size = np.sqrt(W.shape[0])
        plot_size = np.sqrt(W.shape[1])
    elif c== 'y':
        patch_size = np.asarray(W.shape[0])
        plot_size = np.sqrt(W.shape[3])
    
    # initialize images as ones based on parameters defined above
    if pad == 0:
        image = np.zeros((patch_size*plot_size+plot_size,patch_size*plot_size+plot_size))
    elif pad != 0:
        image = np.zeros((patch_size*plot_size+plot_size + (plot_size +1)*pad,
                          patch_size*plot_size+plot_size + (plot_size +1)*pad))
    
    # loop over patches of the image
    for i in range(plot_size.astype(int)):
        for j in range(plot_size.astype(int)):
            
            # reshape the data into a patch 
            if c == 'n':
                patch = np.reshape(W[:,i*plot_size+j],(patch_size,patch_size))
            elif c== 'y':
                patch = np.reshape(W[:,:, :, i*plot_size+j], (W.shape[0], W.shape[1]))
            
            # put the patch into the image
            if pad == 0:
                image[i*patch_size+i:i*patch_size+patch_size+i,
                      j*patch_size+j:j*patch_size+patch_size+j] = patch/np.max(np.abs(patch))
            elif pad != 0:
                image[(pad*(i+1)) + i*patch_size+i:(pad*(i+1)) + i*patch_size+patch_size+i,
                      (pad*(j+1)) + j*patch_size+j:(pad*(j+1)) + j*patch_size+patch_size+j] = patch/np.max(np.abs(patch))
    
    # determine cmap from argument
    if g == 'n':
        pl.imshow(image, interpolation="nearest")
    elif g == 'y':
        pl.imshow(image, interpolation="nearest", cmap=pl.gray())
    
    # show the image    
    pl.show()
    
    
def drawReconstruction(W, W2, g, c='n', pad=0):
    
    '''
    Displays data (e.g., neuron weights) in a grid of squares   
    Inputs: W (data matrix [dimensions X examples]); g (color map arg); 
            c (conv arg); pad (padding argument)
    Outpus: Plot of the tiled data reshpaed into squares
    
    '''
    
    # adjust plot parameters based on convolutional case
    if c == 'n':
        patch_size = np.sqrt(W.shape[0])
        plot_size = np.sqrt(W.shape[1])
    elif c== 'y':
        patch_size = W.shape[0]
        plot_size = W.shape[3]
    
    # subplot 1
    pl.subplot(1, 2, 1)
    
    # initialize images as ones based on parameters defined above
    if pad == 0:
        image = np.zeros((patch_size*plot_size+plot_size,patch_size*plot_size+plot_size))
    elif pad != 0:
        image = np.zeros((patch_size*plot_size+plot_size + (plot_size +1)*pad,
                          patch_size*plot_size+plot_size + (plot_size +1)*pad))
    
    # loop over patches of the image
    for i in range(plot_size.astype(int)):
        for j in range(plot_size.astype(int)):
            
            # reshape the data into a patch 
            if c == 'n':
                patch = np.reshape(W[:,i*plot_size+j],(patch_size,patch_size))
            elif c== 'y':
                patch = np.reshape(W[:,:, :, i*plot_size+j], (W.shape[0], W.shape[1]))
            
            # put the patch into the image
            if pad == 0:
                image[i*patch_size+i:i*patch_size+patch_size+i,
                      j*patch_size+j:j*patch_size+patch_size+j] = patch/np.max(np.abs(patch))
            elif pad != 0:
                image[(pad*(i+1)) + i*patch_size+i:(pad*(i+1)) + i*patch_size+patch_size+i,
                      (pad*(j+1)) + j*patch_size+j:(pad*(j+1)) + j*patch_size+patch_size+j] = patch/np.max(np.abs(patch))
    
    # determine cmap from argument
    if g == 'n':
        pl.imshow(image, interpolation="nearest")
    elif g == 'y':
        pl.imshow(image, interpolation="nearest", cmap=pl.gray())
        
    # subplot 2    
    pl.subplot(1, 2, 2)
    
        # initialize images as ones based on parameters defined above
    if pad == 0:
        image = np.zeros((patch_size*plot_size+plot_size,patch_size*plot_size+plot_size))
    elif pad != 0:
        image = np.zeros((patch_size*plot_size+plot_size + (plot_size +1)*pad,
                          patch_size*plot_size+plot_size + (plot_size +1)*pad))
    
    # loop over patches of the image
    for i in range(plot_size.astype(int)):
        for j in range(plot_size.astype(int)):
            
            # reshape the data into a patch 
            if c == 'n':
                patch = np.reshape(W2[:,i*plot_size+j],(patch_size,patch_size))
            elif c== 'y':
                patch = np.reshape(W2[:,:, :, i*plot_size+j], (W2.shape[0], W2.shape[1]))
            
            # put the patch into the image
            if pad == 0:
                image[i*patch_size+i:i*patch_size+patch_size+i,
                      j*patch_size+j:j*patch_size+patch_size+j] = patch/np.max(np.abs(patch))
            elif pad != 0:
                image[(pad*(i+1)) + i*patch_size+i:(pad*(i+1)) + i*patch_size+patch_size+i,
                      (pad*(j+1)) + j*patch_size+j:(pad*(j+1)) + j*patch_size+patch_size+j] = patch/np.max(np.abs(patch))
    
    # determine cmap from argument
    if g == 'n':
        pl.imshow(image, interpolation="nearest")
    elif g == 'y':
        pl.imshow(image, interpolation="nearest", cmap=pl.gray())
    
    # show the image    
    pl.show()
    
    
def videoCortex(W, g, c='n', pad=0):
    
    # adjust plot parameters based on convolutional case
    if c == 'n':
        patch_size = np.sqrt(W.shape[2])
        plot_size = np.sqrt(W.shape[0])
    elif c== 'y':
        patch_size = W.shape[0]
        plot_size = W.shape[3]
    
    # initialize images as ones based on parameters defined above
    if pad == 0:
        image = np.zeros((patch_size*plot_size+plot_size,patch_size*plot_size+plot_size))
    elif pad != 0:
        image = np.zeros((patch_size*plot_size+plot_size + (plot_size +1)*pad,
                          patch_size*plot_size+plot_size + (plot_size +1)*pad))
    
    # loop over all frames
    img = None
    for f in range(W.shape[0]):  
    
        # select current data and reshape
        data = W[:, f, :]
        data = np.squeeze(data).T
    
        # loop over patches of the image
        for i in range(plot_size.astype(int)):
            for j in range(plot_size.astype(int)):
                
                # reshape the data into a patch 
                if c == 'n':
                    patch = np.reshape(data[:,i*plot_size+j],(patch_size,patch_size))
                elif c== 'y':
                    patch = np.reshape(data[:,:, :, i*plot_size+j], (data.shape[0], data.shape[1]))
                
                # put the patch into the image
                if pad == 0:
                    image[i*patch_size+i:i*patch_size+patch_size+i,
                          j*patch_size+j:j*patch_size+patch_size+j] = patch/np.max(np.abs(patch))
                elif pad != 0:
                    image[(pad*(i+1)) + i*patch_size+i:(pad*(i+1)) + i*patch_size+patch_size+i,
                          (pad*(j+1)) + j*patch_size+j:(pad*(j+1)) + j*patch_size+patch_size+j] = patch/np.max(np.abs(patch))
        
        # determine cmap from argument
        if img is None:
            img = pl.imshow(image)
        else:
            img.set_data(image)
        
        # show the image    
        pl.pause(.001)
        pl.draw()
    
    
def dispSparseHist(Fhat):
    
    """
    Visualize histograms of sparsity criteria
    
    """
    
    # display feature activation histogram
    pl.subplot(2, 2, 1)
    pl.hist(Fhat.flatten(), bins=50)
    pl.xlabel("Activation")
    pl.ylabel("Count")
    pl.title("Feature Activation Histogram")
    
    # display lifetime sparsity histogram
    pl.subplot(2, 2, 2)
    activated_features = (Fhat > 0.1).mean(axis=1)
    pl.hist(activated_features)
    pl.xlabel("Feature Activation Over All Examples")
    pl.ylabel("Count")
    pl.title("Lifetime Sparsity Histogram")
    
    # display population sparsity histogram
    pl.subplot(2, 2, 3)
    activated_features = (Fhat > 0.1).mean(axis=0)
    pl.hist(activated_features)
    pl.xlabel("Ratio of Active Features in Example")
    pl.ylabel("Count")
    pl.title("Population Sparsity Histogram")
    
    # display dispersal histogram
    pl.subplot(2, 2, 4)
    pl.hist((Fhat**2).mean(axis=1))
    pl.xlabel("Mean Squared Feature Activation")
    pl.ylabel("Count")
    pl.title("Dispersal Histogram")
    
    # display
    pl.show()
    