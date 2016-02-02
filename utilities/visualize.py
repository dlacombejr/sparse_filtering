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

    layers = len(cost)

    pl.figure()
    for i in range(layers):
        pl.subplot(1, layers, i + 1)
        pl.plot(cost['layer' + str(i)])
        pl.title("Running Cost Function for Layer " + str(i))
        pl.xlabel("Iteration")
        pl.ylabel("Cost")
    pl.show()
    

def drawplots(W, color='jet', convolution='n', pad=0, examples=None, channels=1):
    
    '''
    Displays data (e.g., neuron weights) in a grid of squares   
    Inputs: W (data matrix [dimensions X examples]); g (color map arg);
            c (conv arg); pad (padding argument)
    Outpus: Plot of the tiled data reshpaed into squares
    
    '''

    # take only a sample if examples is not None (also implies visualizing activation, not weights)
    if examples is not None:
        if convolution == 'n':
            W = W[:, np.random.permutation(W.shape[1])[0:examples]]
        elif convolution == 'y':
            W = W[np.random.permutation(W.shape[0])[0:examples], :, 10, 10].T  # somewhere in the top left of the image

    # determine plot parameters
    patch_size = None
    plot_size = None
    if convolution == 'n' or convolution == 'y' and examples is not None:
        plot_size = np.sqrt(W.shape[1])
        if channels == 1:
            patch_size = np.sqrt(W.shape[0])
        elif channels == 3:
            patch_size = np.sqrt(W.shape[0] / channels)
    elif convolution == 'y' and examples is None:
        patch_size = np.asarray(W.shape[0])
        plot_size = np.sqrt(W.shape[3])
        channels = W.shape[2]
        if channels == 3:
            color = None

    # initialize images as ones based on parameters defined above
    if pad == 0:
        image = np.zeros((patch_size * plot_size + plot_size,
                          patch_size * plot_size + plot_size,
                          channels))
    elif pad != 0:
        image = np.zeros((patch_size * plot_size + plot_size + (plot_size + 1) * pad,
                          patch_size * plot_size + plot_size + (plot_size + 1) * pad,
                          channels))

    # calculate weight divider
    if convolution == 'n' and channels == 3:
        divider = W.shape[0] / channels

    # loop over patches of the image
    for i in range(plot_size.astype(int)):
        for j in range(plot_size.astype(int)):

            for channel in range(channels):

                # reshape the data into a patch
                if convolution == 'n' and channels == 1 \
                        or convolution == 'y' and examples is not None and channels == 1:
                    patch = np.reshape(W[:, i * plot_size + j], (patch_size, patch_size))
                elif convolution == 'n' and channels == 3 \
                        or convolution == 'y' and examples is not None and channels == 3:
                    patch = np.reshape(W[channel * divider: (channel + 1) * divider, i * plot_size + j],
                                       (patch_size, patch_size))
                elif convolution == 'y' and examples is None:
                    patch = np.reshape(W[:, :, channel, i * plot_size + j], (W.shape[0], W.shape[1]))

                # put the patch into the image
                if pad == 0:
                    image[
                        i * patch_size + i: i * patch_size + patch_size + i,
                        j * patch_size + j: j * patch_size + patch_size + j,
                        channel
                    ] = patch / np.max(np.abs(patch))
                elif pad != 0:
                    image[
                        (pad * (i + 1)) + i * patch_size + i:(pad * (i + 1)) + i * patch_size + patch_size + i,
                        (pad * (j + 1)) + j * patch_size + j:(pad * (j + 1)) + j * patch_size + patch_size + j,
                        channel
                    ] = patch / np.max(np.abs(patch))

            # scale the patch of the image
            if channels == 3:  # convolution == 'n' and
                temp = image[
                    i * patch_size + i: i * patch_size + patch_size + i,
                    j * patch_size + j: j * patch_size + patch_size + j,
                    :
                ]

                temp = (temp - np.tile(np.amin(temp), (patch_size, patch_size, channels))) / \
                    np.tile(np.amax(temp) - np.amin(temp), (patch_size, patch_size, channels))

                image[
                    i * patch_size + i: i * patch_size + patch_size + i,
                    j * patch_size + j: j * patch_size + patch_size + j,
                    :
                ] = temp

    # determine color map from argument
    if color == 'jet':
        pl.imshow(np.squeeze(image), interpolation="nearest")
    elif color == 'gray':
        pl.imshow(np.squeeze(image), interpolation="nearest", cmap=pl.gray())
    elif color is None:
        pl.imshow(image)
    pl.xticks([])
    pl.yticks([])
    
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


def convolutional_reconstruction(data, activations, weights, color='jet', convolution='n'):

    """
    data = [channels, dim, dim]
    activations = [examples, neurons, dims, dims]     # slightly smaller due to convolution
    weights = [neurons, channels, l, w]

    1. reshape weights - [neurons, synapses]
    2. get activations from just one example - [neurons, dim, dim]
    3. loop trough the image space and reconstruct
    """

    data = np.squeeze(data)                                             # [dim, dim]
    activations = activations[0, :, :, :].squeeze()                     # [neurons, dims, dims]
    weights = weights.reshape((weights.shape[0],
                               weights.shape[2] * weights.shape[3]))    # [neurons, synapses]

    dim = data.shape[0]
    dims = activations.shape[2]
    patch_dim = np.sqrt(weights.shape[1])
    reconstructed_image = np.zeros((dim, dim))
    for i in range(dims):
        for j in range(dims):
            reconstructed_patch = np.dot(activations[:, i, j].T, weights).reshape((patch_dim, patch_dim))
            reconstructed_image[j:j + patch_dim, i:i + patch_dim] = \
                reconstructed_image[j:j + patch_dim, i:i + patch_dim] + reconstructed_patch

    # determine color map from argument
    if color == 'jet':
        pl.imshow(reconstructed_image, interpolation="nearest")
    elif color == 'gray':
        pl.imshow(reconstructed_image, interpolation="nearest", cmap=pl.gray())

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
    
    
def dispSparseHist(Fhat, layer=0):
    
    """
    Visualize histograms of sparsity criteria
    
    """

    # create superior title
    pl.suptitle('Histograms for Layer ' + str(layer))
    
    # display feature activation histogram
    pl.subplot(2, 2, 1)
    pl.hist(Fhat.flatten(), bins=50)
    pl.xlabel("Activation")
    pl.ylabel("Count")
    pl.title("Feature Activation Histogram")
    
    # display lifetime sparsity histogram
    pl.subplot(2, 2, 2)
    activated_features = (Fhat > 0.1).mean(axis=1)
    pl.hist(activated_features, bins=50)
    pl.xlabel("Feature Activation Over All Examples")
    pl.ylabel("Count")
    pl.title("Lifetime Sparsity Histogram")
    
    # display population sparsity histogram
    pl.subplot(2, 2, 3)
    activated_features = (Fhat > 0.1).mean(axis=0)
    pl.hist(activated_features, bins=50)
    pl.xlabel("Ratio of Active Features in Example")
    pl.ylabel("Count")
    pl.title("Population Sparsity Histogram")
    
    # display dispersal histogram
    pl.subplot(2, 2, 4)
    pl.hist((Fhat**2).mean(axis=1), bins=50)
    pl.xlabel("Mean Squared Feature Activation")
    pl.ylabel("Count")
    pl.title("Dispersal Histogram")
    
    # display
    pl.show()


def visualize_convolved_image(activations, dim=None):

    subplot_dims = np.sqrt(activations.shape[1])

    for i in range(activations.shape[1]):
        pl.subplot(subplot_dims, subplot_dims, i + 1)
        pl.imshow(activations[0, i, :, :].reshape((dim, dim)), interpolation='nearest', cmap=pl.cm.gray)
        pl.xticks([])
        pl.yticks([])

    pl.show()

#
# def visualize_convolved_image2(activations):
#
#     for i in range(activations.shape[0]):
#         pl.subplot(4, 4, i + 1)
#         pl.imshow(activations[i, :, :].reshape((28, 28)), interpolation='nearest', cmap=pl.cm.gray)
#         pl.xticks([])
#         pl.yticks([])
#
#     pl.show()


def plot_mean_std(mean1, std1, mean2, std2):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    pl.figure()
    pl.title('Correlation of Coefficients as a Function of Neuronal Distance')
    # if ylim is not None:
    #     pl.ylim(*ylim)
    pl.xlabel("Neuronal Distance")
    pl.ylabel("Correlation of Coefficients")
    # train_sizes, train_scores, test_scores = learning_curve(
    #     estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    # train_scores_mean = np.mean(train_scores, axis=1)
    # train_scores_std = np.std(train_scores, axis=1)
    # test_scores_mean = np.mean(test_scores, axis=1)
    # test_scores_std = np.std(test_scores, axis=1)
    pl.grid()

    train_sizes = np.linspace(0, 1.0, len(mean1))

    pl.fill_between(train_sizes, mean1 - std1,
                    mean1 + std1, alpha=0.1,
                    color="r")
    pl.fill_between(train_sizes, mean2 - std2,
                    mean2 + std2, alpha=0.1,
                    color="g")

    # pl.fill_between(train_sizes, test_scores_mean - test_scores_std,
    #                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
    pl.plot(train_sizes, mean1, 'o-', color="r",
             label="Regular Sparse Filtering")
    pl.plot(train_sizes, mean2, 'o-', color="g",
             label="Topographic Sparse Filtering")

    # pl.plot(train_sizes, test_scores_mean, 'o-', color="g",
    #          label="Cross-validation score")

    pl.legend(loc="best")
    return pl
