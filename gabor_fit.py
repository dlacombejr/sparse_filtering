import os
import theano
import numpy as np
import pylab as pl
import theano.tensor as t
from theano.ifelse import ifelse
from scipy.io import loadmat, savemat


def gradients(cost, parameters, lr=0.001):

    updates = []

    c = 0
    for param in parameters:

        update = param - lr * theano.grad(cost, param)

        if c == 1 or c == 3:

            # update = t.minimum(t.abs_(update), np.pi) * (update / abs(update))
            #
            # update = t.maximum(update, 0)
            # update = t.minimum(update, np.pi)

            update = ifelse(t.lt(update, 0), np.pi * 2 - 0.001, update)
            update = ifelse(t.gt(update, np.pi * 2), 0.001, update)

        if c == 2:

            update = ifelse(t.lt(update, 2), float(20), update)

        elif c == 5 or c == 6:

            update = t.maximum(update, -5)
            update = t.minimum(update, 5)

        updates.append((param, update))

        c += 1

    return updates


def gabor_fn(sigma, theta, Lambda, psi, gamma, horizontal, vertical):

    """
    :param sigma:       standard deviation of Gaussian envelope
    :param theta:       orientation [0, pi)
    :param Lambda:      wavelength
    :param psi:         phase offset [0, pi)
    :param gamma:       aspect ratio
    :param horizontal:  horizontal offset
    :param vertical:    vertical offset
    :return:            Gabor patch of size 16 x 16
    """

    # assign x and y sigma values
    sigma_x = sigma
    sigma_y = sigma / gamma

    # bounding box
    (x, y) = np.meshgrid(
        np.arange(-8, 8, 1),
        np.arange(-8, 8, 1)
    )
    x = np.asarray(x, dtype=theano.config.floatX)
    y = np.asarray(y, dtype=theano.config.floatX)

    # offset in horizontal and vertical directions
    x = x - horizontal
    y = y - vertical

    # rotation of the gabor
    x_theta = x * t.cos(theta) + y * t.sin(theta)
    y_theta = -x * t.sin(theta) + y * t.cos(theta)

    # gabor function
    first_term = t.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2))
    second_term = t.cos(2 * np.pi / Lambda * x_theta + psi)
    gb = first_term * second_term

    # l2 normalize the gabor
    gb = gb / t.sqrt(t.sum(gb ** 2))

    return gb

# get directory paths
base_path = os.path.dirname(__file__)
folder_path = os.path.join(base_path, "saved", "sample_gSF")

# read in sample weights
file_path = os.path.join(folder_path, "weights.mat")
weights = loadmat(file_path)['layer0']  # [neurons, weights]

# create symbolic variable for the data
data = t.fmatrix()

# designate initial parameter values
s_init = 2
T_init = 0
l_init = 5
p_init = np.pi / 2
g_init = 1
h_init = 0
v_init = 0

# create shared variables
s = theano.shared(np.asarray(s_init, dtype=theano.config.floatX))
T = theano.shared(np.asarray(T_init, dtype=theano.config.floatX))
l = theano.shared(np.asarray(l_init, dtype=theano.config.floatX))
p = theano.shared(np.asarray(p_init, dtype=theano.config.floatX))
g = theano.shared(np.asarray(g_init, dtype=theano.config.floatX))
h = theano.shared(np.asarray(h_init, dtype=theano.config.floatX))
v = theano.shared(np.asarray(v_init, dtype=theano.config.floatX))

# get initial Gabor
gabor = gabor_fn(s, T, l, p, g, h, v)

# calculate cost
cost = t.sum(t.sqr(data - gabor))

# put parameters into list
parameters = [s, T, l, p, g, h, v]

# get updates
updates = gradients(cost, parameters, lr=.005)

# compile theano function
fn = theano.function(
    inputs=[data],
    outputs=[cost, gabor, s, T, l, p, g, h, v],
    updates=updates
    # updates=[(param, param - .005 * theano.grad(cost, param)) for param in parameters]
)

# define important variables
neurons = weights.shape[0]
max_iter = 200000
dimension = 16
n_parameters = 7

# create data structures to save outputs
gabor_saved = np.zeros((neurons, dimension, dimension))
params_saved = np.zeros((neurons, n_parameters))
cost_saved = np.zeros((neurons, max_iter))

# loop over neurons  # todo: loop back over poorly fitted Gabors and initialize from well fitted surrounding ones
for neuron in xrange(neurons):

    # get neuron weights and normalize
    test = weights[neuron, :].reshape(dimension, dimension)
    test = test / np.sqrt(np.sum(test ** 2))

    # set the values to initial ones
    s.set_value(s_init, borrow=True)
    T.set_value(T_init, borrow=True)
    l.set_value(l_init, borrow=True)
    p.set_value(p_init, borrow=True)
    g.set_value(g_init, borrow=True)
    h.set_value(h_init, borrow=True)
    v.set_value(v_init, borrow=True)

    # find the best Gabor function
    c = []
    iteration = 0
    converged = 0
    while iteration < max_iter and converged != 1:

        # get the values of the parameters after a single update
        cost, gab, sigma, theta, Lambda, psi, gamma, horizontal, vertical = fn(test)

        if iteration % 1000 == 0:
            print('neuron %d iteration %d' % (neuron, iteration))
            print cost, sigma, theta, Lambda, psi, gamma, horizontal, vertical
        c.append(cost)

        # iterate counter
        iteration += 1

        # update break variable
        if iteration > 1000 and c[iteration - 1000] == c[iteration - 1]:
            converged = 1

    # store the Gabors, parameters, and cost functions
    gabor_saved[neuron, :, :] = gab
    params_saved[neuron, :] = [sigma, theta, Lambda, psi, gamma, horizontal, vertical]
    cost_saved[neuron, 0:iteration] = c

    # # plot the Gabor fitted to neuron
    # pl.subplot(np.sqrt(neurons), np.sqrt(neurons), neuron + 1)
    # pl.imshow(gab)
    # pl.xticks([])
    # pl.yticks([])


# save the Gabors and their respective parameters
gabors = {'gabors': gabor_saved}
file_path = os.path.join(folder_path, "gabors2.mat")
savemat(file_path, gabors)
params = {'params': params_saved}
file_path = os.path.join(folder_path, "params2.mat")
savemat(file_path, params)

# # show the fitted Gabors
# pl.show()

# # show original weights for comparison
# for neuron in xrange(neurons):
#
#     pl.subplot(np.sqrt(neurons), np.sqrt(neurons), neuron + 1)
#     pl.imshow(weights[neuron, :].reshape(dimension, dimension))
#     pl.xticks([])
#     pl.yticks([])
#
# pl.show()

