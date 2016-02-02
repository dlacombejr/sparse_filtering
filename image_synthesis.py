import os
import theano
import numpy as np
import pylab as pl
import cPickle as pickle
import theano.tensor as t
import utilities.init as init
from utilities.BP import backprop
from scipy.io import loadmat, savemat


def l2_norm(x):

    return t.sqrt(t.sum(t.sqr(x)))

# load in the model that we are interested in
print "loading model..."
file_name = 'model.pkl'
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "saved", "sample", file_name)
model = pickle.load(open(file_path, 'rb'))

# load in the model weights separately
file_name = "weights.mat"
file_path = os.path.join(base_path, "saved", "sample", file_name)
weights = loadmat(file_path)['layer0']

# load in the corresponding data just to compile the training functions
file_name = "patches.mat"  # todo: this should figure out the file automatically
file_path = os.path.join(base_path, "data", file_name)
data = loadmat(file_path)['X']
data = np.float32(data.T)

# compile the training, output, and test functions for the network
print "compiling theano functions..."
_, outputs, _ = model.training_functions(data)
data = None

# define important variables
neurons = weights.shape[0]
dim = np.sqrt(weights.shape[1])
max_iter = 10000

# create data structures to save outputs
images_saved = np.zeros(weights.shape)

# loop over neurons
for neuron in xrange(neurons):

    # # create symbolic variable for synthesized image
    # image = t.fvector('image')

    # create randomly initialized images with the same size as the neuron receptive fields
    image = init.init_weights((1, weights.shape[1]))

    # get the activation value of the neuron
    activation = t.dot(model.layers[0].get_weights()[neuron], image.T)  # using raw activation value
    # _, _, _, _, activations, _ = outputs[model.n_layers - 1](image.get_value())
    # activation = activations[neuron]
    # todo: this will not scale easily to deep architecture; should pass to Network

    # create cost function for evaluation
    Lambda = .1  # parameter of l2 shrinkage
    cost = -(activation[0] - Lambda * l2_norm(image))  # negative in front for gradient ascent

    # put parameters into list
    parameters = image

    # get updates
    updates = backprop(cost, parameters, lr=0.005)

    # compile theano function
    fn = theano.function(
        inputs=[],
        outputs=[cost, image],
        updates=updates
    )

    # visualize the neuron we want to maximize activation of
    pl.imshow(weights[neuron, :].reshape((dim, dim)))
    pl.show()

    # # set the values to initial ones
    # image.set_value(init.unshared_init_weights((1, dim ** 2)), borrow=True)

    # find the best synthesized image for neuron
    c = []
    iteration = 0
    converged = 0
    while iteration < max_iter and converged != 1:

        # get the values of the parameters after a single update
        cost, image_out = fn()
        c.append(cost)

        # print out every once in a while
        every = 10
        if iteration % every == 0:
            print('neuron %d iteration %d; cost %d' % (neuron, iteration, cost))

        # iterate counter
        iteration += 1

        # update break variable
        if iteration > 1000 and c[iteration - 1000] == c[iteration - 1]:
            converged = 1

    # store the synthesized image for neuron
    images_saved[neuron, :] = image_out

    # plot the synthesized image to neuron
    pl.imshow(image_out.reshape((dim, dim)))
    pl.xticks([])
    pl.yticks([])
    pl.show()

# save the images for all neurons
synthesized_images = {'images': images_saved}
folder_path = os.path.join(base_path, "saved", "sample")
file_path = os.path.join(folder_path, "synthesized_images.mat")
savemat(file_path, synthesized_images)
