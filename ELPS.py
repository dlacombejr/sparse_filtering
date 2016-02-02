import os
import time
import theano
import numpy as np
import cPickle as pickle
import utilities.BP as BP
from theano import tensor as t
from utilities import scaling
from scipy.io import loadmat, savemat
import utilities.visualize as visualize
from utilities.init import init_weights
from theano.tensor.nnet.conv import conv2d


def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):

    grads = t.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = t.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


def censor_updates(updates):

    w = updates[0][0]
    updated_w = updates[0][1]
    constrained_w = updated_w / t.sqrt(t.sum(t.sqr(updated_w), axis=1)).dimshuffle(0, 'x')
    updates[0] = (w, constrained_w)

    return updates


class ELPS(object):

    def __init__(self, w, b, x, a, target, neurons, examples, batch_size, convolutional):

        self.c = convolutional

        self.w = w
        self.b = b
        self.params = [self.w, self.b]
        self.x = x

        self.a = a
        self.target = target

        self.neurons = neurons
        self.examples = examples

        # self.k = np.int32([0, 1])

        self.inhibition_increment_value = np.float32(neurons / float(examples))

        # self.inhibition_increment = self.increment_inhibition_function()

        # theano.shared(np.asarray(neurons / examples, dtype=theano.config.floatX))
        self.one = np.float32(1.)
        # theano.shared(np.asarray(1, dtype=theano.config.floatX))

        self.index = np.arange(batch_size)

    def feed_forward(self):

        out = None
        if self.c == 'y':
            out = t.nnet.sigmoid(conv2d(self.x, self.w) + self.b.dimshuffle('x', 0, 'x', 'x'))
        elif self.c == 'n':
            out = t.nnet.sigmoid(t.dot(self.x, self.w.T) + self.b.dimshuffle('x', 0))

        return out

    def algorithm(self):

        if self.c == 'y':
            k = t.argmax(self.feed_forward() - self.a.dimshuffle('x', 0, 'x', 'x'), axis=[1])
            self.target = t.inc_subtensor(self.target[k], self.one)
            self.a = t.inc_subtensor(self.a[k], self.inhibition_increment)
        if self.c == 'n':
            activation = self.feed_forward()
            k = t.argmax(activation - self.a.dimshuffle('x', 0), axis=[1])
            self.target = t.inc_subtensor(self.target[self.index, k], self.one)
            a = t.inc_subtensor(self.a[k], self.inhibition_increment_value)
            # self.a.set_value(new)

            # ceiling = t.ge(a, 1.)
            # a = t.inc_subtensor(a[ceiling], 1000)

            # ceiling = (a >= 1.).nonzero()
            # a = t.inc_subtensor(a[ceiling], 1000)


            # self.a.set_value(new, borrow='True')
            # self.inhibition_increment(self.inhibition_increment_value)

        return activation, self.target, a

    # def criterion(self):
    #
    #     # activation = self.feed_forward()
    #     activation, target, inhibition = self.algorithm()
    #
    #     return t.sqrt(t.sum(t.sqr(activation - target))), inhibition

    def get_cost_updates(self):

        activation, target, inhibition = self.algorithm()

        cost = t.sqrt(t.sum(t.sqr(activation - target)))

        # cost, inhibition = self.criterion()
        updates = RMSprop(cost, self.params, lr=0.001)
        # updates = BP.censor_updates(updates, conv='y')

        # updates = censor_updates(updates)

        updates.append((self.a, inhibition))

        return cost, updates, activation

    def training_function(self):

        cost, updates, activation = self.get_cost_updates()

        # _, a = self.algorithm()
        #
        # updates.append((self.a, a))

        fn = theano.function(
            inputs=[self.x],
            outputs=[cost, self.w, self.target, self.a, activation],
            updates=updates,
            on_unused_input='ignore'
        )

        return fn

    # def increment_inhibition_function(self):
    #
    #     increment = t.fscalar()
    #     index = t.ivector()
    #     update = (self.a, t.set_subtensor(self.a[index], increment))
    #     fn = theano.function(
    #         inputs=[increment],
    #         outputs=[self.a],
    #         updates=[update],
    #         givens={self.a: self.a, index: self.k})
    #
    #     return fn
    #
    # def reset_target(self, target):
    #
    #     self.target = target
    #
    # def reset_inhibition(self, inhibition):
    #
    #     self.a = inhibition


# set some parameters
filename = "patches.mat"  # training

# load in data
print "loading data..."
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "data", filename)
data = loadmat(file_path)['X']

# preprocess the data and convert to float; NOTE: data may have already been normalized using LCN (check data read)
print "pre-processing data..."
if filename == 'training.mat':
    data = np.float32(data.reshape(-1, 3, int(np.sqrt(data.shape[1] / 3)), int(np.sqrt(data.shape[1] / 3))))
    data = data[:, :, :, :]
    channels = data.shape[1]
    if filename == 'unlabeled.mat' or filename == 'unlabeled_10000.mat' or filename == 'train.mat':
        for channel in range(channels):
            data[:, channel, :, :] = np.reshape(scaling.LCNinput(data[:, channel, :, :].
                                                                 reshape((data.shape[0], 1,
                                                                          data.shape[2],
                                                                          data.shape[3])),
                                                                 kernel_shape=9), (
                                                data.shape[0],
                                                data.shape[2],
                                                data.shape[3])
                                                )
elif filename == 'patches.mat':
    data -= data.mean(axis=0)
    data = np.float32(data.T)
    channels = 1

# set som more parameters
aws = 'y'
convolutional = 'n'
neurons = 625
kernel_size = 11
examples = data.shape[0]
if convolutional == 'y':
    out_dim = data.shape[3] - (kernel_size - 1)
elif convolutional == 'n':
    input_dim = data.shape[1]

batch_size = 1000
epochs = 100
n_batches = examples / batch_size

# create symbolic variables for the model
if convolutional == 'y':
    input_ = t.ftensor4()
    weights = init_weights((neurons, channels, kernel_size, kernel_size))
    biases = theano.shared(np.asarray(np.zeros(neurons), dtype=theano.config.floatX))
    inhibition = theano.shared(np.asarray(np.zeros(neurons), dtype=theano.config.floatX))
    target = theano.shared(np.asarray(np.zeros((batch_size, neurons, out_dim, out_dim)), dtype=theano.config.floatX))
elif convolutional == 'n':
    input_ = t.fmatrix()
    weights = init_weights((neurons, input_dim))
    biases = theano.shared(np.asarray(np.zeros(neurons), dtype=theano.config.floatX))
    inhibition = theano.shared(np.asarray(np.zeros(neurons), dtype=theano.config.floatX))
    target = theano.shared(np.asarray(np.zeros((batch_size, neurons)), dtype=theano.config.floatX))

print "building the model..."
model = ELPS(weights, biases, input_, inhibition, target, neurons, examples, batch_size, convolutional)

print "getting training functions..."
train = model.training_function()

# train the sparse filtering network
print "training network..."
t = time.time()
cost_layer = []

# iterate over training epochs
for epoch in xrange(epochs):

    # go though [mini]batches
    for batch_index in xrange(n_batches):


        batch_begin = batch_index * batch_size
        batch_end = batch_begin + batch_size

        c, weight, target, a_out, activation = train(data[batch_begin:batch_end])
        cost_layer.append(c)
        print("Cost at epoch %i and batch %i: %f" % (epoch, batch_index, c))

        if batch_index % 10 == 0:
            print activation
            print np.max(activation)

            print activation - a_out

            print target
            print np.max(target)
            print np.sum(target == 1)
            print a_out
            print np.max(a_out)

    #     # reset target
    #     model.reset_target(target)
    #     # model.target.set_value(
    #     #     np.asarray(np.zeros((batch_size, neurons, out_dim, out_dim)), dtype=theano.config.floatX)
    #     # )
    #
    # rest inhibition
    # model.reset_inhibition(inhibition)
    model.a.set_value(
        np.asarray(np.zeros(neurons), dtype=theano.config.floatX)
    )

# calculate and display elapsed training time
elapsed = time.time() - t
print('Elapsed training time: %f' % elapsed)

# create sub-folder for saved model
if aws == 'n':
    directory_format = "./saved/%4d-%02d-%02d_%02dh%02dm%02ds"
    directory_name = directory_format % time.localtime()[0:6]
    os.mkdir(directory_name)
elif aws == 'y':
    import boto
    from boto.s3.key import Key
    s3 = boto.connect_s3()
    my_bucket = 'dlacombejr.bucket'
    bucket = s3.get_bucket(my_bucket)
    k = Key(bucket)
    directory_format = "./saved/%4d-%02d-%02d_%02dh%02dm%02ds"
    directory_name = directory_format % time.localtime()[0:6]
    os.mkdir(directory_name)

# save the model for later use
full_path = directory_name + '/model.pkl'
pickle.dump(model, open(full_path, 'w'), pickle.HIGHEST_PROTOCOL)
if aws == 'y':
    k.key = full_path
    k.set_contents_from_filename(full_path)
    os.remove(full_path)

# save weights separately
savemat(directory_name + '/weights.mat', {'weights': weight})
if aws == 'y':
    k.key = directory_name + '/weights.mat'
    k.set_contents_from_filename(directory_name + '/weights.mat')
    os.remove(directory_name + '/weights.mat')

# save the cost functions
savemat(directory_name + '/cost.mat', {'cost': cost_layer})
if aws == 'y':
    k.key = directory_name + '/cost.mat'
    k.set_contents_from_filename(directory_name + '/cost.mat')
    os.remove(directory_name + '/cost.mat')

# save the target
savemat(directory_name + '/target.mat', {'target': target})
if aws == 'y':
    k.key = directory_name + '/target.mat'
    k.set_contents_from_filename(directory_name + '/target.mat')
    os.remove(directory_name + '/target.mat')

# save the inhibition
savemat(directory_name + '/inhibition.mat', {'inhibition': a_out})
if aws == 'y':
    k.key = directory_name + '/inhibition.mat'
    k.set_contents_from_filename(directory_name + '/inhibition.mat')
    os.remove(directory_name + '/inhibition.mat')

# plot the cost
c = {'layer0': cost_layer}
visualize.plotCost(c)

# visualize the receptive fields of the first layer
visualize.drawplots(weight.T, color='gray', convolution='n',
                    pad=0, examples=None, channels=1)
