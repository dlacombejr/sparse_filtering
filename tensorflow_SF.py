import os
import time
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import utilities.visualize as visualize


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
    fs = tf.sqrt(tf.square(f) + 1e-8)                                           # soft-absolute function
    l2fs = tf.sqrt(tf.reduce_sum(tf.square(fs), reduction_indices=1))           # l2 norm of row
    nfs = fs / tf.tile(tf.expand_dims(l2fs, 1), [1, 50000])                     # normalize rows
    l2fn = tf.sqrt(tf.reduce_sum(tf.square(nfs), reduction_indices=0))          # l2 norm of column
    f_hat = nfs / tf.tile(tf.expand_dims(l2fn, 0), [100, 1])                    # normalize columns

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

        f = tf.matmul(self.w, self.x.T)

        return f

    def feed_forward(self):

        """ Performs sparse filtering normalization procedure """

        f_hat = self.norm(self.dot())

        return f_hat


def main():

    # define global parameters
    filename = 'patches.mat'
    n_filters = 100
    learn_rate = 0.001
    iterations = [200]

    # load in data and preprocess
    print "loading data..."
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "data", filename)
    data = loadmat(file_path)['X']
    data -= data.mean(axis=0)
    data = np.float32(data.T)

    # construct the network
    print "building model..."
    weights = tf.Variable(tf.random_uniform([n_filters, data.shape[1]]))
    model = SparseFilter(weights, data)

    # define loss, optimizer, and train function
    loss = tf.reduce_sum(model.feed_forward())
    optimizer = tf.train.GradientDescentOptimizer(learn_rate)
    train = optimizer.minimize(loss)

    # initialize all the variables
    init = tf.initialize_all_variables()

    # run the session
    sess = tf.Session()
    sess.run(init)

    # train the sparse filtering network
    print "training network..."
    t = time.time()
    cost_running = []

    # iterate over training epochs
    for epoch in xrange(iterations[0]):

        sess.run(train)
        current_cost = sess.run(loss)
        cost_running.append(current_cost)
        print("Cost at epoch %i: %0.4f" % (epoch, current_cost))

    # calculate and display elapsed training time
    elapsed = time.time() - t
    print('Elapsed training time: %f' % elapsed)

    # plot the cost function over time
    c = {'layer0': cost_running}
    visualize.plotCost(c)

    # visualize the receptive fields of the first layer
    weights_final = sess.run(weights)
    print weights_final.shape
    visualize.drawplots(weights_final.T, color='gray', convolution='n',
                        pad=0, examples=None, channels=1)


if __name__ == '__main__':
    main()
