import theano
import theano.tensor as T


def one_hot(t, r=None):
    """
    given a tensor t of dimension d with integer values from range(r), return a
    new tensor of dimension d + 1 with values 0/1, where the last dimension
    gives a one-hot representation of the values in t.

    if r is not given, r is set to max(t) + 1
    """
    if r is None:
        r = T.max(t) + 1

    ranges = T.shape_padleft(T.arange(r), t.ndim)
    return T.eq(ranges, T.shape_padright(t, 1))


def max_mask(t, axis):
    """
    given a tensor t and an axis, returns a mask tensor of the same size which is
    1 where the tensor has a maximum along the given axis, and 0 elsewhere.
    """
    a = T.argmax(t, axis=axis)
    a_oh = one_hot(a, t.shape[axis])
    # we want the 'one hot' dimension in the same position as the axis over
    # which we took the argmax. This takes some dimshuffle trickery:
    reordered_dims = range(axis) + [a_oh.ndim - 1] + range(axis, a_oh.ndim - 1)
    print reordered_dims
    return a_oh.dimshuffle(reordered_dims)

    # TODO: generalise this to multiple axes


def lacombe(t, axis):

    a = T.argmax(t, axis=axis)
    a_oh = t.zeros_like(t)
    a_oh = T.inc_subtensor(a_oh[np.arange(2), a], 1)

    return a_oh


if __name__ == '__main__':
    import numpy as np

    # test one_hot
    a = np.array([0,1,2,3,4,5], dtype=theano.config.floatX)
    b = np.array([9,2,0,7,4,5,1], dtype=theano.config.floatX)

    x = T.vector('x')
    f1 = theano.function([x], one_hot(x))

    af1 = f1(a)
    bf1 = f1(b)
    assert af1.shape == (6,6)
    assert bf1.shape == (7,10)

    print af1
    print bf1

    # test max_mask
    a = np.array([[2,3,1],[5,0,2]], dtype=theano.config.floatX)
    y = T.matrix('y')
    f2 = theano.function([y], max_mask(y, 0))
    f3 = theano.function([y], max_mask(y, 1))
    print a
    print f2(a)
    print f3(a)

    print('---------------')
    f_lacombe = theano.function([y], lacombe(y, 1))
    print f_lacombe(a)