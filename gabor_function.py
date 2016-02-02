import numpy as np
import pylab as pl
import numpy.matlib as ml


def gabor_fn_og(sigma, theta, Lambda, psi, gamma, horizontal, vertical):

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

    # sigma = Lambda / math.pi * np.sqrt(np.log(2) / float(2)) * (2 ** bw + 1) / float(2 ** bw - 1);
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    (x, y) = np.meshgrid(
        np.arange(-8, 8, 1) - ml.repmat(horizontal, 1, 16),
        np.arange(-8, 8, 1) - ml.repmat(vertical, 1, 16)
    )

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    # Gabor function
    first_term = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2))
    second_term = np.cos(2 * np.pi / Lambda * x_theta + psi)
    gb = first_term * second_term

    # l2 normalize the Gabor patch
    gb = gb / np.sqrt(np.sum(gb ** 2))

    return gb

# sandbox
a = -1
b = np.pi - 1

# a = a + np.pi
# a = a % np.pi

sample1 = gabor_fn_og(2, 0, 5, np.pi / 2, .5, 0, 0)
sample2 = gabor_fn_og(2, np.pi * 2, 3, np.pi / 2, .5, 0, 0)
pl.imshow(sample1)
pl.show()
pl.imshow(sample2)
pl.show()