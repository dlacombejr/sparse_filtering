import os
import numpy as np
import pylab as pl
from scipy.io import loadmat
from utilities.visualize import drawplots


# get directory paths
base_path = os.path.dirname(__file__)
folder_path = os.path.join(base_path, "saved", "sample_gSF")

# read in sample weights
file_path = os.path.join(folder_path, "params2.mat")
params_saved = loadmat(file_path)['params']  # [neurons, parameters]

# define important variables
neurons, n_parameters = params_saved.shape
dimension = 16


# visualize the phase across the cortical sheet
phase = params_saved[:, 3].reshape(np.sqrt(neurons), np.sqrt(neurons))

# pl.hist(phase)
# pl.show()

# phase = np.abs(phase) % np.pi


pl.subplot(2, 4, 1)
pl.imshow(phase, interpolation='nearest', cmap='hsv')
pl.xticks([])
pl.yticks([])
pl.title('phase')
pl.colorbar()
# pl.show()

phase_vectors = np.zeros((neurons * 3, 2))
c = 0
for i in xrange(int(np.sqrt(neurons))):
    for j in xrange(int(np.sqrt(neurons))):

        if i != 24 and j != 24:
            phase_vectors[c, 0] = phase[i, j]
            phase_vectors[c, 1] = phase[i, j + 1]
            phase_vectors[c + 1, 0] = phase[i, j]
            phase_vectors[c + 1, 1] = phase[i + 1, j]
        elif i == 24 and j == 24:
            phase_vectors[c, 0] = phase[i, j]
            phase_vectors[c, 1] = phase[i, -1]
            phase_vectors[c + 1, 0] = phase[i, j]
            phase_vectors[c + 1, 1] = phase[-1, j]
        elif j == 24:
            phase_vectors[c, 0] = phase[i, j]
            phase_vectors[c, 1] = phase[i, -1]
            phase_vectors[c + 1, 0] = phase[i, j]
            phase_vectors[c + 1, 1] = phase[i + 1, j]
        elif i == 24:
            phase_vectors[c, 0] = phase[i, j]
            phase_vectors[c, 1] = phase[i, j + 1]
            phase_vectors[c + 1, 0] = phase[i, j]
            phase_vectors[c + 1, 1] = phase[-1, j]

        c += 2

pl.subplot(2, 4, 5)
pl.scatter(phase_vectors[:, 0], phase_vectors[:, 1], s=1)
pl.xlim([0, np.pi])
pl.ylim([0, np.pi])
pl.xticks((0, np.pi / 2, np.pi))
xT = pl.xticks()[0]
# xL=[r'$0\pi$',r'$\pi/2$', r'$\pi$']
xL=[r'$0\pi$',r'$\pi$', r'$2\pi$']
pl.xticks(xT, xL)
pl.yticks(xT, xL)
pl.gca().set_aspect('equal', adjustable='box')
pl.title('phase')
# pl.show()

# visualize the orientation across the cortical sheet
orientation = params_saved[:, 1].reshape(np.sqrt(neurons), np.sqrt(neurons))

# pl.hist(orientation)
# pl.show()

orientation = np.abs(orientation) % np.pi
# orientation = ((orientation + (np.pi / 10)) / 2) * 10
# orientation = np.abs(orientation) % np.pi

pl.subplot(2, 4, 2)
pl.imshow(orientation, interpolation='nearest', cmap='hsv')
pl.xticks([])
pl.yticks([])
pl.title('orientation')
pl.colorbar()
# pl.show()

orientation_vectors = np.zeros((neurons * 3, 2))
c = 0
for i in xrange(int(np.sqrt(neurons))):
    for j in xrange(int(np.sqrt(neurons))):

        if i != 24 and j != 24:
            orientation_vectors[c, 0] = orientation[i, j]
            orientation_vectors[c, 1] = orientation[i, j + 1]
            orientation_vectors[c + 1, 0] = orientation[i, j]
            orientation_vectors[c + 1, 1] = orientation[i + 1, j]
        elif i == 24 and j == 24:
            orientation_vectors[c, 0] = orientation[i, j]
            orientation_vectors[c, 1] = orientation[i, -1]
            orientation_vectors[c + 1, 0] = orientation[i, j]
            orientation_vectors[c + 1, 1] = orientation[-1, j]
        elif j == 24:
            orientation_vectors[c, 0] = orientation[i, j]
            orientation_vectors[c, 1] = orientation[i, -1]
            orientation_vectors[c + 1, 0] = orientation[i, j]
            orientation_vectors[c + 1, 1] = orientation[i + 1, j]
        elif i == 24:
            orientation_vectors[c, 0] = orientation[i, j]
            orientation_vectors[c, 1] = orientation[i, j + 1]
            orientation_vectors[c + 1, 0] = orientation[i, j]
            orientation_vectors[c + 1, 1] = orientation[-1, j]

        c += 2

pl.subplot(2, 4, 6)
pl.scatter(orientation_vectors[:, 0], orientation_vectors[:, 1], s=1)
pl.xlim([0, np.pi])
pl.ylim([0, np.pi])
pl.xticks((0, np.pi / 2, np.pi))
xT = pl.xticks()[0]
xL=[r'$0\pi$',r'$\pi/2$', r'$\pi$']
pl.xticks(xT, xL)
pl.yticks(xT, xL)
pl.gca().set_aspect('equal', adjustable='box')
pl.title('orientation')
# pl.show()

# visualize the frequency across the cortical sheet
frequency = params_saved[:, 2].reshape(np.sqrt(neurons), np.sqrt(neurons))
frequency[frequency > 8] = 8

pl.subplot(2, 4, 3)
pl.imshow(frequency, interpolation='nearest', cmap='winter')
pl.xticks([])
pl.yticks([])
pl.title('frequency')
pl.colorbar()
# pl.show()

frequency_vectors = np.zeros((neurons * 3, 2))
c = 0
for i in xrange(int(np.sqrt(neurons))):
    for j in xrange(int(np.sqrt(neurons))):

        if i != 24 and j != 24:
            frequency_vectors[c, 0] = frequency[i, j]
            frequency_vectors[c, 1] = frequency[i, j + 1]
            frequency_vectors[c + 1, 0] = frequency[i, j]
            frequency_vectors[c + 1, 1] = frequency[i + 1, j]
        elif i == 24 and j == 24:
            frequency_vectors[c, 0] = frequency[i, j]
            frequency_vectors[c, 1] = frequency[i, -1]
            frequency_vectors[c + 1, 0] = frequency[i, j]
            frequency_vectors[c + 1, 1] = frequency[-1, j]
        elif j == 24:
            frequency_vectors[c, 0] = frequency[i, j]
            frequency_vectors[c, 1] = frequency[i, -1]
            frequency_vectors[c + 1, 0] = frequency[i, j]
            frequency_vectors[c + 1, 1] = frequency[i + 1, j]
        elif i == 24:
            frequency_vectors[c, 0] = frequency[i, j]
            frequency_vectors[c, 1] = frequency[i, j + 1]
            frequency_vectors[c + 1, 0] = frequency[i, j]
            frequency_vectors[c + 1, 1] = frequency[-1, j]

        c += 2

frequency_vectors[frequency_vectors == 0] = 1

pl.subplot(2, 4, 7)
pl.scatter(frequency_vectors[:, 0], frequency_vectors[:, 1], s=1)
pl.xlim([2, 8])
pl.ylim([2, 8])
pl.gca().set_aspect('equal', adjustable='box')
pl.title('frequency')
# pl.show()


# visualize location (matlab does cortical sheet)
horizontal = params_saved[:, 5].reshape(np.sqrt(neurons), np.sqrt(neurons))
horizontal = horizontal * 2 + 8
horizontal[horizontal < 0] = None
horizontal[horizontal > 16] = None

# place holder for matlab image:
pl.subplot(2, 4, 4)
pl.imshow(frequency, interpolation='nearest', cmap='winter')
pl.xticks([])
pl.yticks([])
pl.title('location')
pl.colorbar()



horizontal_vectors = np.zeros((neurons * 3, 2))
c = 0
for i in xrange(int(np.sqrt(neurons))):
    for j in xrange(int(np.sqrt(neurons))):

        if i != 24 and j != 24:
            horizontal_vectors[c, 0] = horizontal[i, j]
            horizontal_vectors[c, 1] = horizontal[i, j + 1]
            horizontal_vectors[c + 1, 0] = horizontal[i, j]
            horizontal_vectors[c + 1, 1] = horizontal[i + 1, j]
        elif i == 24 and j == 24:
            horizontal_vectors[c, 0] = horizontal[i, j]
            horizontal_vectors[c, 1] = horizontal[i, -1]
            horizontal_vectors[c + 1, 0] = horizontal[i, j]
            horizontal_vectors[c + 1, 1] = horizontal[-1, j]
        elif j == 24:
            horizontal_vectors[c, 0] = horizontal[i, j]
            horizontal_vectors[c, 1] = horizontal[i, -1]
            horizontal_vectors[c + 1, 0] = horizontal[i, j]
            horizontal_vectors[c + 1, 1] = horizontal[i + 1, j]
        elif i == 24:
            horizontal_vectors[c, 0] = horizontal[i, j]
            horizontal_vectors[c, 1] = horizontal[i, j + 1]
            horizontal_vectors[c + 1, 0] = horizontal[i, j]
            horizontal_vectors[c + 1, 1] = horizontal[-1, j]

        c += 2

pl.subplot(2, 4, 8)
pl.scatter(horizontal_vectors[:, 0], horizontal_vectors[:, 1], s=1)
pl.xlim([0, 16])
pl.ylim([0, 16])
pl.gca().set_aspect('equal', adjustable='box')
pl.title('location')
# pl.show()


pl.show()


###########
# visualize the original weights and the learned Gabor patches

# read in sample weights
file_path = os.path.join(folder_path, "weights.mat")
weights = loadmat(file_path)['layer0']  # [neurons, weights]

# visualize the weights
drawplots(weights.T, color='gray', convolution='n', pad=0, examples=None, channels=1)

# read in sample Gabor patches
file_path = os.path.join(folder_path, "gabors2.mat")
gabors = loadmat(file_path)['gabors']  # [neurons, weights]
gabors = gabors.reshape((gabors.shape[0], gabors.shape[1] * gabors.shape[2]))

# visualize the Gabor patches
drawplots(gabors.T, color='gray', convolution='n', pad=0, examples=None, channels=1)

