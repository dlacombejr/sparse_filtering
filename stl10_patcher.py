import os
import h5py
import numpy as np
import pylab as pl
import utilities.scaling as scaling
from scipy.io import loadmat, savemat


# define global parameters
filename = 'unlabeled.mat'  # 'STL_10.mat'
examples = 1000

# load in data
print "loading data..."
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "data", filename)
# data = loadmat(file_path)['X']
data = h5py.File(file_path)['X']
data = np.array(data)
data = data.T

# reshape data and convert to float then select the examples and perform LCN
print "pre-processing data..."
data = np.float32(data.reshape(-1, 3, int(np.sqrt(data.shape[1] / 3)), int(np.sqrt(data.shape[1] / 3))))
data = data[0:examples, :, :, :]
channels = data.shape[1]
# for channel in range(channels):
#     print "channel %d" % channel
#     data[:, channel, :, :] = np.reshape(scaling.LCNinput(data[:, channel, :, :].
#                                                          reshape((data.shape[0], 1,
#                                                                   data.shape[2],
#                                                                   data.shape[3])),
#                                                          kernel_shape=9), (
#                                         data.shape[0],
#                                         data.shape[2],
#                                         data.shape[3]))

# # visualize first image to check normalization was properly executed
# x = data[0, :, :, :].T
# x = (x - np.tile(np.amin(x), (96, 96, 3))) / np.tile(np.amax(x) - np.amin(x), (96, 96, 3))  # scale for visualization!
# pl.imshow(x)
# pl.show()

# convert the data to patches (only take the inner portion to avoid edge effects of LCN)
print "patching data..."
edge = 4
patch_size = 14  # 16
patch_per_dim = (data.shape[2] - edge * 2) / patch_size
n_patches = data.shape[0] * (patch_per_dim ** 2)

patches = np.zeros((n_patches, (patch_size ** 2) * channels))
index = 0
for image in xrange(data.shape[0]):
    for i in xrange(patch_per_dim):
        for j in xrange(patch_per_dim):
            start_i = i * patch_size + edge
            end_i = start_i + patch_size
            start_j = j * patch_size + edge
            end_j = start_j + patch_size
            patches[index, :] = data[image, :, start_i:end_i, start_j:end_j].flatten()

            index += 1
            print "image %d patch %d" % (image, index)

            # x = data[image, :, start_i:end_i, start_j:end_j].T
            # x = (x - np.tile(np.amin(x), (patch_size, patch_size, channels))) / \
            #     np.tile(np.amax(x) - np.amin(x), (patch_size, patch_size, channels))
            # pl.imshow(x)
            # pl.show()

# save the patches  # todo: save the output using h5py
print "saving the patch data..."
# p = {'patches': patches}
# savemat('STL_10_unlabeled_patches.mat', p)

h5f = h5py.File('STL_10_unlabeled_patches_raw1000.h5', 'w')
h5f.create_dataset('patches', data=patches)
h5f.close()