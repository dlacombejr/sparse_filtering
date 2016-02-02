import os
import h5py
import numpy as np
import pylab as pl
import utilities.scaling as scaling
from scipy.io import loadmat, savemat


# define global parameters
file_names = ['unlabeled.mat', 'train.mat', 'test.mat']  # 'STL_10.mat'
examples = None

# loop over the data-sets
for file_ in file_names:

    # load in data
    print "loading data..."
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "data", file_)

    if file_ == file_names[0]:
        data = h5py.File(file_path)['X']
        data = np.array(data)
        data = data.T

    else:
        data = loadmat(file_path)['X']

    print data.shape

    # reshape data and convert to float then select the examples and perform LCN
    print "pre-processing data..."
    data = np.float32(data.reshape(-1, 3, int(np.sqrt(data.shape[1] / 3)), int(np.sqrt(data.shape[1] / 3))))
    data = data[0:examples, :, :, :]
    channels = data.shape[1]
    for channel in range(channels):
        print "channel %d" % channel
        data[:, channel, :, :] = np.reshape(scaling.LCNinput(data[:, channel, :, :].
                                                             reshape((data.shape[0], 1,
                                                                      data.shape[2],
                                                                      data.shape[3])),
                                                             kernel_shape=9), (
                                            data.shape[0],
                                            data.shape[2],
                                            data.shape[3]))

    # save the LCN normalized images
    print "saving the normalized data..."
    if file_ == file_names[0]:
        file_name = "STL_10_lcn_" + file_ + ".h5"
        h5f = h5py.File(file_name, 'w')
        h5f.create_dataset('X', data=data)
        h5f.close()
    else:
        file_name = "STL_10_lcn_" + file_
        normed_data = {'X': data}
        savemat(file_name, normed_data)