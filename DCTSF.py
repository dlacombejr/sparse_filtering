import os
import time
import h5py
import numpy as np
import cPickle as pickle
from utilities import scaling
import sparse_filtering as sf
from scipy.io import loadmat, savemat
from theano.tensor.signal.downsample import max_pool_2d


def main():

    # define global parameters
    model_type = [
        'GroupSF'
    ]
    convolution = 'n'
    filename = "unlabeled_10000.mat"
    input_examples = 10000
    channels = 3
    n_filters = [
        10000
    ]
    n_hidden_previous_layer = 5000
    dimensions = (
        [n_filters[0], n_hidden_previous_layer],
    )
    pool = None
    group = 3
    step = 1
    learn_rate = 0.001  # 0.0001
    iterations = [
        10
    ]
    opt = 'GD'
    whitening = 'y'
    test_model = 'y'
    examples = None
    batch_size = 1000
    lcn_kernel = [
        5,
        5,
        3,
        3,
        3
    ]  # these may have to be odd values so that there is a middle
    aws = 'y'

    # load in data
    print "loading data..."
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "data", filename)
    data = None
    if filename == 'train.mat' or filename == 'unlabeled_10000.mat':
        data = loadmat(file_path)['X']
    elif filename == 'unlabeled.mat' or filename == 'STL_10_lcn_unlabeled.mat.h5':
        data = h5py.File(file_path, 'r')['X']
        data = np.array(data)
        data = data.T

    # preprocess the data and convert to float
    print "pre-processing data..."
    data = np.float32(data.reshape(-1, 3, int(np.sqrt(data.shape[1] / 3)), int(np.sqrt(data.shape[1] / 3))))
    data = data[0:examples, :, :, :]
    print data.shape
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

    # create sub-folder for saved model
    directory_name = None
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

    # load in the front-end of the model and obtain output
    frontend_model_directory_name = "./saved/2016-01-26_18h54m23s"
    if aws == 'y':
        k.key = frontend_model_directory_name + '/model.pkl'
        model = pickle.loads(k.get_contents_as_string())

    # compile the training, output, and test functions for the network
    print "compiling theano functions..."
    train, outputs, test = model.training_functions(data)

    # get output of frontend model to treat as input to DCTSF
    print "getting output of frontend model..."
    batch_size_out_data = 50
    train_input = np.zeros((input_examples, n_hidden_previous_layer, 1, 1))
    n_batches = input_examples / batch_size_out_data
    for batch in xrange(n_batches):
        print "for batch %d" % batch
        batch_start = batch * batch_size_out_data
        batch_end = batch_start + batch_size_out_data
        temp = test[model.n_layers - 1](data[batch_start:batch_end])
        train_input[batch_start:batch_end] = np.sum(temp[0], axis=(2, 3), keepdims=True)
    train_input = train_input.reshape(train_input.shape[0], train_input.shape[1] *
                                      train_input.shape[2] * train_input.shape[3])

    # normalize the output of the frontend model
    train_input -= train_input.mean(axis=1)[:, np.newaxis]
    train_input /= np.std(train_input, axis=1)[:, np.newaxis]

    # make the data float32
    train_input = np.float32(train_input)

    # determine number of batches
    n_batches, rem = divmod(data.shape[0], batch_size)

    # construct the network
    print "building model..."
    model = sf.Network(
        model_type=model_type,
        weight_dims=dimensions,
        p=pool,
        group_size=group,
        step=step,
        lr=learn_rate,
        opt=opt,
        c=convolution,
        test=test_model,
        batch_size=batch_size,
        random='y',
        weights=None,
        lcn_kernel=lcn_kernel
    )

    # compile the training, output, and test functions for the network
    print "compiling theano functions..."
    train, outputs, test = model.training_functions(train_input)

    # train the sparse filtering network
    print "training network..."
    start_time = time.time()
    cost = {}
    weights = {}
    for l in xrange(model.n_layers):

        cost_layer = []
        w = None

        # iterate over training epochs
        for epoch in xrange(iterations[l]):

            # go though [mini]batches
            for batch_index in xrange(n_batches):

                # create index for random [mini]batch
                index = np.int32(np.random.randint(data.shape[0], size=batch_size))

                c, w = train[l](index=index)
                cost_layer.append(c)
                print("Layer %i cost at epoch %i and batch %i: %f" % (l + 1, epoch, batch_index, c))

        # add layer cost and weights to the dictionaries
        cost['layer' + str(l)] = cost_layer
        weights['layer' + str(l)] = w

    # calculate and display elapsed training time
    elapsed = time.time() - start_time
    print('Elapsed training time: %f' % elapsed)

    # save the model for later use
    full_path = directory_name + '/model.pkl'
    pickle.dump(model, open(full_path, 'w'), pickle.HIGHEST_PROTOCOL)
    if aws == 'y':
        k.key = full_path
        k.set_contents_from_filename(full_path)
        os.remove(full_path)

    # save weights separately
    savemat(directory_name + '/weights.mat', weights)
    if aws == 'y':
        k.key = directory_name + '/weights.mat'
        k.set_contents_from_filename(directory_name + '/weights.mat')
        os.remove(directory_name + '/weights.mat')

    # save the cost functions
    savemat(directory_name + '/cost.mat', cost)
    if aws == 'y':
        k.key = directory_name + '/cost.mat'
        k.set_contents_from_filename(directory_name + '/cost.mat')
        os.remove(directory_name + '/cost.mat')

    # create log file
    log_file = open(directory_name + "/log.txt", "wb")  # todo: create log file by looping through args
    for m in range(len(model_type)):
        log_file.write(
            "Model layer %d: \n model:%s \n dimensions:%4s \n iterations:%3d \n" % (m,
                                                                                    model_type[m],
                                                                                    dimensions[m],
                                                                                    iterations[m])
        )
        if model == 'GroupSF' or model == 'GroupConvolutionalSF':
            log_file.write(
                " Groups: %d \n Step: %d" % (group, step)
            )
        ex = data.shape[0]
        if examples is not None:
            ex = examples

    log_file.write(
        " Data-set: %s \n Examples: %6d \n Whitened: %s" % (filename, ex, whitening)
    )
    log_file.write('\nElapsed training time: %f' % elapsed)
    log_file.close()
    if aws == 'y':
        k.key = directory_name + "/log.txt"
        k.set_contents_from_filename(directory_name + "/log.txt")
        os.remove(directory_name + "/log.txt")

    # get output activations for analyses
    for batch in xrange(n_batches):

        # get variables of interest
        activations_norm = {}
        activations_raw = {}
        activations_shuffled = {}
        reconstruction = {}
        error_recon = {}
        pooled = {}

        # f_hat, rec, err, f_hat_shuffled, f, p = outputs[l]()
        begin = batch * batch_size
        end = begin + batch_size
        f_hat, rec, err, f_hat_shuffled, f, p = outputs[model.n_layers - 1](train_input[begin:end])

        # define [mini]batch title
        batch_title = 'layer' + '_end' + '_batch' + '%03d' % batch

        # define norm and raw file names
        norm_file_name = directory_name + '/activations_norm_' + batch_title + '.mat'

        activations_norm[batch_title] = f_hat
        activations_raw[batch_title] = f
        activations_shuffled[batch_title] = f_hat_shuffled
        reconstruction[batch_title] = err
        error_recon[batch_title] = rec
        pooled[batch_title] = p

        # save model as well as weights and activations separately
        savemat(norm_file_name, activations_norm)

        if aws == 'y':

            k.key = norm_file_name
            k.set_contents_from_filename(norm_file_name)
            os.remove(norm_file_name)

    # output helper file for concatenating activations
    helper = {'batches': n_batches, 'output_size': f_hat.shape}
    helper_file_name = directory_name + '/helper.mat'
    savemat(helper_file_name, helper)
    if aws == 'y':
        k.key = helper_file_name
        k.set_contents_from_filename(helper_file_name)
        os.remove(helper_file_name)


if __name__ == '__main__':
    main()
