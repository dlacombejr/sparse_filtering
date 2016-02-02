import os
import pylab as pl
import utilities.visualize as visualize
from scipy.io import loadmat, savemat


def main():

    # get the folders in "saved" and select most recent
    base_path = os.path.dirname(__file__)
    folder_path = os.path.join(base_path, "saved")
    folders = os.listdir(folder_path)
    folder = folders[-1]

    # load in activation data
    file_path = os.path.join(folder_path, folder, "weights.mat")
    data = loadmat(file_path)

    # visualize the receptive fields of the first layer
    visualize.drawplots(data['layer0'].T, color='gray', convolution='y',
                        pad=0, examples=None, channels=1)

    # visualize the distribution of weights
    for l in xrange(3):

        layer = 'layer' + str(l)
        activations = data[layer]

        pl.subplot(2, 2, l + 1)
        pl.hist(activations.flatten(), bins=50)
        pl.xlabel("Weights")
        pl.ylabel("Count")
        pl.title("Feature Weight Histogram for layer " + str(layer))

    pl.show()

    # # visualize the distribution of lifetime and population sparseness
    # for l in xrange(len(data)):
    #     layer = 'layer' + str(l)
    #     visualize.dispSparseHist(activations_shuffled[layer].reshape(args.dimensions[l][0],
    #                                                                  data.shape[0] *
    #                                                                  activations_shuffled[layer].shape[2] *
    #                                                                  activations_shuffled[layer].shape[3]),
    #                              layer=l)
    #
    # # visualize the distribution of activity across the "cortical sheet" and reconstruction
    # if args.filename == 'patches_video.mat':
    #     f_hat = activations_norm['layer0'].T.reshape(video.shape[0], video.shape[1], args.dimensions[0][0])
    #     visualize.videoCortex(f_hat[0:100, :, :], 'y', args.convolution, 1)
    # else:
    #     visualize.drawplots(activations_norm['layer0'], color='gray', convolution=args.convolution,
    #                         pad=1, examples=100)


if __name__ == '__main__':
    main()
