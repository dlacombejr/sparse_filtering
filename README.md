# Sparse Filtering

This repository contains modules for building many different types of sparse filtering models and architectures. The structure of this repository is as follows:

* `sparse_filtering.py`
	This file contains the network and layer classes, as well as all of the different types of sparse filtering models that can be placed within the layers. 

* `test.py`
	This file can be called from the command prompt with a number of different input arguments. Given those arguments, it can build an arbitrarily deep neural network that can be fully connected or convolutional. It reads in the specified dataset and preprocesses them and then trains the network with said data. The network and other optional items are saved to a file in the `saved` subdirectory. Optional visualizations can be executed as well. 

* `archive`
	Archived sparse filtering scripts that are currently unused. 

* `data`
	Datasets of images to train the neural networks.

* `scripts`
	These are standalone scripts that can build specific neural networks for sparse filtering. Currently, these scripts are outdated and do not match the functionality of the primary sparse filtering objects. 

* `shell`
	Shell files that can initialize Amazon Machine Images with all known dependencies. Also contains shell files for automatic pushing to and pulling from Github.

* `utilities`
	Contains all of the various utilities. 

---

For emptying the saved folder:

	rm -rfv saved/*

For stopping run-away code:

	pkill -9 python

For pulling data from instance:

	scp -i dan-key-pair-useast.pem ubuntu@ec2-52-91-225-232.compute-1.amazonaws.com:~/sparse_filtering/saved/2015-10-16_18h58m42s/* ~/Documents/research/AWS/sparse_filtering/saved/