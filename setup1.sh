#!/bin/bash

# remember to 'chmod +x the_file_name'
# call as: Yes | ./setup1.sh

# update 
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get -y dist-upgrade
# sudo apt-get -o Dpkg::Options::="--force-confold" --force-yes -y dist-upgrade

# install theano dependencies
sudo apt-get install -y gcc g++ gfortran build-essential git wget linux-image-generic libopenblas-dev python-dev python-pip python-nose python-numpy python-scipy

# install pylab
sudo apt-get install python-matplotlib

# install scikit learn
sudo apt-get --assume-yes install python-sklearn

# install theano
sudo pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git 

# install pylearn2
git clone git://github.com/lisa-lab/pylearn2.git
export PYTHONPATH="$PYTHONPATH:/home/ubuntu/pylearn2"

# grab cuda toolkit and depackage
sudo wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb 
sudo dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb  

# add the package and install the cuda driver
sudo apt-get update
sudo apt-get install -y cuda 

# update the path to include cuda nvcc and ld_library_path
echo -e "\nexport PATH=/usr/local/cuda/bin:$PATH\n\nexport LD_LIBRARY_PATH=/usr/local/cuda/lib64" >> .bashrc  

# reboot the system for cuda to load
sudo reboot  

