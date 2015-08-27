#!/bin/bash

# remember to 'chmod +x the_file_name'

########### wait for reboot ###########

# install included samples and test cuda
cuda-install-samples-7.0.sh ~/
cd NVIDIA\_CUDA-7.0\_Samples/1\_Utilities/deviceQuery  
make  
./deviceQuery

# configure theano to use gpu by default
echo -e "\n[global]\nfloatX=float32\ndevice=gpu\n[mode]=FAST_RUN\n\n[nvcc]\nfastmath=True\n\n[cuda]\nroot=/usr/local/cuda" >> ~/.theanorc 