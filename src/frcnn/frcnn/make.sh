#!/usr/bin/env bash
export CUDA_PATH=/usr/local/cuda/
export CXXFLAGS="-std=c++11"
export CFLAGS="-std=c99"

export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export CPATH=/usr/local/cuda-9.0/include${CPATH:+:${CPATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

REL_SCRIPT_DIR=$(dirname "$0")
INITIAL_DIR=$(pwd)
cd $REL_SCRIPT_DIR

CUDA_ARCH="-gencode arch=compute_30,code=sm_30 \
           -gencode arch=compute_35,code=sm_35 \
           -gencode arch=compute_50,code=sm_50 \
           -gencode arch=compute_52,code=sm_52 \
           -gencode arch=compute_60,code=sm_60 \
           -gencode arch=compute_61,code=sm_61 \
	   -gencode arch=compute_70,code=sm_70 "

# Build RoiPooling module
cd layer_utils/roi_pooling/src/cuda
echo "Compiling roi_pooling kernels by nvcc..."
nvcc -c -o roi_pooling_kernel.cu.o roi_pooling_kernel.cu -x cu -Xcompiler -fPIC $CUDA_ARCH 
cd ../../
python build.py
cd ../../

# Build RoIAlign
cd layer_utils/roi_align/src/cuda
echo 'Compiling crop_and_resize kernels by nvcc...'
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../../
python build.py
cd ../../

# Build NMS
cd nms/src/cuda
echo "Compiling nms kernels by nvcc..."
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../../
python build.py
cd ../

cd $INITIAL_DIR
