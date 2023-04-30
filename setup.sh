#!/bin/sh

# Receive libtorch path
libtorch=
while getopts "l:" opt
do
   case "$opt" in
      l ) libtorch="$OPTARG" ;;
   esac
done
# /home/kuga/Workspace/libtorch/share/cmake/Torch/

# Install submodules
if [ ! -d "extern" ]; then
    git submodule update --init
    cd extern/taro-dvstoolkit/
    python setup.py develop
    cd ../../
fi

# Install requirements.txt
pip install -r requirements.txt

# Compile modules
cmake_path=build
if [ -d "${cmake_path}" ]
then
    rm -rf ${cmake_path}
fi
mkdir ${cmake_path}
cd ${cmake_path}

if [ ! -n "$libtorch" ]; then  
  cmake .. 
else  
  cmake .. -DTORCH_DIR:STRING=$libtorch
fi
make
