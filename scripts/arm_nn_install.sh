#!/bin/bash

if [ "$EUID" -ne 0 ]; then
  echo "Please run as root!"
  exit
fi

# Cmake parallel processing flag
NUM_CORES="$(($(nproc) - 1))"
export MAKEFLAGS="--parallel $NUM_CORES"
echo "Number of utilized cores: $NUM_CORES"

# Increase virtual memory swapfile allocation from 100 to 1024
SWAP_SIZE="CONF_SWAPSIZE=100"
if ! grep -Fxq "$SWAP_SIZE" /etc/dphys-swapfile; then
  echo "Increasing virtual memory swapfile allocation from 100 to 1024..."
  sed -ie 's/CONF_SWAPSIZE=100/CONF_SWAPSIZE=1024/g' /etc/dphys-swapfile
  # Restart dphys
  /etc/init.d/dphys-swapfile stop && /etc/init.d/dphys-swapfile start
  echo "Done!"
fi

# Install SCONS and CMAKE
echo "Checking for updates and installing various tools..."
apt-get update
apt-get install -y scons cmake autoconf libtool libpcre3 libpcre3-dev
echo "Done!"

# Create a base directory
echo "Creating a base directory..."
mkdir armnn-install
cd armnn-install || exit 1

BASEDIR=$(pwd)
export BASEDIR
echo "BASEDIR: $BASEDIR"

# Install ComputeLib
echo "Downloading ComputeLibrary..."
git clone https://github.com/Arm-software/ComputeLibrary.git "$BASEDIR"/ComputeLibrary
cd "$BASEDIR"/ComputeLibrary || exit 1
echo "Installing ComputeLibrary..."
scons arch=arm64-v8a neon=1 extra_cxx_flags="-fPIC" opencl=1 embed_kernels=1 benchmark_tests=0 validation_tests=0 -j$NUM_CORES
echo "Done!"

# Install Protobuf
echo "Downloading ComputeLibrary..."
git clone -b v3.5.0 https://github.com/google/protobuf.git "$BASEDIR"/protobuf
cd "$BASEDIR"/protobuf || exit 1
git submodule update --init --recursive
echo "Configuring ComputeLibrary..."
./autogen.sh
./configure --prefix="$BASEDIR"/protobuf-host
echo "Installing ComputeLibrary..."
make -j$NUM_CORES
make install -j$NUM_CORES
echo "Done!"

# Install Boost
echo "Downloading Boost..."
LOCATION=$(curl -s https://api.github.com/repos/boostorg/boost/releases/latest | grep "zipball_url" | awk '{ print $2 }' | sed 's/,$//' | sed 's/"//g')
curl -L -o "$BASEDIR"/boost_latest.tar.xz "$LOCATION"
tar xf "$BASEDIR"/boost_latest.tar.bz2
cd "$BASEDIR"/boost_latest || exit 1
echo "Installing Boost..."
./bootstrap.sh
./b2 cxxflags=-fPIC link=static \
  --prefix="$BASEDIR"/boost \
  --build-dir="$BASEDIR"/boost_latest/build toolset=gcc \
  --with-program_options install \
  --with-filesystem \
  --with-test \
  --with-log
echo "Done!"

# Download TensorFlow, ArmNN, and FlatBuffers, then run generate_tensorflow_protobuf.sh
echo "Downloading ArmNN..."
git clone https://github.com/Arm-software/armnn "$BASEDIR"/armnn
echo "Downloading TensorFlow..."
git clone https://github.com/tensorflow/tensorflow.git "$BASEDIR"/tensorflow
cd "$BASEDIR"/tensorflow || exit 1
git checkout 590d6eef7e91a6a7392c8ffffb7b58f2e0c8bc6b
echo "Downloading FlatBuffers for TF..."
git clone https://github.com/google/flatbuffers.git ./flatbuffers
echo "Configuring TensorFlow and Protobuf"
./"$BASEDIR"/armnn/scripts/generate_tensorflow_protobuf.sh ../tensorflow-protobuf ../protobuf-host
echo "Done!"

# Download and install FlatBuffers
echo "Downloading "
git clone https://github.com/google/flatbuffers.git "$BASEDIR"/flatbuffers
echo "Downloading FlatBuffers..."
cd "$BASEDIR"/flatbuffers || exit 1
echo "Installing FlatBuffers..."
cmake -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-fPIC" \
  -DFLATBUFFERS_BUILD_FLATC=1 \
  -DCMAKE_INSTALL_PREFIX:PATH="$BASEDIR"/flatbuffers-1.12.0 -DFLATBUFFERS_BUILD_TESTS=0
make -j$NUM_CORES
echo "Done!"

#Install SWIG
echo "Downloading FlatBuffers..."
git clone https://github.com/swig/swig.git "$BASEDIR"/swig
cd "$BASEDIR"/swig || exit 1
echo "Configuring FlatBuffers..."
./autogen.sh && ./configure --prefix=/home/pi/armnn-tflite/swigtool/
echo "Installing FlatBuffers..."
make -j$NUM_CORES
make install -j$NUM_CORES
echo "Done!"

# Add lines to /etc/profile if necessary
echo "Updating \"/etc/profile\"..."
ADD_SWIG_PATH="export SWIG_PATH=/home/pi/armnn-tflite/swigtool/bin"
ADD_PATH="export PATH=$SWIG_PATH:$PATH"
if ! grep -Fxq "$ADD_SWIG_PATH" /etc/profile; then
  echo "$ADD_SWIG_PATH" >>/etc/profile
fi
if ! grep -Fxq "$ADD_PATH" /etc/profile; then
  echo "$ADD_PATH" >>/etc/profile
fi
source /etc/profile
echo "Done!"

# Build Arm NN
echo "Building ArmNN"
mkdir "$BASEDIR"/armnn/build
cd "$BASEDIR"/armnn/build || exit 1
echo "Making base libraries..."
cmake .. \
  -DARMCOMPUTE_ROOT="$BASEDIR"/ComputeLibrary \
  -DARMCOMPUTE_BUILD_DIR="$BASEDIR"/ComputeLibrary/build \
  -DTF_LITE_GENERATED_PATH="$BASEDIR"/tensorflow/tensorflow/lite/schema \
  -DFLATBUFFERS_ROOT="$BASEDIR"/flatbuffers \
  -DFLATBUFFERS_LIBRARY="$BASEDIR"/flatbuffers/libflatbuffers.a \
  -DFLATBUFFERS_INCLUDE_PATH="$BASEDIR"/flatbuffers/include \
  -DFLATC_DIR="$BASEDIR"/flatbuffers/build \
  -DDYNAMIC_BACKEND_PATHS="$BASEDIR"/armnn/src/dynamic/sample \
  -DSAMPLE_DYNAMIC_BACKEND=1 \
  -DBUILD_TF_LITE_PARSER=1 \
  -DBUILD_TF_PARSER=1 \
  -DARMCOMPUTENEON=1 \
  -DBUILD_TESTS=1 \
  -DARMNNREF=1
make -j$NUM_CORES

cp "$BASEDIR"/armnn/build/*.so "$BASEDIR"/armnn/

mkdir "$BASEDIR"/armnn-tflite/armnn/src/dynamic/sample/build
cd "$BASEDIR"/armnn-tflite/armnn/src/dynamic/sample/build || exit 1
echo "Making Boost libraries..."
cmake .. -DBOOST_ROOT="$BASEDIR"/boost \
  -DBoost_SYSTEM_LIBRARY="$BASEDIR"/boost/lib/libboost_system.a \
  -DBoost_FILESYSTEM_LIBRARY="$BASEDIR"/boost/lib/libboost_filesystem.a \
  -DARMNN_PATH="$BASEDIR"/armnn/libarmnn.so
echo "Installing Boost libraries..."
make -j$NUM_CORES
echo "Done!"

# Install PyArmNN
# Following instructions for "Standalone build" from:
# https://git.mlplatform.org/ml/armnn.git/tree/python/pyarmnn/README.md
SWIG_EXECUTABLE="$BASEDIR"/swigtool/bin/swig
ARMNN_INCLUDE="$BASEDIR"/armnn/include/
ARMNN_LIB="$BASEDIR"/armnn/build/
export SWIG_EXECUTABLE ARMNN_INCLUDE ARMNN_LIB

cd "$BASEDIR"/armnn/python/pyarmnn || exit 1
apt-get install -y python3.6-dev build-essential checkinstall libreadline-gplv2-dev libncursesw5-dev libssl-dev \
  libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libcblas-dev libhdf5-dev \
  libhdf5-serial-dev libatlas-base-dev libjasper-dev libqtgui4 libqt4-test

python3 setup.py clean --all
python3 swig_generate.py -v
python3 setup.py build_ext --inplace
python3 setup.py sdist
python3 setup.py bdist_wheel

python3 -m pip3 install dist/pyarmnn-21.0.0-cp37-cp37m-linux_armv7l.whl
python3 -m pip3 install opencv-python==3.4.6.27
