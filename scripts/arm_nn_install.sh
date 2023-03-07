#!/bin/bash -ex

if [ "$EUID" -ne 0 ]; then
  echo "Please run as root!"
  exit
fi

INSTALL_FLAG=1
if ! [ "$1" == "-i" ]; then
  INSTALL_FLAG=0
fi

cleanup() {
  local rv=$?

  kill -9 "$PID"
  printf "\r"

  if [ $rv -ne 0 ]; then
    printf "Error! Please see error logs:\n"
  else
    printf "Installation complete! Goodbye :)\n"
  fi

  exit $rv
}

spinner() {
  local spin="Waiting..."

  while :; do
    for i in $(seq 0 1 ${#spin}); do
      printf "%s" "${spin:i:1}"
      sleep 1
    done
    printf "\r                      \r"
  done

  return 0
}

disp_msg() {
  printf "\r\n%s\n" "$@" >&5
  return 0
}

echo_stderr() {
  printf "\nERR: %s\n" "$@" >&6
  exit 1
}

make_basedir() {
  # Create a base directory
  echo "Creating a base directory..."
  mkdir -m 777 -p armnn-install
  cd armnn-install || exit 1
  BASEDIR=$(pwd)
  export BASEDIR

  # Set up logging
  echo "Creating a new log file..."
  mkdir -m 777 -p logs
  NOW=$(date "+%d-%m-%Y_%H-%M-%S")
  LOG="$BASEDIR"/logs/out_$NOW.log
  ERR="$BASEDIR"/logs/err_$NOW.log
  date >>"$LOG"
  date >>"$ERR"
  chmod 777 "$LOG" "$ERR"
  echo "Done!"
  echo "Logs can be found at \"$BASEDIR/logs\""

  return 0
}

download_lib() {

  # Check if the user disabled installations
  if [ "$INSTALL_FLAG" -ne 0 ]; then
    return 0
  fi
  echo "Downloading $1..."
  local DIR="$BASEDIR/$1"

  # TODO: Git interation needs improvement
  if [ ! -d "$DIR" ]; then
    git clone "$2" "$3" "$DIR" || echo_stderr "git error when downloading $1"
    git -C "$DIR" submodule update --init || true
  else
    (git -C "$DIR" fetch && git -C "$DIR" merge) || true
    (git submodule update --recursive --remote --init) || true
  fi

  cd "$DIR" || return 1
  echo "Done!"

  return 0
}

run_prog() {

  # begin the installation process
  disp_msg "Installation will now begin!"

  # redirect stdout/stderr to log files
  exec >>"$LOG" 2>>"$ERR"

  # Cmake parallel processing flag
  NUM_CORES="$(($(nproc) - 1))"
  export MAKEFLAGS="--parallel $NUM_CORES"
  echo "Number of utilized cores: $NUM_CORES" >"$LOG" 2>"$ERR"

  # Increase virtual memory swapfile allocation from 100 to 1024
  SWAP_SIZE="CONF_SWAPSIZE=100"
  if ! grep -Fxq "$SWAP_SIZE" /etc/dphys-swapfile; then
    echo "Increasing virtual memory swapfile allocation from 100 to 1024..."
    sed -ie 's/CONF_SWAPSIZE=100/CONF_SWAPSIZE=1024/g' /etc/dphys-swapfile
    # Restart dphys
    /etc/init.d/dphys-swapfile stop && /etc/init.d/dphys-swapfile start
    echo "Done!"
  fi

  # Install various things
  disp_msg "Installing necessary tools..."
  echo "Checking for updates and installing various tools and packages..."
  apt-get update
  apt-get install -y scons cmake autoconf libtool libpcre3 libpcre3-dev build-essential checkinstall libncursesw5-dev \
    libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libhdf5-dev libhdf5-serial-dev libatlas-base-dev
  echo "Done!"

  # Install ComputeLib
  disp_msg "Installing ComputeLibrary..."
  download_lib "ComputeLibrary" "https://github.com/Arm-software/ComputeLibrary.git"
  echo "Installing ComputeLib"
  scons arch=arm64-v8a neon=1 extra_cxx_flags="-fPIC" opencl=1 embed_kernels=1 benchmark_tests=0 validation_tests=0 \
    -j$NUM_CORES
  echo "Done!"

  # Install Protobuf
  disp_msg "Installing Protobuf..."
  download_lib "protobuf" "-b v3.5.0" "https://github.com/google/protobuf.git"

  echo "Configuring Protobuf..."
  ./autogen.sh
  ./configure --prefix="$BASEDIR"/protobuf-host

  echo "Installing Protobuf..."
  make -j$NUM_CORES
  make install -j$NUM_CORES

  echo "Done!"

  # Install Boost
  disp_msg "Installing Boost..."
  download_lib "boost" "--recursive" "https://github.com/boostorg/boost.git"

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
  disp_msg "Installing TensorFlow..."

  download_lib "armnn" "https://github.com/Arm-software/armnn" ""

  download_lib "tensorflow" "https://github.com/tensorflow/tensorflow.git"
  git checkout 590d6eef7e91a6a7392c8ffffb7b58f2e0c8bc6b

  download_lib "tensorflow/flatbuffers" "https://github.com/google/flatbuffers.git"

  echo "Configuring TensorFlow and Protobuf"
  ./"$BASEDIR"/armnn/scripts/generate_tensorflow_protobuf.sh ../tensorflow-protobuf ../protobuf-host

  echo "Done!"

  # Download and install FlatBuffers
  disp_msg "Installing FlatBuffers..."

  download_lib "flatbuffers" "https://github.com/google/flatbuffers.git"

  echo "Installing FlatBuffers..."
  cmake -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-fPIC" \
    -DFLATBUFFERS_BUILD_FLATC=1 \
    -DCMAKE_INSTALL_PREFIX:PATH="$BASEDIR"/flatbuffers-1.12.0 -DFLATBUFFERS_BUILD_TESTS=0
  make -j$NUM_CORES

  echo "Done!"

  #Install SWIG
  disp_msg "Installing SWIG..."

  download_lib "swig" "https://github.com/swig/swig.git"

  echo "Configuring SWIG..."
  ./autogen.sh && ./configure --prefix=/home/pi/armnn-tflite/swigtool/

  echo "Installing SWIG..."
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
  disp_msg "Installing ArmNN..."

  echo "Building ArmNN"

  # shellcheck disable=SC2174
  mkdir -m 777 -p "$BASEDIR"/armnn/build
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

  # shellcheck disable=SC2174
  mkdir -m 777 -p "$BASEDIR"/armnn-tflite/armnn/src/dynamic/sample/build
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
  disp_msg "Setting up python libraries..."

  SWIG_EXECUTABLE="$BASEDIR"/swigtool/bin/swig
  ARMNN_INCLUDE="$BASEDIR"/armnn/include/
  ARMNN_LIB="$BASEDIR"/armnn/build/
  export SWIG_EXECUTABLE ARMNN_INCLUDE ARMNN_LIB

  cd "$BASEDIR"/armnn/python/pyarmnn || exit 1

  python3 setup.py clean --all
  python3 swig_generate.py -v
  python3 setup.py build_ext --inplace
  python3 setup.py sdist
  python3 setup.py bdist_wheel

  python3 -m pip3 install dist/pyarmnn-21.0.0-cp37-cp37m-linux_armv7l.whl
  python3 -m pip3 install opencv-python==3.4.6.27

  apt-get install libcblas-dev
  apt-get install libhdf5-dev
  apt-get install libhdf5-serial-dev
  apt-get install libatlas-base-dev
  apt-get install libjasper-dev
  apt-get install libqtgui4
  apt-get install libqt4-test

  return 0
}

# Init stuff
trap 'cleanup' EXIT
exec 5>&1; exec 6>&2;

# Start the spinner!
spinner &
PID=$!

# Set up root directory
make_basedir

# run the installer along with a simple spinner
run_prog
