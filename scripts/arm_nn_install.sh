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
    printf "Error! Please see error logs:\n" >&5
  else
    printf "Installation complete! Goodbye :)\n"
  fi

  exit $rv
}

spinner() {
  local spin="Waiting..."

  while :; do
    for i in $(seq 0 1 2); do
      printf "\r%s" "${spin:0:((8 + i))}"
      sleep 1
    done
  done

  return 0
}

disp_msg() {
  printf "\r%s\n" "$@" >&5
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
  cd armnn-install || return 1
  BASEDIR=$(pwd)
  export BASEDIR

  # Set up logging
  echo "Creating a new log file..."
  mkdir -m 777 -p logs
  cd "$BASEDIR"/logs || return 1
  mkdir -m 777 -p old_logs
  mv ./*.log logs/old_logs
  NOW=$(date "+%d-%m-%Y_%H-%M-%S")
  LOG="$BASEDIR"/logs/out_$NOW.log
  ERR="$BASEDIR"/logs/err_$NOW.log
  date >>"$LOG"
  date >>"$ERR"
  chmod 777 "$LOG" "$ERR"
  echo "Done!"
  echo "Logs can be found at \"$BASEDIR/logs\""

  cd "$BASEDIR" || return 1

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
    git clone "$2" "$DIR" || echo_stderr "git error when downloading $1"
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
  echo "Number of utilized cores: $NUM_CORES"

  # Increase virtual memory swapfile allocation from 100 to 1024
#  SWAP_SIZE="CONF_SWAPSIZE=100"
#  if ! grep -Fxq "$SWAP_SIZE" /etc/dphys-swapfile; then
#    echo "Increasing virtual memory swapfile allocation from 100 to 1024..."
#    sed -ie 's/CONF_SWAPSIZE=100/CONF_SWAPSIZE=1024/g' /etc/dphys-swapfile
#    # Restart dphys
#    /etc/init.d/dphys-swapfile stop && /etc/init.d/dphys-swapfile start
#    echo "Done!"
#  fi

  # Install various things
  disp_msg "Installing necessary tools..."
#  echo "Checking for updates and installing various tools and packages..."
#  apt-get update
#  apt-get install -y scons cmake swig autoconf libtool bison byacc libpcre3 libpcre3-dev build-essential checkinstall \
#    libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libhdf5-dev libhdf5-serial-dev \
#    libatlas-base-dev
#  echo "Done!"

  # Install ComputeLib
  disp_msg "Installing ComputeLibrary..."
#  download_lib "ComputeLibrary" "https://github.com/Arm-software/ComputeLibrary.git"
#  echo "Installing ComputeLib"
#  scons arch=arm64-v8a neon=1 extra_cxx_flags="-fPIC" opencl=1 embed_kernels=1 benchmark_tests=0 validation_tests=0 \
#    -j$NUM_CORES
#  echo "Done!"

  # Install Protobuf
  disp_msg "Installing Protobuf..."
  download_lib "protobuf" "https://github.com/google/protobuf.git"
  mkdir -m 777 build && (cd build || echo_stderr "error making directory $BASEDIR/protobuf/build")
  echo "Configuring Protobuf..."
  cmake .. -DCMAKE_INSTALL_PREFIX="$BASEDIR/protobuf-host"
  echo "Installing Protobuf..."
  make -j3
  make install -j3
  echo "Done!"

  # Download TensorFlow, ArmNN, and FlatBuffers, then run generate_tensorflow_protobuf.sh
#  disp_msg "Installing TensorFlow..."
#  download_lib "armnn" "https://github.com/Arm-software/armnn"
#  download_lib "tensorflow" "https://github.com/tensorflow/tensorflow.git"
#  download_lib "tensorflow/flatbuffers" "https://github.com/google/flatbuffers.git"
#  echo "Configuring TensorFlow and Protobuf"
#  cd "$BASEDIR"/tensorflow
#  ../armnn/scripts/generate_tensorflow_protobuf.sh ../tensorflow-protobuf ../protobuf-host
#  cd "$BASEDIR"
#  echo "Done!"

  # Download and install FlatBuffers
  disp_msg "Installing FlatBuffers..."
#  download_lib "flatbuffers" "https://github.com/google/flatbuffers.git"
#  echo "Installing FlatBuffers..."
#  mkdir -m 777 build && cd build
#  cmake .. -G "Unix Makefiles" \
#    -DCMAKE_BUILD_TYPE=Release \
#    -DCMAKE_CXX_FLAGS="-fPIC" \
#    -DFLATBUFFERS_BUILD_FLATC=1 \
#    -DCMAKE_INSTALL_PREFIX:PATH="$BASEDIR"/flatbuffers -DFLATBUFFERS_BUILD_TESTS=0
#  make -j$NUM_CORES
#  cd "$BASEDIR"
#  echo "Done!"

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
    -DBOOST_ROOT="$BASEDIR"/boost \
    -DTF_GENERATED_SOURCES="$BASEDIR"/tensorflow-protobuf \
    -DPROTOBUF_ROOT="$BASEDIR"/protobuf-host \
    -DFLATBUFFERS_ROOT="$BASEDIR"/flatbuffers \
    -DFLATC_DIR="$BASEDIR"/flatbuffers/build \
    -DFLATBUFFERS_INCLUDE_PATH="$BASEDIR"/flatbuffers/include \
    -DFLATBUFFERS_LIBRARY="$BASEDIR"/flatbuffers/build/libflatbuffers.a \
    -DDYNAMIC_BACKEND_PATHS="$BASEDIR"/armnn/src/dynamic/sample \
    -DBUILD_ARMNN_TFLITE_DELEGATE=ON \
    -DSAMPLE_DYNAMIC_BACKEND=1 \
    -DBUILD_TF_LITE_PARSER=1 \
    -DBUILD_PYTHON_SRC=1 \
    -DBUILD_TF_PARSER=1 \
    -DARMCOMPUTENEON=1 \
    -DBUILD_TESTS=1 \
    -DARMNNREF=1
  make -j$NUM_CORES

  cp "$BASEDIR"/armnn/build/*.so "$BASEDIR"/armnn/
  cp "$BASEDIR"/armnn/build/*.so /lib

  # shellcheck disable=SC2174
  mkdir -m 777 -p "$BASEDIR"/armnn-tflite/armnn/src/dynamic/sample/build
  cd "$BASEDIR"/armnn-tflite/armnn/src/dynamic/sample/build || exit 1
  echo "Making Boost libraries..."
  cmake .. -DARMNN_PATH="$BASEDIR"/armnn/libarmnn.so
  echo "Installing Boost libraries..."
  make -j$NUM_CORES
  echo "Done!"

  # Install PyArmNN
  # Following instructions for "Standalone build" from:
  # https://git.mlplatform.org/ml/armnn.git/tree/python/pyarmnn/README.md
  disp_msg "Setting up python libraries..."

  ARMNN_LIB="$BASEDIR"/armnn/build/
  ARMNN_INCLUDE="$BASEDIR"/armnn/include:"$BASEDIR"/armnn/profiling/common/include
  export ARMNN_INCLUDE ARMNN_LIB

  cd "$BASEDIR"/armnn/python/pyarmnn || exit 1

  sudo python3 setup.py clean --all
  sudo python3 swig_generate.py -v
  sudo python3 setup.py build_ext --inplace
  sudo python3 setup.py sdist
  sudo python3 setup.py bdist_wheel

  sudo python3 -m pip3 install dist/*.whl
  sudo python3 -m pip3 install opencv-python

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
exec 5>&1
exec 6>&2

# Start the spinner!
spinner &
PID=$!

# Set up root directory
make_basedir

# run the installer along with a simple spinner
run_prog
